"""
Regression model training stage functions.
"""
import os
import torch
import random
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from datasets import DatasetDict
from collections import Counter
from torch.utils.data import WeightedRandomSampler, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from langdetect import detect, LangDetectException

from ..utils import SpearmanCallback, analyze_viral_score_distribution, fix_biased_dataset
from ..config import MAX_LEN_TITLE, MAX_LEN_DESC
# Import early stopping callback
from transformers import EarlyStoppingCallback
from ..utils import PairwiseRankingLoss
        
def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    Compute pairwise ranking loss for the model.
    The num_items_in_batch parameter is not used but included for compatibility 
    with Trainer's call signature.
    """
    pairwise_loss_fct = PairwiseRankingLoss()
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits.view(-1)
    # PairwiseRankingLoss already handles batch sizes internally
    loss = pairwise_loss_fct(logits, labels)
    return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    """
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    # Calculate MSE
    mse = mean_squared_error(labels, predictions)
    
    # Calculate Spearman correlation
    spearman = spearmanr(labels, predictions).correlation
    
    return {
        "eval_mse": mse,
        "eval_spearman": spearman
    }

def detect_language(text):
    """
    Detect language of the text and return the language code.
    Returns 'en' for English or error cases as fallback.
    """
    try:
        if not text or len(text.strip()) < 5:
            return 'en'
        lang = detect(text)
        return lang
    except LangDetectException:
        return 'en'

def create_balanced_language_sampler(dataset):
    """
    Creates a WeightedRandomSampler that balances samples across language groups.
    This gives more weight to underrepresented languages to ensure balanced training.
    
    Args:
        dataset: The dataset to sample from (must contain 'language' field)
        
    Returns:
        WeightedRandomSampler: Sampler that can be used with DataLoader
    """
    # Count occurrences of each language
    lang_counts = Counter([ex.get("language", "unknown") for ex in dataset])
    
    # Calculate weights - inverse to language frequency
    # (rarer languages get higher weights)
    weights = []
    for ex in dataset:
        lang = ex.get("language", "unknown")
        count = lang_counts[lang]
        # Weight is inversely proportional to frequency
        # Add 1 to avoid division by zero
        weight = 1.0 / (count + 1)
        weights.append(weight)
    
    # Create and return the sampler
    return WeightedRandomSampler(weights, len(dataset))

class LanguageAwareCollator:
    """
    Custom collator function that removes language field before creating tensors
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, examples):
        # Save language info separately
        languages = [example.pop("language") if "language" in example else "unknown" for example in examples]
        
        # Perform padding and create tensors
        batch = self.tokenizer.pad(examples, return_tensors="pt", padding=True)
        
        # Put languages back but not as a tensor field
        batch["language_metadata"] = languages
        
        return batch

# Custom Dataset for language-balanced sampling
class LanguageBalancedDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        # Make a list of indices
        self.indices = list(range(len(hf_dataset)))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual example
        example = self.dataset[self.indices[idx]]
        
        # Return a dict that doesn't include language in tensors
        result = {k: v for k, v in example.items() if k != "language"}
        
        # Add language field separately
        if "language" in example:
            result["language"] = example["language"]
            
        return result

# Custom Trainer that uses language-balanced sampling
class LanguageBalancedTrainer(Trainer):
    def __init__(self, language_sampler=None, *args, **kwargs):
        self.language_sampler = language_sampler
        
        # Extract tokenizer from kwargs
        tokenizer = kwargs.pop("tokenizer", None)
        
        # Create custom collator
        if tokenizer is not None:
            kwargs["data_collator"] = LanguageAwareCollator(tokenizer)
            
        super().__init__(*args, **kwargs)
    
    def get_train_dataloader(self):
        """
        Override to use weighted sampling for language balancing
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            return super().get_train_dataloader()
            
        # Create PyTorch Dataset
        torch_dataset = LanguageBalancedDataset(train_dataset)
        
        # Use default DataLoader if no language sampler
        if self.language_sampler is None:
            return super().get_train_dataloader()
        
        # Create DataLoader with the language sampler
        data_loader = DataLoader(
            torch_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self.language_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
        return data_loader

def stage_regression(target="title", epochs=3, bs=32, model_ckpt="sentence-transformers/all-mpnet-base-v2", 
                    lr=2e-5, scheduler_type="linear", weight_decay=0.01, warmup_ratio=0.1, 
                    use_pairwise=True, use_spearman_metric=True, patience=2,
                    dataset_path="hf_dataset_reg", gradient_accumulation_steps=4,
                    use_language_features=True, use_balanced_sampling=True):
    """
    Train a regression model to predict viral_score from title or description.
    
    Args:
        target: "title" or "description"
        epochs: Number of training epochs
        bs: Batch size
        model_ckpt: Pretrained model to use
        lr: Learning rate
        scheduler_type: Learning rate scheduler
        weight_decay: Weight decay for regularization
        warmup_ratio: Portion of training to use for warmup
        use_pairwise: Use pairwise ranking loss instead of MSE (default: True)
        use_spearman_metric: Use Spearman correlation as metric for best model (default: True)
        patience: Number of evaluation calls with no improvement after which training will be stopped (default: 2)
        dataset_path: Path to the dataset
        gradient_accumulation_steps: Number of steps to accumulate gradients before optimizer step (default: 4)
        use_language_features: Whether to add language detection features (default: True)
        use_balanced_sampling: Whether to use language-balanced sampling (default: True)
    """
    print(f"▶️ Training regression model for: {target}")
    print(f"   Model: {model_ckpt}, Epochs: {epochs}, Batch size: {bs}")
    print(f"   Learning rate: {lr}, Scheduler: {scheduler_type}")
    print(f"   Weight decay: {weight_decay}, Warmup ratio: {warmup_ratio}")
    print(f"   Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"   Using pairwise loss: {use_pairwise}")
    print(f"   Using Spearman metric for best model: {use_spearman_metric}")
    print(f"   Using language detection features: {use_language_features}")
    print(f"   Using language-balanced sampling: {use_balanced_sampling}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Using dataset path: {dataset_path}")
    
    # First, analyze the viral score distribution to check for bias
    bias_detected, _ = analyze_viral_score_distribution(dataset_path)
    
    # If bias is detected, use the fixed dataset
    fixed_dataset_path = dataset_path
    if bias_detected:
        print("⚠️ Label bias detected. Using fixed dataset with Gaussian noise added.")
        fixed_dataset_path = fix_biased_dataset(dataset_path, f"{dataset_path}_fixed")
        dataset_path = fixed_dataset_path
    
    # Use the appropriate dataset
    dsd = DatasetDict.load_from_disk(dataset_path)
    
    # Extract the first 5 examples to check their contents
    first_examples = list(dsd["train"].select(range(5)))
    print(f"First examples ({target}):")
    for i, ex in enumerate(first_examples):
        print(f"Example {i}: {target}='{ex[target]}', score={ex['viral_score']:.4f}")
    
    # Prepare a clean dataset with only text and labels fields
    def prepare_regression_example(example):
        text = example[target]
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        result = {"text": text, "labels": float(example["viral_score"])}
        
        # Add language detection if enabled
        if use_language_features or use_balanced_sampling:
            lang = detect_language(text)
            result["language"] = lang
            
        return result
    
    print("Preparing dataset with regression examples...")
    train_ds = dsd["train"].map(prepare_regression_example)
    test_ds = dsd["test"].map(prepare_regression_example)
    
    # If using language features or balanced sampling, show language distribution
    if use_language_features or use_balanced_sampling:
        lang_counts = {}
        for ex in train_ds:
            lang = ex.get("language", "unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # Display language distribution
        print("\nLanguage distribution in training data:")
        for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = 100 * count / len(train_ds)
            print(f"  {lang}: {count:,} samples ({percentage:.2f}%)")
    
    # Standardize the labels (mean=0, std=1)
    train_labels = [ex["labels"] for ex in train_ds]
    label_mean = sum(train_labels) / len(train_labels)
    label_std = (sum((x - label_mean) ** 2 for x in train_labels) / len(train_labels)) ** 0.5
    
    print(f"Label statistics - Mean: {label_mean:.4f}, Std: {label_std:.4f}")
    
    def normalize_labels(example):
        result = {"labels": (example["labels"] - label_mean) / label_std, "text": example["text"]}
        
        # Preserve language field if present
        if "language" in example:
            result["language"] = example["language"]
            
        return result
    
    # Apply normalization
    train_ds = train_ds.map(normalize_labels)
    test_ds = test_ds.map(normalize_labels)
    
    # Configure the tokenizer
    tok = AutoTokenizer.from_pretrained(model_ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # Create the model
    if use_pairwise:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_ckpt, 
            num_labels=1, 
            problem_type="single_label_classification"  # Changed for pairwise loss
        )
        # Configure dropout for regularization if available in config
        if hasattr(model.config, "hidden_dropout_prob"):
            model.config.hidden_dropout_prob = 0.1
        if hasattr(model.config, "attention_probs_dropout_prob"):
            model.config.attention_probs_dropout_prob = 0.1
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_ckpt, 
            num_labels=1, 
            problem_type="regression"
        )
        # Configure dropout for regularization if available in config
        if hasattr(model.config, "hidden_dropout_prob"):
            model.config.hidden_dropout_prob = 0.1
        if hasattr(model.config, "attention_probs_dropout_prob"):
            model.config.attention_probs_dropout_prob = 0.1
    
    # Define the preprocessing function with appropriate max length
    max_len = MAX_LEN_TITLE if target == "title" else MAX_LEN_DESC
    print(f"Using max_length={max_len} for {target}")
    
    # Common language groups to add special tokens for
    common_lang_groups = {
        "slavic": ["ru", "uk", "be", "bg", "pl", "cs", "sk", "sr", "hr", "sl"],
        "cjk": ["zh", "ja", "ko"],
        "indic": ["hi", "bn", "te", "ta", "mr", "ml"],
        "arabic": ["ar", "fa", "ur"],
        "latin": ["en", "es", "fr", "pt", "it", "de", "nl"],
    }
    
    # Create reverse mapping from language to group
    lang_to_group = {}
    for group, langs in common_lang_groups.items():
        for lang in langs:
            lang_to_group[lang] = group
    
    def preprocess_function(examples):
        # Store language info separately
        languages = None
        if "language" in examples:
            languages = examples["language"]  # Save before tokenization
        
        # Basic tokenization (don't include language field in tokenization)
        features = {}
        for k, v in examples.items():
            if k != "language":
                features[k] = v
        
        tokenized = tok(features["text"], padding="max_length", truncation=True, max_length=max_len)
        
        # Add language info back to the features
        if languages is not None:
            tokenized["language"] = languages
        
        # If language features are enabled, modify input text to include language marker
        if use_language_features and languages is not None:
            # Add language group info to text - but don't modify tokenized ids
            for i, lang in enumerate(languages):
                # Get language group (default to "other" if not found)
                lang_group = lang_to_group.get(lang, "other")
                
                # Add language group as prefix to text (for displaying only)
                examples["text"][i] = f"[{lang_group}] {examples['text'][i]}"
                
        return tokenized
    
    # Tokenize the datasets
    print("Tokenizing datasets...")
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)
    
    # Create language sampler for balanced training if enabled
    language_sampler = None
    if use_balanced_sampling:
        language_sampler = create_balanced_language_sampler(tokenized_train)
        print("Created language-balanced sampler for more diverse training")
    
    # Set up training arguments
    metric_for_best_model = "eval_loss"  # Default
    if use_spearman_metric:
        metric_for_best_model = "eval_spearman"
        
    training_args = TrainingArguments(
        output_dir=f"{target}_reg_ckpt",
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=lr,
        lr_scheduler_type=scheduler_type,
        fp16=True,
        fp16_full_eval=True,  # Safer mixed-precision eval
        gradient_accumulation_steps=gradient_accumulation_steps,  # Use the provided value
        max_grad_norm=1.0,  # Gradient clipping to avoid NaN
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=(metric_for_best_model == "eval_spearman"),  # Higher is better for Spearman
        report_to=[],
        # Configure progress bar behavior
        disable_tqdm=False,
        logging_steps=500,
        logging_first_step=True,
        logging_nan_inf_filter=True,
        log_level="error",
        logging_dir=None,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio
    )
    

    # Define callbacks list
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    
    # Add Spearman callback if requested
    if use_spearman_metric:
        spearman_callback = SpearmanCallback(tokenized_test, tok)
        callbacks.append(spearman_callback)
    
    # Create a list of features that should not be converted to tensors
    # to avoid the "Unable to create tensor" error
    non_tensor_features = ["language"]
    
    # Filter out non-tensor features from datasets to avoid errors
    def filter_non_tensor_features(dataset):
        return dataset.remove_columns([col for col in non_tensor_features if col in dataset.column_names])
    
    # Apply filtering only for standard trainer that doesn't have custom collator
    if not use_balanced_sampling:
        tokenized_train = filter_non_tensor_features(tokenized_train)
        tokenized_test = filter_non_tensor_features(tokenized_test)
    
    # Choose the appropriate trainer based on whether we're using language balancing
    if use_balanced_sampling:
        # Create custom trainer with language balancing
        if use_pairwise:
            # Create LanguageBalancedTrainer with pairwise loss
            trainer = LanguageBalancedTrainer(
                language_sampler=language_sampler,
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                tokenizer=tok,
                callbacks=callbacks,
                compute_metrics=compute_metrics
            )
            # Set compute_loss method directly on trainer instance
            trainer.compute_loss = compute_loss
        else:
            # Create standard LanguageBalancedTrainer
            trainer = LanguageBalancedTrainer(
                language_sampler=language_sampler,
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                tokenizer=tok,
                callbacks=callbacks,
                compute_metrics=compute_metrics
            )
    else:
        # Use standard trainer
        if use_pairwise:
            # Create Trainer with pairwise loss
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                tokenizer=tok,
                callbacks=callbacks,
                compute_metrics=compute_metrics
            )
            # Set compute_loss method directly on trainer instance
            trainer.compute_loss = compute_loss
        else:
            # Create standard Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                tokenizer=tok,
                callbacks=callbacks,
                compute_metrics=compute_metrics
            )
    
    # Train the model
    trainer.train()
    
    # Save the final model - if using Spearman metric, it will already have saved the best one
    if not use_spearman_metric or not hasattr(trainer, "best_model_checkpoint"):
        trainer.save_model(f"{target}_reg_ckpt")
    else:
        # Copy the best model to the final output directory
        best_checkpoint = trainer.best_model_checkpoint
        if best_checkpoint and os.path.exists(best_checkpoint):
            print(f"Using best checkpoint: {best_checkpoint}")
            model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)
            model.save_pretrained(f"{target}_reg_ckpt")
            tok.save_pretrained(f"{target}_reg_ckpt")
    
    print(f"✅ Regression model saved ➜ {target}_reg_ckpt/")
    
    # Evaluate on test set using trainer.predict
    print("Running inference on test set...")
    predictions = trainer.predict(tokenized_test)
    
    # Extract predictions and ground truth
    y_pred = predictions.predictions.squeeze()
    y_true = predictions.label_ids
    
    # Denormalize predictions for proper metrics
    y_pred_denorm = y_pred * label_std + label_mean
    y_true_denorm = y_true * label_std + label_mean
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)  # Normalized MSE
    mse_denorm = mean_squared_error(y_true_denorm, y_pred_denorm)  # Denormalized MSE
    spearman = spearmanr(y_true, y_pred).correlation
    
    print(f"Test MSE (normalized): {mse:.6f}")
    print(f"Test MSE (original scale): {mse_denorm:.6f}")
    print(f"Spearman correlation: {spearman:.4f}")
    
    # Language-specific analysis if language features enabled
    if use_language_features or use_balanced_sampling:
        # Analyze performance by language group
        languages = [ex.get("language", "unknown") for ex in test_ds]
        lang_groups = {}
        
        for idx, lang in enumerate(languages):
            # Map to language group
            group = lang_to_group.get(lang, "other")
            if group not in lang_groups:
                lang_groups[group] = {"true": [], "pred": []}
            
            lang_groups[group]["true"].append(y_true_denorm[idx])
            lang_groups[group]["pred"].append(y_pred_denorm[idx])
        
        print("\nPerformance by language group:")
        for group, data in lang_groups.items():
            if len(data["true"]) < 10:  # Skip groups with too few samples
                continue
            
            group_mse = mean_squared_error(data["true"], data["pred"])
            group_spearman = spearmanr(data["true"], data["pred"]).correlation
            
            print(f"  {group}: {len(data['true'])} samples, MSE: {group_mse:.6f}, Spearman: {group_spearman:.4f}")
    
    # Save some sample predictions
    sample_indices = random.sample(range(len(y_pred)), min(10, len(y_pred)))
    print("\nSample predictions (original scale):")
    for idx in sample_indices:
        text = tokenized_test[idx]["text"] if isinstance(tokenized_test[idx]["text"], str) else tokenized_test[idx]["text"][0]
        if len(text) > 50:
            text = text[:50] + "..."
        
        # Include language if available
        lang_info = ""
        if (use_language_features or use_balanced_sampling) and hasattr(test_ds[idx], "get") and "language" in test_ds[idx]:
            lang_info = f" [Lang: {test_ds[idx]['language']}]"
            
        print(f"Text: '{text}'{lang_info}")
        print(f"True: {y_true_denorm[idx]:.4f}, Pred: {y_pred_denorm[idx]:.4f}")
        print("-" * 40)
        
    return True 