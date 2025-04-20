"""
Regression model training stage functions.
"""
import os
import torch
import random
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from datasets import DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

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

def stage_regression(target="title", epochs=3, bs=32, model_ckpt="sentence-transformers/all-mpnet-base-v2", 
                    lr=2e-5, scheduler_type="linear", weight_decay=0.01, warmup_ratio=0.1, 
                    use_pairwise=True, use_spearman_metric=True, patience=2,
                    dataset_path="hf_dataset_reg", gradient_accumulation_steps=4):
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
    """
    print(f"▶️ Training regression model for: {target}")
    print(f"   Model: {model_ckpt}, Epochs: {epochs}, Batch size: {bs}")
    print(f"   Learning rate: {lr}, Scheduler: {scheduler_type}")
    print(f"   Weight decay: {weight_decay}, Warmup ratio: {warmup_ratio}")
    print(f"   Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"   Using pairwise loss: {use_pairwise}")
    print(f"   Using Spearman metric for best model: {use_spearman_metric}")
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
        return {"text": text, "labels": float(example["viral_score"])}
    
    train_ds = dsd["train"].map(prepare_regression_example)
    test_ds = dsd["test"].map(prepare_regression_example)
    
    # Standardize the labels (mean=0, std=1)
    train_labels = [ex["labels"] for ex in train_ds]
    label_mean = sum(train_labels) / len(train_labels)
    label_std = (sum((x - label_mean) ** 2 for x in train_labels) / len(train_labels)) ** 0.5
    
    print(f"Label statistics - Mean: {label_mean:.4f}, Std: {label_std:.4f}")
    
    def normalize_labels(example):
        return {"labels": (example["labels"] - label_mean) / label_std, "text": example["text"]}
    
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
    
    def preprocess_function(examples):
        return tok(examples["text"], padding="max_length", truncation=True, max_length=max_len)
    
    # Tokenize the datasets
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)
    
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
    
    # Create custom compute_loss function for pairwise loss if needed
    if use_pairwise:
        # Create Trainer with pairwise loss
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=tok,
            callbacks=callbacks,
            compute_metrics=compute_metrics  # Add compute_metrics function
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
            processing_class=tok,
            callbacks=callbacks,
            compute_metrics=compute_metrics  # Add compute_metrics function
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
    
    # Save some sample predictions
    sample_indices = random.sample(range(len(y_pred)), min(10, len(y_pred)))
    print("\nSample predictions (original scale):")
    for idx in sample_indices:
        text = tokenized_test[idx]["text"]
        if len(text) > 50:
            text = text[:50] + "..."
        print(f"Text: '{text}'")
        print(f"True: {y_true_denorm[idx]:.4f}, Pred: {y_pred_denorm[idx]:.4f}")
        print("-" * 40)
        
    return True 