"""
train_viral_titles_pro.py  ▸  End‑to‑end pipeline featuring **Prep → SFT → Reward Model → RLHF (DPO)**
so the model actively optimises for the `viral_score` metric, giving you
higher‑CTR YouTube titles/descriptions than pure SFT.

Quick start
-----------
```bash
pip install "trl[peft]>=0.8" transformers datasets duckdb pandas tiktoken accelerate boto3 sentence-transformers
python train_viral_titles_pro.py all
```

Stages & CLI
------------
| Stage | CLI arg | What it does |
|-------|---------|--------------|
| prep   | `prep`   | Build HF dataset with dynamic viral thresholds from DuckDB |
| sft    | `sft`    | LoRA fine‑tune base LLM on high‑viral examples |
| reward | `reward` | Train regression reward model predicting `viral_score` |
| rlhf   | `rlhf`   | DPO‑train policy against reward model |
| all    | `all`    | run all stages sequentially |
| regression_title | `regression_title` | Train a regression model to predict viral_score from title |
| regression_description | `regression_description` | Train a regression model to predict viral_score from description |

Environment variables
---------------------
| Var           | Default                    | Purpose |
|---------------|----------------------------|---------|
| `BASE_MODEL`  | `mistralai/Mistral-7B-Instruct-v0.3` | LLM to fine‑tune |
| `DB_PATH`     | `youtube_dataset.duckdb`   | DuckDB warehouse |
| `S3_BUCKET`   |                            | (opt) S3 bucket |
| `S3_KEY`      |                            | (opt) S3 key |
| AWS creds     | via `~/.aws` or env vars   | S3 access |

Outputs
-------
* `hf_dataset/` – Hugging Face dataset on disk
* `sft_ckpt/`   – LoRA adapter of SFT stage
* `rm_ckpt/`    – Reward model checkpoint
* `dpo_ckpt/`   – Final DPO policy checkpoint

"""
import os, sys, argparse, random, pathlib
from datetime import datetime
import duckdb, pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, RewardTrainer, DPOTrainer, SFTConfig, RewardConfig, DPOConfig, DataCollatorForCompletionOnlyLM
import torch
import boto3
from collections import Counter
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
# Import Trainer directly from transformers
from transformers import Trainer, TrainingArguments
# Import tqdm for progress bar configuration
import tqdm.auto as tqdm
# Import threading for RLock
import threading


# ─────────────── Config & Environment ───────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

# Windows-specific console configuration for proper ANSI handling
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # Enable ANSI escape sequence processing
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception as e:
        print(f"Warning: Could not enable ANSI colors in Windows console: {e}")

# Configure tqdm for proper display in Windows environment
tqdm.tqdm.set_lock(threading.RLock())  # For managing multiple concurrent bars
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
# Force tqdm to not use carriage returns for Windows compatibility
tqdm.tqdm.monitor_interval = 0  # Disable monitor thread that can cause issues

BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
DB_PATH    = pathlib.Path(os.getenv("DB_PATH", "youtube_dataset.duckdb"))
S3_BUCKET  = os.getenv("S3_BUCKET")
S3_KEY     = os.getenv("S3_KEY")
MAX_LEN    = 32
# Define separate max lengths for titles and descriptions
MAX_LEN_TITLE = 64
MAX_LEN_DESC = 256
SEED       = 42
random.seed(SEED)

def fetch_duckdb():
    if DB_PATH.exists(): return
    if S3_BUCKET and S3_KEY:
        boto3.client("s3").download_file(S3_BUCKET, S3_KEY, str(DB_PATH))
    if not DB_PATH.exists():
        sys.exit("❌ DuckDB warehouse not found. Run build_youtube_dataset.py first.")

# Create a new function to prepare regression dataset
def stage_prep_regression():
    """Prepare a dataset specifically for regression models with full range of viral scores."""
    print("▶️ Preparing regression dataset with full viral score distribution")
    fetch_duckdb()
    con = duckdb.connect(DB_PATH)
    
    # Count distribution of viral scores
    viral_dist = con.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE viral_score < 0.05) as vs_lt_05,
            COUNT(*) FILTER (WHERE viral_score >= 0.05 AND viral_score < 0.10) as vs_05_10,
            COUNT(*) FILTER (WHERE viral_score >= 0.10 AND viral_score < 0.15) as vs_10_15,
            COUNT(*) FILTER (WHERE viral_score >= 0.15 AND viral_score < 0.20) as vs_15_20,
            COUNT(*) FILTER (WHERE viral_score >= 0.20) as vs_20_plus,
            COUNT(*) FILTER (WHERE viral_score IS NOT NULL) as total_with_score
        FROM youtube_videos
        WHERE title IS NOT NULL AND description IS NOT NULL
    """).fetchone()
    
    print(f"  Viral score distribution:")
    print(f"    < 0.05: {viral_dist[0]:,}")
    print(f"    0.05 - 0.10: {viral_dist[1]:,}")
    print(f"    0.10 - 0.15: {viral_dist[2]:,}")
    print(f"    0.15 - 0.20: {viral_dist[3]:,}")
    print(f"    ≥ 0.20: {viral_dist[4]:,}")
    print(f"    Total with scores: {viral_dist[5]:,}")
    
    # Query all data without a viral score threshold
    df = con.execute("""
        SELECT title, description, viral_score
        FROM youtube_videos
        WHERE title IS NOT NULL 
          AND description IS NOT NULL
          AND viral_score IS NOT NULL
        ORDER BY random()
    """).df()
    con.close()
    
    print(f"✓ Loaded {len(df):,} rows for regression training (full viral score range)")
    
    # Sanity check for duplicates
    n_dupes = df.duplicated(subset=["title", "description"]).sum()
    if n_dupes > 0:
        print(f"❌ WARNING: Found {n_dupes} duplicate rows (by title+description). Dropping duplicates.")
        df = df.drop_duplicates(subset=["title", "description"]).reset_index(drop=True)
        print(f"  After dropping duplicates: {len(df):,} rows remain.")
    else:
        print("✓ No duplicate rows found (by title+description).")
    
    # Create the dataset
    ds = Dataset.from_pandas(df)
    split = ds.train_test_split(test_size=0.1, seed=SEED)
    split.save_to_disk("hf_dataset_reg")
    print("✅ Regression dataset saved ➜ hf_dataset_reg/")
    
    return True

# ───────────────────────── Prep Stage ─────────────────────────
def stage_prep():
    fetch_duckdb()
    con = duckdb.connect(DB_PATH)
    
    # Run diagnostic queries to understand data distribution
    print("📊 Data Distribution Analysis:")
    
    # Check total count
    total_count = con.execute("SELECT COUNT(*) FROM youtube_videos").fetchone()[0]
    print(f"  Total videos: {total_count:,}")
    
    # Check viral score distribution
    viral_score_dist = con.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE viral_score >= 0.20) as vs_20_plus,
            COUNT(*) FILTER (WHERE viral_score >= 0.205) as vs_205_plus,
            COUNT(*) FILTER (WHERE viral_score >= 0.21) as vs_21_plus,
            COUNT(*) FILTER (WHERE viral_score >= 0.215) as vs_215_plus,
            COUNT(*) FILTER (WHERE viral_score >= 0.22) as vs_22_plus
        FROM youtube_videos
    """).fetchone()
    
    print(f"  Viral score distribution:")
    print(f"    ≥ 0.20: {viral_score_dist[0]:,}")
    print(f"    ≥ 0.205: {viral_score_dist[1]:,}")
    print(f"    ≥ 0.21: {viral_score_dist[2]:,}")
    print(f"    ≥ 0.215: {viral_score_dist[3]:,}")
    print(f"    ≥ 0.22: {viral_score_dist[4]:,}")
    
    # Check date distribution
    date_dist = con.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE publishedAt >= (CURRENT_DATE - INTERVAL '1 years')) as last_1yr,
            COUNT(*) FILTER (WHERE publishedAt >= (CURRENT_DATE - INTERVAL '3 years')) as last_3yr,
            COUNT(*) FILTER (WHERE publishedAt >= (CURRENT_DATE - INTERVAL '5 years')) as last_5yr,
            COUNT(*) FILTER (WHERE publishedAt IS NOT NULL) as with_date,
            COUNT(*) FILTER (WHERE publishedAt IS NULL) as without_date
        FROM youtube_videos
    """).fetchone()
    
    print(f"  Date distribution:")
    print(f"    Last 1 year: {date_dist[0]:,}")
    print(f"    Last 3 years: {date_dist[1]:,}")
    print(f"    Last 5 years: {date_dist[2]:,}")
    print(f"    With date: {date_dist[3]:,}")
    print(f"    Without date: {date_dist[4]:,}")
    
    # Analyze combined filters
    combined_counts = con.execute("""
        SELECT
            COUNT(*) FILTER (
                WHERE viral_score >= 0.10 
                AND title IS NOT NULL AND description IS NOT NULL
                AND (publishedAt >= (CURRENT_DATE - INTERVAL '5 years') OR publishedAt IS NULL)
            ) as current_filter,
            COUNT(*) FILTER (
                WHERE viral_score >= 0.05
                AND title IS NOT NULL AND description IS NOT NULL
            ) as relaxed_filter
        FROM youtube_videos
    """).fetchone()
    
    print(f"  Combined filters:")
    print(f"    Current filter (VS≥0.10, with date handling): {combined_counts[0]:,}")
    print(f"    Relaxed filter (VS≥0.05, no date filter): {combined_counts[1]:,}")
    
    # Modified main query
    df = con.execute("""
        SELECT title, description, viral_score
        FROM youtube_videos
        WHERE title IS NOT NULL 
          AND description IS NOT NULL
        ORDER BY random()
    """).df()
    con.close()
    print(f"✓ loaded {len(df):,} rows for training (full range)")
    # Sanity check for duplicates
    n_dupes = df.duplicated(subset=["title", "description"]).sum()
    if n_dupes > 0:
        print(f"❌ WARNING: Found {n_dupes} duplicate rows (by title+description). Dropping duplicates.")
        df = df.drop_duplicates(subset=["title", "description"]).reset_index(drop=True)
        print(f"  After dropping duplicates: {len(df):,} rows remain.")
    else:
        print("✓ No duplicate rows found (by title+description).")
    # build prompts
    data = []
    for _, r in df.iterrows():
        title = r['title'] or ""
        desc  = (r['description'] or "").strip()[:300]

        prompt = (
            "### Instruction\n"  # keep format consistent across tasks
            "Write a viral YouTube title and a 300‑character description.\n\n"
            "### Input\n{\n  \"topic\": \"PLACEHOLDER\"\n}\n\n"
            "### Response\nTitle:"
        )

        resp = title + (f"\nDescription: {desc}" if desc else "")
        data.append({
            "prompt": prompt,
            "response": resp,
            "score": float(r['viral_score']),
            "title": title,
            "description": desc
        })
    ds = Dataset.from_list(data)
    ds = ds.shuffle(SEED)
    split = ds.train_test_split(test_size=0.1, seed=SEED)
    split.save_to_disk("hf_dataset")
    print("✅ Dataset saved ➜ hf_dataset/")

def sanity_check_dataset(dsd):
    """Sanity checks for the training dataset before SFT."""
    train = dsd["train"]
    n = len(train)
    if n < 1000:
        print(f"❌ ERROR: Training set too small: {n} examples.")
        sys.exit(1)
    # Check for empty prompts/responses
    n_empty_prompt = sum(not (ex["prompt"] and ex["prompt"].strip()) for ex in train)
    n_empty_resp = sum(not (ex["response"] and ex["response"].strip()) for ex in train)
    if n_empty_prompt > 0 or n_empty_resp > 0:
        print(f"❌ ERROR: Found {n_empty_prompt} empty prompts and {n_empty_resp} empty responses.")
        sys.exit(1)
    # Check for excessive duplication
    resp_counts = Counter(ex["response"] for ex in train)
    most_common_resp, resp_freq = resp_counts.most_common(1)[0]
    if resp_freq > n * 0.1:
        print(f"❌ ERROR: Most common response appears {resp_freq} times (>10% of data). Example: {most_common_resp[:80]}")
        sys.exit(1)
    # Check for short responses
    short_resps = sum(len(ex["response"]) < 10 for ex in train)
    if short_resps > n * 0.2:
        print(f"❌ ERROR: {short_resps} responses are shorter than 10 chars (>20% of data). Possible data issue.")
        sys.exit(1)
    # Check for train/test split
    if "test" not in dsd or len(dsd["test"]) == 0:
        print(f"❌ ERROR: No test split found or test set is empty.")
        sys.exit(1)
    print("✓ Dataset sanity checks passed.")

# ───────────────────────── SFT Stage ─────────────────────────
def stage_sft(epochs=3, bs=4):
    dsd = DatasetDict.load_from_disk("hf_dataset")
    sanity_check_dataset(dsd)

    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Training examples: {len(dsd['train']):,}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: 
        tok.pad_token = tok.eos_token

    def formatting_prompts_func(ex): 
        return ex['prompt'] + ex['response']
    
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0,
                                 bnb_4bit_compute_dtype=torch.float16)
    # SFT config
    args = SFTConfig(
        output_dir="sft_ckpt",
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,  # Reduced batch size
        gradient_accumulation_steps=16,  # Increased gradient accumulation
        learning_rate=2e-4,
        logging_steps=50,
        fp16=True,
        save_total_limit=2,
        report_to=[],
        max_length=MAX_LEN,
        gradient_checkpointing=True,
        model_init_kwargs={
            "quantization_config": bnb_cfg,
            "device_map":"auto",
            "torch_dtype":torch.float16
        }
    )
    peft_cfg = LoraConfig(
        r=8, 
        lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05, 
        bias="none",
        task_type="CAUSAL_LM"
    )
    # Define your response template (the string that always precedes the response)
    response_template = "### Response\nTitle:"

    # Create the collator
    collator = DataCollatorForCompletionOnlyLM(response_template, processing_class=tok)

    # Pass the collator to SFTTrainer
    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=args,
        train_dataset=dsd['train'],
        peft_config=peft_cfg,
        processing_class=tok,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    trainer.train(); 
    trainer.save_model("sft_ckpt")
    print("✅ SFT done ➜ sft_ckpt/")

# ───────────────────── Reward Model Stage ──────────────────────
def stage_reward(epochs=2):
    dsd = DatasetDict.load_from_disk("hf_dataset")
    base_rm = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(base_rm)
    def pref_pairs(ex):
        idx = sorted(range(len(ex['score'])), 
                     key=lambda i: ex['score'][i], 
                     reverse=True)
        chosen, rejected = [], []
        for i in range(0, len(idx)-1, 2):
            a,b = idx[i], idx[i+1]
            chosen.append(ex['prompt'][a]+ex['response'][a] if ex['score'][a]>=ex['score'][b] else ex['prompt'][b]+ex['response'][b])
            rejected.append(ex['prompt'][b]+ex['response'][b] if ex['score'][a]>=ex['score'][b] else ex['prompt'][a]+ex['response'][a])
        return {
            "chosen":chosen, 
            "rejected":rejected
        }
    tds = dsd['train'].map(
        pref_pairs, 
        batched=True, 
        batch_size=32, 
        remove_columns=dsd['train'].column_names
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_rm, 
        num_labels=1, 
        problem_type="regression"
    )
    rm_cfg = RewardConfig(
        output_dir="rm_ckpt",
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        learning_rate=1e-5,
        logging_steps=100,
        report_to=[],
        disable_dropout=True,
        max_length=MAX_LEN,
    )
    trainer = RewardTrainer(
        model=model,
        args=rm_cfg,
        train_dataset=tds,
        processing_class=tok,
    )
    trainer.train(); 
    trainer.save_model("rm_ckpt")
    print("✅ Reward model ➜ rm_ckpt/")

# ─────────────────────── RLHF (DPO) Stage ───────────────────────
def stage_rlhf(epochs=3, bs=1):
    off = pathlib.Path("offload_dir"); 
    off.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained("sft_ckpt")
    mk = {
        "device_map":"auto",
        "offload_folder":str(off),
        "offload_state_dict":True,
        "torch_dtype":torch.float16,
        "quantization_config":BitsAndBytesConfig(load_in_4bit=True)}
    policy = AutoModelForCausalLM.from_pretrained("sft_ckpt", **mk)
    reward = AutoModelForSequenceClassification.from_pretrained("rm_ckpt", **mk)
    dsd = DatasetDict.load_from_disk("hf_dataset")
    def mk_pair(ex): 
        return {
            "prompt":ex['prompt'],
            "chosen":ex['response'],
            "rejected":ex['response'][::-1]}
    dpo_ds = dsd['train'].map(mk_pair, remove_columns=dsd['train'].column_names)
    dpo_cfg = DPOConfig(
        output_dir="dpo_ckpt",
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        logging_steps=50,
        fp16=True,
        optim="adamw_torch_fused",
        beta=0.1,
        max_length=MAX_LEN,
        max_prompt_length=MAX_LEN//2,
        max_completion_length=MAX_LEN//2,
    )
    trainer = DPOTrainer(
        model=policy,
        ref_model=None,
        train_dataset=dpo_ds,
        args=dpo_cfg,
        processing_class=tok,
        reward_model=reward,
    )
    trainer.train(); 
    trainer.save_model("dpo_ckpt")

    print("✅ RLHF DPO done ➜ dpo_ckpt/")

def stage_regression(target="title", epochs=3, bs=32, model_ckpt="sentence-transformers/all-MiniLM-L12-v2", lr=2e-5, scheduler_type="linear"):
    """
    Train a regression model to predict viral_score from title or description.
    target: "title" or "description"
    epochs: number of training epochs
    bs: batch size
    model_ckpt: pretrained model to use
    lr: learning rate
    scheduler_type: learning rate scheduler (linear, cosine, constant_with_warmup)
    """
    print(f"▶️ Training regression model for: {target}")
    print(f"   Model: {model_ckpt}, Epochs: {epochs}, Batch size: {bs}")
    print(f"   Learning rate: {lr}, Scheduler: {scheduler_type}")
    
    # Use the dedicated regression dataset
    dsd = DatasetDict.load_from_disk("hf_dataset_reg")
    
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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, 
        num_labels=1, 
        problem_type="regression"
    )
    
    # Define the preprocessing function with appropriate max length
    max_len = MAX_LEN_TITLE if target == "title" else MAX_LEN_DESC
    print(f"Using max_length={max_len} for {target}")
    
    def preprocess_function(examples):
        return tok(examples["text"], padding="max_length", truncation=True, max_length=max_len)
    
    # Tokenize the datasets
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"{target}_reg_ckpt",
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=lr,
        lr_scheduler_type=scheduler_type,
        fp16=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to=[],
        # Configure progress bar behavior
        disable_tqdm=False,
        logging_steps=500,
        logging_first_step=True,
        logging_nan_inf_filter=True,
        log_level="error",
        logging_dir=None
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tok,
    )
    
    # Train the model
    trainer.train()
    trainer.save_model(f"{target}_reg_ckpt")
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

# ─────────────────────────── CLI ──────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prep", "prep_regression", "regression_title", "regression_description", "all"]);
    p.add_argument("--epochs", type=int, default=3);
    p.add_argument("--bs", type=int, default=32);
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate");
    p.add_argument("--scheduler", type=str, default="linear", 
                   choices=["linear", "cosine", "constant_with_warmup"], 
                   help="Learning rate scheduler");
    args = p.parse_args(); 
    
    print(f"Running stage: {args.stage}")
    if args.stage == "prep":
        stage_prep()
    if args.stage == "prep_regression":
        stage_prep_regression()
    if args.stage == "regression_title":
        stage_regression(target="title", epochs=args.epochs, bs=args.bs, 
                        lr=args.lr, scheduler_type=args.scheduler)
    if args.stage == "regression_description":
        stage_regression(target="description", epochs=args.epochs, bs=args.bs,
                        lr=args.lr, scheduler_type=args.scheduler)
    if args.stage == "all":
        stage_prep()
        stage_prep_regression()
        stage_regression(target="title", epochs=args.epochs, bs=args.bs,
                        lr=args.lr, scheduler_type=args.scheduler)
        stage_regression(target="description", epochs=args.epochs, bs=args.bs,
                        lr=args.lr, scheduler_type=args.scheduler)
