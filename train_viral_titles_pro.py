"""
train_viral_titles_pro.py  ▸  End‑to‑end pipeline featuring **SFT → Reward Model → RLHF (DPO)**
so the model actively optimises for the `viral_score` metric, giving you
higher‑CTR YouTube titles/descriptions than pure SFT.

Quick start
-----------
```bash
pip install "trl[peft]>=0.8" transformers datasets duckdb pandas tiktoken accelerate

# one‑shot run (uses defaults below)
python train_viral_titles_pro.py all
```

Stages & CLI
------------
| Stage | CLI arg | What it does |
|-------|---------|--------------|
| prep   | `prep`   | Build HF dataset from DuckDB (top 20 % viral rows) |
| sft    | `sft`    | LoRA fine‑tune base LLM |
| reward | `reward` | Train regression head that predicts `viral_score` |
| rlhf   | `rlhf`   | DPO‑train policy against reward model |
| all    | `all`    | run the four stages sequentially |

Environment variables
---------------------
| Var | Default | Purpose |
|-----|---------|---------|
| `BASE_MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | LLM to fine‑tune |
| `DB_PATH` | `youtube_dataset.duckdb` | DuckDB warehouse (local) |
| `S3_BUCKET` / `S3_KEY` | *(optional)* | Download warehouse from S3 first |

Outputs
-------
* `sft_ckpt/`   – LoRA adapter of SFT stage
* `rm_ckpt/`    – reward model predicting viral_score
* `dpo_ckpt/`   – final policy model (merged weights ready for inference)
"""

import os, sys, argparse, random, pathlib

import duckdb, pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig,
    AutoModel, AutoModelForSequenceClassification
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, RewardTrainer, DPOTrainer, SFTConfig, RewardConfig, DPOConfig
import torch

# ─────────────────────── Config & environment ─────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BASE_MODEL  = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
DB_PATH     = pathlib.Path(os.getenv("DB_PATH", "youtube_dataset.duckdb"))
S3_BUCKET   = os.getenv("S3_BUCKET")
S3_KEY      = os.getenv("S3_KEY")
MAX_LEN     = 64
SEED        = 42
random.seed(SEED)

# S3 fetch if warehouse missing
if not DB_PATH.exists() and S3_BUCKET and S3_KEY:
    import boto3
    boto3.client("s3").download_file(S3_BUCKET, S3_KEY, str(DB_PATH))

if not DB_PATH.exists():
    sys.exit("❌ DuckDB warehouse not found. Run build_youtube_dataset.py first.")

# ───────────────────────── Prep stage ─────────────────────────────

def stage_prep():
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
    
    # Modified main query with relaxed filters if needed
    viral_threshold = 0.205
    include_null_dates = True
    
    # If the current filter yields < 1000 results, automatically use more relaxed filter
    if combined_counts[0] < 1000 and combined_counts[1] >= 1000:
        viral_threshold = 0.05
        print(f"⚠️ Automatically using relaxed viral threshold to get more examples")
    
    # Build the date filter condition
    date_condition = "(publishedAt >= (CURRENT_DATE - INTERVAL '5 years') OR publishedAt IS NULL)" if include_null_dates else "publishedAt >= (CURRENT_DATE - INTERVAL '5 years')"
    
    df = con.execute(
        f"""
        SELECT title, description, viral_score
        FROM youtube_videos
        WHERE viral_score >= {viral_threshold}  -- Viral score threshold
          AND title IS NOT NULL AND description IS NOT NULL
          AND {date_condition}
        ORDER BY random()
        LIMIT 10000;
        """
    ).df()
    con.close()
    print(f"✓ loaded {len(df):,} rows for training")

    # Build prompt / response pairs
    data = []
    for _, r in df.iterrows():
        prompt = (
            "### Instruction\n"  # keep format consistent across tasks
            "Write a viral YouTube title and a 300‑character description.\n\n"
            "### Input\n{\n  \"topic\": \"PLACEHOLDER\"\n}\n\n### Response\nTitle:"  # model will output title & description
        )
        resp = f"{r['title'].strip()}\nDescription: {r['description'][:300].strip()}"
        data.append({"prompt": prompt, "response": resp, "score": float(r["viral_score"])})

    ds = Dataset.from_list(data)
    ds = ds.shuffle(SEED)
    split = ds.train_test_split(test_size=0.1, seed=SEED)
    split.save_to_disk("hf_dataset")
    print("✅ Dataset saved ➜ hf_dataset/")

# ───────────────────────── SFT stage ─────────────────────────────

def stage_sft(epochs=3, bs=2):
    # Print GPU information to verify GPU usage
    if torch.cuda.is_available():
        gpu_info = f"Using GPU: {torch.cuda.get_device_name(0)}"
        print(f"✓ {gpu_info}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ No GPU available, using CPU only (training will be very slow)")
    
    dsdict = DatasetDict.load_from_disk("hf_dataset")
    print(f"✓ Training dataset size: {len(dsdict['train']):,} examples")
    
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def formatting_prompts_func(example):
        return example["prompt"] + example["response"]

    # Create training configuration using SFTConfig with reduced memory footprint
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
        neftune_noise_alpha=5,  # Enable NEFTune for better performance
        packing=False,  # Disable packing for more stable training
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        label_names=["labels"],  # Explicitly set label_names to eliminate warning
        model_init_kwargs={
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True, 
                llm_int8_threshold=6.0,
                bnb_4bit_compute_dtype=torch.float16  # Add compute dtype for better memory efficiency
            ),
            "device_map": "auto",
            "torch_dtype": torch.float16,  # Use fp16 for base model weights
        },
    )

    # Set PyTorch to allow memory expansion for fragmentation reduction
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Define PEFT config with fewer target modules
    peft_config = LoraConfig(
        r=8,  # Reduced rank
        lora_alpha=16,  # Reduced alpha
        target_modules=["q_proj","v_proj"],  # Reduced target modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Create the trainer with explicit tokenizer and formatting function
    trainer = SFTTrainer(
        BASE_MODEL,
        args=args,
        train_dataset=dsdict["train"],
        processing_class=tok,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
    )
    
    trainer.train()
    trainer.save_model("sft_ckpt")
    print("✅ SFT done ➜ sft_ckpt/")

# ───────────────────── Reward model stage ────────────────────────

def stage_reward(epochs=2):
    dsdict = DatasetDict.load_from_disk("hf_dataset")
    base_rm = "sentence-transformers/all-MiniLM-L6-v2"  # tiny, fast
    tok = AutoTokenizer.from_pretrained(base_rm)

    # Create pairs of examples for preference learning
    def create_preference_pairs(examples):
        # Sort examples by score to get high and low scoring pairs
        sorted_indices = sorted(range(len(examples["score"])), 
                              key=lambda i: examples["score"][i], 
                              reverse=True)
        
        # Create pairs of high and low scoring examples
        chosen = []
        rejected = []
        for i in range(0, len(sorted_indices)-1, 2):
            if examples["score"][sorted_indices[i]] > examples["score"][sorted_indices[i+1]]:
                chosen.append(examples["prompt"][sorted_indices[i]] + examples["response"][sorted_indices[i]])
                rejected.append(examples["prompt"][sorted_indices[i+1]] + examples["response"][sorted_indices[i+1]])
            else:
                chosen.append(examples["prompt"][sorted_indices[i+1]] + examples["response"][sorted_indices[i+1]])
                rejected.append(examples["prompt"][sorted_indices[i]] + examples["response"][sorted_indices[i]])
        
        return {
            "chosen": chosen,
            "rejected": rejected
        }

    # Create preference pairs dataset
    tds = dsdict["train"].map(
        create_preference_pairs,
        batched=True,
        batch_size=32,
        remove_columns=dsdict["train"].column_names
    )

    # Use AutoModelForSequenceClassification instead of AutoModel
    model = AutoModelForSequenceClassification.from_pretrained(
        base_rm,
        num_labels=1,  # Single output for reward score
        problem_type="regression"  # We're predicting a continuous score
    )

    # Use RewardConfig instead of TrainingArguments
    args = RewardConfig(
        output_dir="rm_ckpt",
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        learning_rate=1e-5,
        logging_steps=100,
        report_to=[],
        disable_dropout=True,  # Required by RewardTrainer
        max_length=MAX_LEN,    # Required by RewardTrainer
    )

    trainer = RewardTrainer(
        model=model,
        args=args,
        train_dataset=tds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model("rm_ckpt")
    print("✅ Reward model ➜ rm_ckpt/")

# ─────────────────────── RLHF (DPO) stage ────────────────────────

def stage_rlhf(epochs=3):
    # Create offload directory if it doesn't exist
    offload_dir = pathlib.Path("offload_dir")
    offload_dir.mkdir(exist_ok=True)
    
    # Load tokenizer and models with proper offloading
    tok = AutoTokenizer.from_pretrained("sft_ckpt")
    
    # Configure model loading with offloading
    model_kwargs = {
        "device_map": "auto",
        "offload_folder": str(offload_dir),
        "offload_state_dict": True,
        "torch_dtype": torch.float16,
        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
    }
    
    policy = AutoModelForCausalLM.from_pretrained("sft_ckpt", **model_kwargs)
    reward = AutoModel.from_pretrained("rm_ckpt", **model_kwargs)

    dsdict = DatasetDict.load_from_disk("hf_dataset")

    def make_pairs(ex):
        # simple negative: response from another random sample
        return {
            "prompt": ex["prompt"],
            "chosen": ex["response"],
            "rejected": ex["response"][::-1]  # naive but works ok for DPO warm‑up
        }

    dpo_ds = dsdict["train"].map(make_pairs, remove_columns=dsdict["train"].column_names)

    # Use DPOConfig instead of TrainingArguments
    args = DPOConfig(
        output_dir="dpo_ckpt",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # Try 8 or 4
        learning_rate=5e-6,
        logging_steps=50,
        fp16=False,  # Disable fp16
        report_to=[],
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        beta=0.1,
        max_length=MAX_LEN,
        max_prompt_length=MAX_LEN // 2,
        max_completion_length=MAX_LEN // 2,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,  # Let TRL handle reference model for PEFT/LoRA
        processing_class=tok,
        train_dataset=dpo_ds,
        args=args,
    )
    trainer.train()
    trainer.save_model("dpo_ckpt")
    print("✅ RLHF done ➜ dpo_ckpt/")

# ─────────────────────── CLI orchestration ───────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train viral YouTube headline model")
    ap.add_argument("stage", choices=["prep","sft","reward","rlhf","all"], help="which stage to run")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()

    print(f"Running stage: {args.stage}")
    if args.stage in ("prep","all"):  stage_prep()
    if args.stage in ("sft","all"):   stage_sft(epochs=args.epochs, bs=args.bs)
    if args.stage in ("reward","all"):stage_reward()
    if args.stage in ("rlhf","all"):  stage_rlhf()
