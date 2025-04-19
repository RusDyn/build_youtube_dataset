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
from trl import SFTTrainer, RewardTrainer, DPOTrainer, SFTConfig, RewardConfig, DPOConfig
import torch
import boto3

# ─────────────── Config & Environment ───────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
DB_PATH    = pathlib.Path(os.getenv("DB_PATH", "youtube_dataset.duckdb"))
S3_BUCKET  = os.getenv("S3_BUCKET")
S3_KEY     = os.getenv("S3_KEY")
MAX_LEN    = 32
SEED       = 42
random.seed(SEED)

def fetch_duckdb():
    if DB_PATH.exists(): return
    if S3_BUCKET and S3_KEY:
        boto3.client("s3").download_file(S3_BUCKET, S3_KEY, str(DB_PATH))
    if not DB_PATH.exists():
        sys.exit("❌ DuckDB warehouse not found. Run build_youtube_dataset.py first.")

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
        WHERE viral_score >= {viral_threshold}
          AND title IS NOT NULL AND description IS NOT NULL
          AND {date_condition}
        ORDER BY random()
        LIMIT 10000;
    """).df()
    con.close()
    print(f"✓ loaded {len(df):,} rows for training (threshold={viral_threshold})")
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
        data.append({"prompt": prompt, "response": resp, "score": float(r['viral_score'])})
    ds = Dataset.from_list(data)
    ds = ds.shuffle(SEED)
    split = ds.train_test_split(test_size=0.1, seed=SEED)
    split.save_to_disk("hf_dataset")
    print("✅ Dataset saved ➜ hf_dataset/")

# ───────────────────────── SFT Stage ─────────────────────────
def stage_sft(epochs=3, bs=4):
    dsd = DatasetDict.load_from_disk("hf_dataset")

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
    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=args,
        train_dataset=dsd['train'],
        peft_config=peft_cfg,
        processing_class=tok,
        formatting_func=formatting_prompts_func,
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

# ─────────────────────────── CLI ──────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prep","sft","reward","rlhf","all"]);
    p.add_argument("--epochs", type=int, default=3);
    p.add_argument("--bs", type=int, default=4);
    args = p.parse_args(); 
    
    print(f"Running stage: {args.stage}")
    if args.stage in ("prep","all"):    stage_prep()
    if args.stage in ("sft","all"):     stage_sft(epochs=args.epochs, bs=args.bs)
    if args.stage in ("reward","all"):  stage_reward(epochs=args.epochs)
    if args.stage in ("rlhf","all"):    stage_rlhf(epochs=args.epochs, bs=args.bs)
