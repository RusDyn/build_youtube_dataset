#!/usr/bin/env python
"""
Viral YouTube Titles Training Pipeline
=====================================

End‑to‑end pipeline featuring **Prep → SFT → Reward Model → RLHF (DPO)**
so the model actively optimises for the `viral_score` metric, giving you
higher‑CTR YouTube titles/descriptions than pure SFT.

Quick start
-----------
```bash
pip install "trl[peft]>=0.8" transformers datasets duckdb pandas tiktoken accelerate boto3 sentence-transformers
python main.py all
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
| analyze_labels | `analyze_labels` | Analyze the distribution of viral scores in the dataset |
| fix_labels | `fix_labels` | Fix a dataset with biased viral scores by adding small Gaussian noise to the labels |

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
* `hf_dataset/` – Hugging Face dataset on disk
* `sft_ckpt/`   – LoRA adapter of SFT stage
* `rm_ckpt/`    – Reward model checkpoint
* `dpo_ckpt/`   – Final DPO policy checkpoint
"""
import sys
import argparse
import threading
import tqdm.auto as tqdm
from viral_titles import (
    configure_windows_console,
    stage_prep,
    stage_prep_regression,
    stage_regression,
    analyze_viral_score_distribution,
)

# Configure Windows console for ANSI colors
configure_windows_console()

# Configure tqdm for proper display
tqdm.tqdm.set_lock(threading.RLock())  # For managing multiple concurrent bars
tqdm.tqdm.monitor_interval = 0  # Disable monitor thread that can cause issues

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Viral YouTube Titles Training Pipeline")
    parser.add_argument(
        "stage", 
        choices=[
            "prep", "prep_regression", 
            "analyze_labels", "regression_title", "regression_description", "all",
        ],
        help="Stage of the pipeline to run"
    )
    
    # Essential arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--bs", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--model_ckpt", type=str, 
        #default="microsoft/deberta-v3-base", 
        default="microsoft/mdeberta-v3-base",
        help="Model checkpoint to use"
    )
    parser.add_argument(
        "--dataset", type=str, default="hf_dataset_reg",
        help="Dataset path for analysis or fixing"
    )
    
    # Add enhanced regression options
    parser.add_argument('--use-pairwise', action='store_true', 
                        help='Use pairwise ranking loss instead of MSE')
    parser.add_argument('--use-spearman', action='store_true',
                        help='Use Spearman correlation as metric for best model')
    parser.add_argument('--patience', type=int, default=2,
                        help='Early stopping patience')
    parser.add_argument('--enhanced', action='store_true',
                        help='Use all enhancements (balanced dataset + pairwise loss + spearman metric)')
    parser.add_argument('--accumulation', type=int, default=4,
                        help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    # Configure Windows console if needed
    configure_windows_console()
    
    # Handle the enhanced mode by setting all improvements
    if args.enhanced:
        args.use_pairwise = True
        args.use_spearman = True
    
    print(f"Running stage: {args.stage}")
    
    if args.stage == "prep":
        stage_prep()
    elif args.stage == "prep_regression":
        stage_prep_regression()
    elif args.stage == "analyze_labels":
        analyze_viral_score_distribution(args.dataset)
    elif args.stage == "regression_title":
        stage_regression(
            target="title", 
            epochs=args.epochs, 
            bs=args.bs, 
            lr=args.lr,
            model_ckpt=args.model_ckpt,
            scheduler_type="linear",
            weight_decay=0.01,
            warmup_ratio=0.1,
            use_pairwise=args.use_pairwise,
            use_spearman_metric=args.use_spearman,
            patience=args.patience,
            dataset_path=args.dataset,
            gradient_accumulation_steps=args.accumulation
        )
    elif args.stage == "regression_description":
        stage_regression(
            target="description", 
            epochs=args.epochs, 
            bs=args.bs,
            lr=args.lr,
            model_ckpt=args.model_ckpt,
            scheduler_type="linear",
            weight_decay=0.01,
            warmup_ratio=0.1,
            use_pairwise=args.use_pairwise,
            use_spearman_metric=args.use_spearman,
            patience=args.patience,
            dataset_path=args.dataset,
            gradient_accumulation_steps=args.accumulation
        )
    
    #elif args.stage == "sft":
    #    stage_sft(epochs=args.epochs, bs=args.bs)
    #elif args.stage == "reward":
    #    stage_reward(epochs=args.epochs)
    #elif args.stage == "rlhf":
    #    stage_rlhf(epochs=args.epochs, bs=args.bs)
    elif args.stage == "all":
        # Run all stages sequentially
        stage_prep()
        stage_prep_regression()
        analyze_viral_score_distribution("hf_dataset_reg")
        stage_regression(
            target="title", 
            epochs=args.epochs, 
            bs=args.bs,
            lr=args.lr,
            model_ckpt=args.model_ckpt,
            scheduler_type="linear",
            weight_decay=0.01,
            warmup_ratio=0.1,
            use_pairwise=args.use_pairwise,
            use_spearman_metric=args.use_spearman,
            patience=args.patience,
            dataset_path=args.dataset,
            gradient_accumulation_steps=args.accumulation
        )
        stage_regression(
            target="description", 
            epochs=args.epochs, 
            bs=args.bs,
            lr=args.lr,
            model_ckpt=args.model_ckpt,
            scheduler_type="linear",
            weight_decay=0.01,
            warmup_ratio=0.1,
            use_pairwise=args.use_pairwise,
            use_spearman_metric=args.use_spearman,
            patience=args.patience,
            dataset_path=args.dataset,
            gradient_accumulation_steps=args.accumulation
        )
        #stage_sft(epochs=args.epochs, bs=args.bs)
        #stage_reward(epochs=args.epochs)
        #stage_rlhf(epochs=args.epochs, bs=args.bs)

if __name__ == "__main__":
    main() 