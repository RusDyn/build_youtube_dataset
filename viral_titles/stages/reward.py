"""
Reward model training stage functions.
"""
import torch
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

from ..config import MAX_LEN

def stage_reward(epochs=2):
    """
    Train a regression reward model predicting viral_score.
    
    Args:
        epochs: Number of training epochs
    """
    dsd = DatasetDict.load_from_disk("hf_dataset")
    base_rm = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(base_rm)
    
    def pref_pairs(ex):
        idx = sorted(range(len(ex['score'])), 
                     key=lambda i: ex['score'][i], 
                     reverse=True)
        chosen, rejected = [], []
        for i in range(0, len(idx)-1, 2):
            a, b = idx[i], idx[i+1]
            chosen.append(ex['prompt'][a] + ex['response'][a] if ex['score'][a] >= ex['score'][b] else ex['prompt'][b] + ex['response'][b])
            rejected.append(ex['prompt'][b] + ex['response'][b] if ex['score'][a] >= ex['score'][b] else ex['prompt'][a] + ex['response'][a])
        return {
            "chosen": chosen, 
            "rejected": rejected
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
        tokenizer=tok,
    )
    
    trainer.train()
    trainer.save_model("rm_ckpt")
    print("✅ Reward model ➜ rm_ckpt/") 