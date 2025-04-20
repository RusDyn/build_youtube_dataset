"""
RLHF (DPO) stage functions.
"""
import torch
import pathlib
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig

from ..config import MAX_LEN

def stage_rlhf(epochs=3, bs=1):
    """
    DPO-train policy against reward model.
    
    Args:
        epochs: Number of training epochs
        bs: Batch size
    """
    # Create offload directory for large models
    off = pathlib.Path("offload_dir")
    off.mkdir(exist_ok=True)
    
    # Load tokenizer from SFT checkpoint
    tok = AutoTokenizer.from_pretrained("sft_ckpt")
    
    # Model loading configuration for 4-bit quantization and offloading
    mk = {
        "device_map": "auto",
        "offload_folder": str(off),
        "offload_state_dict": True,
        "torch_dtype": torch.float16,
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True)
    }
    
    # Load policy and reward models
    policy = AutoModelForCausalLM.from_pretrained("sft_ckpt", **mk)
    reward = AutoModelForSequenceClassification.from_pretrained("rm_ckpt", **mk)
    
    # Load dataset
    dsd = DatasetDict.load_from_disk("hf_dataset")
    
    # Prepare dataset for DPO (using reverse of response as rejected)
    def mk_pair(ex): 
        return {
            "prompt": ex['prompt'],
            "chosen": ex['response'],
            "rejected": ex['response'][::-1]
        }
    
    dpo_ds = dsd['train'].map(mk_pair, remove_columns=dsd['train'].column_names)
    
    # DPO configuration
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
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=policy,
        ref_model=None,
        train_dataset=dpo_ds,
        args=dpo_cfg,
        tokenizer=tok,
        reward_model=reward,
    )
    
    # Train and save model
    trainer.train()
    trainer.save_model("dpo_ckpt")
    
    print("✅ RLHF DPO done ➜ dpo_ckpt/") 