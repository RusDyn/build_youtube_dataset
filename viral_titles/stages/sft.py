"""
Supervised Fine-Tuning (SFT) stage functions.
"""
import torch
import pathlib
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from ..utils import sanity_check_dataset
from ..config import BASE_MODEL, MAX_LEN, get_bnb_config

def stage_sft(epochs=3, bs=4):
    """
    Supervised Fine-Tuning stage.
    
    Args:
        epochs: Number of training epochs
        bs: Batch size
    """
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
    
    bnb_cfg = get_bnb_config()
    
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
    
    # Define response template (the string that always precedes the response)
    response_template = "### Response\nTitle:"

    # Create the collator
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tok)

    # Pass the collator to SFTTrainer
    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=args,
        train_dataset=dsd['train'],
        peft_config=peft_cfg,
        tokenizer=tok,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    
    trainer.train() 
    trainer.save_model("sft_ckpt")
    print("✅ SFT done ➜ sft_ckpt/") 