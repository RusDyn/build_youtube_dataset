"""
Custom training callbacks for viral titles training.
"""
import os
import torch
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

class SpearmanCallback(TrainerCallback):
    """
    Callback to save the best model based on Spearman correlation.
    """
    def __init__(self, eval_dataset, tokenizer):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.best_spearman = -1.0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_spearman" not in metrics:
            return
            
        current_spearman = metrics["eval_spearman"]
        print(f"Evaluation Spearman: {current_spearman:.4f}")
        
        # Save the best model based on Spearman
        if current_spearman > self.best_spearman:
            self.best_spearman = current_spearman
            
            # Only save if we're the main process
            trainer = kwargs.get("trainer")
            if trainer and trainer.is_world_process_zero():
                # Save the model
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-best-spearman"
                output_dir = os.path.join(args.output_dir, checkpoint_folder)
                trainer.save_model(output_dir)
                
                # Also save the tokenizer and training arguments
                if trainer.tokenizer is not None:
                    trainer.tokenizer.save_pretrained(output_dir)
                    
                # Save trainer state
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                
                print(f"🔥 New best model saved with Spearman: {current_spearman:.4f}") 