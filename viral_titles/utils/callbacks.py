"""
Custom training callbacks for viral titles training.
"""
import os
import torch
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from scipy.stats import spearmanr

class SpearmanCallback(TrainerCallback):
    """
    Callback to compute Spearman correlation during evaluation and save the best model.
    """
    def __init__(self, eval_dataset, tokenizer):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.best_spearman = -1.0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Ensure metrics dictionary exists
        if metrics is None:
            metrics = {}
            
        # Get the model from kwargs
        model = kwargs.get("model")
        if model is None:
            return metrics
        
        # Run prediction on eval dataset
        trainer = kwargs.get("trainer")
        if trainer is None:
            return metrics
            
        predictions = trainer.predict(self.eval_dataset)
        
        # Calculate Spearman correlation
        y_pred = predictions.predictions.squeeze()
        y_true = predictions.label_ids
        
        current_spearman = spearmanr(y_true, y_pred).correlation
        
        # Log the Spearman correlation
        metrics["eval_spearman"] = current_spearman
        print(f"Evaluation Spearman: {current_spearman:.4f}")
        
        # Save the best model based on Spearman
        if current_spearman > self.best_spearman:
            self.best_spearman = current_spearman
            
            # Only save if we're the main process
            if trainer.is_world_process_zero():
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
                
        # Return the metrics dict to ensure it's properly updated
        return metrics 