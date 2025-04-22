#!/usr/bin/env python
"""
Optimize YouTube Viral Title Prediction Ensemble

This script implements the step-by-step optimization plan to improve the
Spearman correlation of our ensemble model beyond 0.80.

Step 1: Retrain stacking ensemble without OpenAI embeddings
Step 2: Use rank-based averaging instead of min-max scaling
Step 3: Optimize weights for weighted-average on a holdout split
Step 4: Adjust soft-clipping parameters
Step 5: Create optimal blend of weighted-average and stacking models
"""
import os
import argparse
import subprocess
import pickle
from pathlib import Path

def run_step(step_number, args):
    """Run the specified optimization step"""
    if step_number == 1:
        print("="*80)
        print("Step 1: Retraining stacking ensemble without OpenAI embeddings")
        print("="*80)
        
        cmd = [
            "python", "train_viral_ensemble.py",
            "--ensemble_type", "stacking",
            "--model_paths"
        ] + args.model_paths + ["--dataset", args.dataset]
        
        # Add any additional args
        if args.rank_average:
            cmd.append("--rank_average")
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
        
        # Rename the output file to avoid overwriting it
        output_file = "ensemble_title_stacking_model.pkl"
        if os.path.exists(output_file):
            new_name = "ensemble_title_stacking_no_openai_model.pkl"
            os.rename(output_file, new_name)
            print(f"Renamed {output_file} to {new_name}")
        
    elif step_number == 2:
        print("="*80)
        print("Step 2: Training weighted-average ensemble with rank scaling")
        print("="*80)
        
        cmd = [
            "python", "train_viral_ensemble.py",
            "--ensemble_type", "weighted_average",
            "--rank_average",
            "--model_paths"
        ] + args.model_paths + ["--dataset", args.dataset]
        
        # Add holdout split for weight optimization
        cmd.extend(["--holdout_split", str(args.holdout_split)])
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
        
        # Rename the output file to avoid overwriting it
        output_file = "ensemble_title_weighted_average_model.pkl"
        if os.path.exists(output_file):
            new_name = "ensemble_title_weighted_average_rank_model.pkl"
            os.rename(output_file, new_name)
            print(f"Renamed {output_file} to {new_name}")
            
    elif step_number == 3:
        print("="*80)
        print("Step 3: Creating final blend of optimized models")
        print("="*80)
        
        # Check if we have only a single model path
        if len(args.model_paths) == 1:
            print("Warning: Only a single model path was provided.")
            print("For better results, please provide at least two model paths.")
            print("Using the single model for demonstration purposes.")
        
        # First model is the rank-based weighted average
        weighted_model = "ensemble_title_weighted_average_rank_model.pkl"
        
        # Second model is the stacking model without OpenAI embeddings
        stacking_model = "ensemble_title_stacking_no_openai_model.pkl"
        
        if not os.path.exists(weighted_model) or not os.path.exists(stacking_model):
            print(f"Error: Missing required model files: {weighted_model} or {stacking_model}")
            print("Please run steps 1 and 2 first.")
            return
        
        cmd = [
            "python", "train_ensemble_final.py",
            "--weighted_model", weighted_model,
            "--stacking_model", stacking_model,
            "--rank_average",
            "--optimize_blend",
            "--holdout_split", str(args.holdout_split),
            "--model_paths"
        ] + args.model_paths + ["--dataset", args.dataset]
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    elif step_number == 4:
        print("="*80)
        print("Step 4: Adding a new multilingual checkpoint for improved diversity")
        print("="*80)
        
        print("This step requires training or fine-tuning an additional multilingual model")
        print("like xlm-roberta-large on the same dataset, which is a more extensive process.")
        print("After doing this, re-run steps 1-3 with the additional model included in --model_paths.")
            
    else:
        print(f"Step {step_number} not implemented yet.")

def main():
    parser = argparse.ArgumentParser(description="Run viral title ensemble optimization steps")
    parser.add_argument("--step", type=int, required=True,
                        help="Optimization step to run (1-4)")
    parser.add_argument("--model_paths", type=str, nargs="+",
                        default=["title_reg_ckpt"],
                        help="Paths to trained models to include in ensemble")
    parser.add_argument("--holdout_split", type=float, default=0.1,
                        help="Proportion of training data to use for weight optimization")
    parser.add_argument("--rank_average", action="store_true",
                        help="Use rank-based averaging in the stacking ensemble")
    parser.add_argument("--dataset", type=str, default="hf_dataset",
                        help="Dataset path to use")
    
    args = parser.parse_args()
    
    # Run the specified step
    run_step(args.step, args)
    
    # Print summary of results if we have them
    results_file = "final_ensemble_results.pkl"
    if os.path.exists(results_file) and args.step == 3:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        print("\n" + "="*80)
        print("Optimization Results Summary:")
        print("="*80)
        metrics = results["metrics"]
        print(f"Weighted Average Spearman: {metrics['weighted_spearman']:.4f}")
        print(f"Stacking Model Spearman:   {metrics['stacking_spearman']:.4f}")
        print(f"Final Ensemble Spearman:   {metrics['final_spearman']:.4f}")
        print(f"Blend weights: {results['blend_weights']}")
        
        # Print improvement over baseline
        baseline = 0.7161  # From the original weighted average model
        improvement = metrics['final_spearman'] - baseline
        print(f"\nImprovement over baseline: {improvement:.4f} ({improvement/baseline*100:.1f}%)")
        print(f"Target improvement: 0.0839 (11.7%) to reach 0.80")
        
        if metrics['final_spearman'] >= 0.80:
            print("\n🎉 SUCCESS! Reached the 0.80 Spearman correlation target! 🎉")
        else:
            progress = improvement / 0.0839 * 100
            print(f"\nProgress toward 0.80 target: {progress:.1f}%")
            print("Next recommended step: Add a multilingual model (Step 4)")

if __name__ == "__main__":
    main() 