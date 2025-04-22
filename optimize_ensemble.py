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

def debug_model_outputs(results_file="final_ensemble_results.pkl"):
    """
    Analyze model outputs to diagnose potential issues.
    
    This is helpful for identifying problems like:
    - Negative correlations
    - Binary predictions (all 0 or 1)
    - Inverted predictions
    - Distribution issues
    """
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found. Run step 3 first.")
        return
    
    import numpy as np
    from scipy.stats import spearmanr
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print("\n" + "="*80)
    print("MODEL OUTPUT DIAGNOSTICS")
    print("="*80)
    
    # Get the data
    texts = results["texts"]
    true_labels = results["true_labels"]
    weighted_preds = np.array(results["weighted_preds"])
    stacking_preds = np.array(results["stacking_preds"])
    final_preds = np.array(results["final_preds"])
    
    # Check for binary outputs
    def check_binary(preds, name):
        unique_vals = np.unique(preds)
        if len(unique_vals) <= 2:
            print(f"⚠️ WARNING: {name} predictions are binary with only {len(unique_vals)} unique values!")
            print(f"   Unique values: {unique_vals}")
        elif len(unique_vals) < 10:
            print(f"⚠️ WARNING: {name} predictions have only {len(unique_vals)} unique values!")
            print(f"   Unique values: {unique_vals}")
    
    check_binary(weighted_preds, "Weighted")
    check_binary(stacking_preds, "Stacking")
    check_binary(final_preds, "Final")
    
    # Check correlations
    w_corr = spearmanr(true_labels, weighted_preds).correlation
    s_corr = spearmanr(true_labels, stacking_preds).correlation
    f_corr = spearmanr(true_labels, final_preds).correlation
    
    print(f"\nCorrelations:")
    print(f"Weighted model: {w_corr:.4f}")
    print(f"Stacking model: {s_corr:.4f}")
    print(f"Final ensemble: {f_corr:.4f}")
    
    # Check for negative correlations
    if w_corr < 0:
        print(f"⚠️ WARNING: Weighted model has NEGATIVE correlation ({w_corr:.4f})!")
        print(f"   Try inverting predictions: 1 - predictions")
    
    if s_corr < 0:
        print(f"⚠️ WARNING: Stacking model has NEGATIVE correlation ({s_corr:.4f})!")
        print(f"   Try inverting predictions: 1 - predictions")
    
    # Print distributions
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    def print_dist(preds, name):
        hist, _ = np.histogram(preds, bins=bins)
        print(f"\n{name} prediction distribution:")
        for i, count in enumerate(hist):
            bin_range = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
            percentage = count / len(preds) * 100
            print(f"{bin_range}: {count:4d} ({percentage:5.1f}%)")
    
    print_dist(true_labels, "Ground truth")
    print_dist(weighted_preds, "Weighted")
    print_dist(stacking_preds, "Stacking")
    print_dist(final_preds, "Final")
    
    # Show some example predictions
    import random
    print("\nExample predictions:")
    print("Text | True | Weighted | Stacking | Final")
    print("-" * 80)
    
    sample_indices = random.sample(range(len(texts)), min(5, len(texts)))
    for idx in sample_indices:
        text = texts[idx]
        if len(text) > 40:
            text = text[:40] + "..."
        print(f"{text} | {true_labels[idx]:.2f} | {weighted_preds[idx]:.2f} | {stacking_preds[idx]:.2f} | {final_preds[idx]:.2f}")
    
    print("\nRecommendations:")
    if w_corr < 0 or s_corr < 0:
        print("1. Fix negative correlations by inverting predictions (1 - pred)")
    
    if len(np.unique(final_preds)) <= 2:
        print("2. Fix binary predictions by using a larger soft_clip_margin (try 0.2 or higher)")
        print("   or by using linear_clip instead of soft_clip")
    
    print("3. Always apply percentile_rank to predictions before blending for consistent scaling")
    print("4. Try reducing soft_clip_margin to preserve more of the distribution")
    
    return results

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
        ] + args.model_paths + [
            "--dataset", args.dataset,
            "--score_field", args.score_field,
            "--soft_clip_margin", "0"
        ]
        
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
        ] + args.model_paths + [
            "--dataset", args.dataset,
            "--score_field", args.score_field,
            "--soft_clip_margin", "0"
        ]
        
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
            "--holdout_split", str(args.holdout_split),
            "--model_paths"
        ] + args.model_paths + ["--dataset", args.dataset, "--score_field", args.score_field]
        
        # Add additional arguments
        if args.optimize_blend:
            cmd.append("--optimize_blend")
        
        # Add soft clipping parameter
        cmd.extend(["--soft_clip_margin", str(args.soft_clip_margin)])
        
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
    parser.add_argument("--dataset", type=str, default="hf_dataset_reg_improved",
                        help="Dataset path to use (default: hf_dataset_reg_improved)")
    parser.add_argument("--score_field", type=str, default="viral_score",
                        help="Field name containing the target score in the dataset")
    parser.add_argument("--debug", action="store_true",
                        help="Run diagnostic checks on model outputs")
    parser.add_argument("--soft_clip_margin", type=float, default=0.2,
                        help="Margin for soft clipping (larger values = more gradual transition)")
    parser.add_argument("--use_linear_clip", action="store_true",
                        help="Use linear clipping instead of soft clipping")
    parser.add_argument("--optimize_blend", action="store_true",
                        help="Optimize blend weights (for step 3)")
    
    args = parser.parse_args()
    
    # If debug mode is requested, run diagnostics
    if args.debug:
        print("Running diagnostic checks on model outputs...")
        debug_model_outputs()
        return
    
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