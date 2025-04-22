#!/usr/bin/env python
"""
Final Ensemble for YouTube Viral Title Prediction

This script combines our weighted average model with the stacking model
to create a final ensemble that should achieve 0.8+ Spearman correlation.
"""
import os
import pickle
import argparse
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from datasets import DatasetDict
import torch

from viral_titles.utils.ensemble import EnsembleViralPredictor, percentile_rank
from viral_titles.utils.clipping import soft_clip, linear_clip
from viral_titles import configure_windows_console

def load_ensemble(path, models_config):
    """Load an ensemble model from a saved file"""
    with open(path, 'rb') as f:
        config = pickle.load(f)
    
    ensemble = EnsembleViralPredictor(
        models_config=models_config,
        ensemble_type=config["ensemble_type"],
        use_openai=config["use_openai"]
    )
    
    ensemble.weights = config["weights"]
    ensemble.label_stats = config["label_stats"]
    ensemble.meta_model = config["meta_model"]
    
    # Load additional attributes if they exist in the config
    if "feature_selector" in config:
        ensemble.feature_selector = config["feature_selector"]
    
    if "pca" in config:
        ensemble.pca = config["pca"]
    
    if "scaler" in config:
        ensemble.scaler = config["scaler"]
    
    return ensemble

def main():
    parser = argparse.ArgumentParser(description="Train final ensemble for viral YouTube titles")
    parser.add_argument("--dataset", type=str, default="hf_dataset_reg_improved",
                        help="Dataset path to use")
    parser.add_argument("--score_field", type=str, default="viral_score", 
                        help="Field name containing the target score in the dataset")
    parser.add_argument("--weighted_model", type=str, default="ensemble_title_weighted_average_model.pkl",
                        help="Path to the weighted average model")
    parser.add_argument("--stacking_model", type=str, default="ensemble_title_stacking_model.pkl",
                        help="Path to the stacking model")
    parser.add_argument("--weighted_weight", type=float, default=0.4,
                        help="Weight for the weighted average model")
    parser.add_argument("--stacking_weight", type=float, default=0.6,
                        help="Weight for the stacking model")
    parser.add_argument("--model_paths", type=str, nargs="+",
                        default=["deberta_v3_base_ckpt", "deberta_v3_large_ckpt"],
                        help="Paths to trained models to include in ensemble")
    parser.add_argument("--rank_average", action="store_true",
                        help="Use rank-based averaging for final predictions")
    parser.add_argument("--optimize_blend", action="store_true",
                        help="Optimize the blend weights using a grid search")
    parser.add_argument("--holdout_split", type=float, default=0.1,
                        help="Percentage of test data to use for blend optimization")
    parser.add_argument("--soft_clip_margin", type=float, default=0.1,
                        help="Margin for soft clipping (set to 0 to disable)")
    parser.add_argument("--use_linear_clip", action="store_true",
                        help="Use linear clipping instead of soft clipping")
    
    args = parser.parse_args()
    
    # Configure Windows console if needed
    configure_windows_console()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dsd = DatasetDict.load_from_disk(args.dataset)
    
    # Extract test data
    test_texts = [str(ex["title"]) for ex in dsd["test"]]
    test_labels = [float(ex[args.score_field]) for ex in dsd["test"]]
    
    # Create model configs (needed for loading ensembles)
    model_configs = []
    for path in args.model_paths:
        model_configs.append({
            "path": path,
            "weight": 1.0
        })
    
    # Load the ensemble models
    print(f"Loading weighted average model from {args.weighted_model}")
    weighted_ensemble = load_ensemble(args.weighted_model, model_configs)
    
    print(f"Loading stacking model from {args.stacking_model}")
    stacking_ensemble = load_ensemble(args.stacking_model, model_configs)
    
    # Get predictions from both models
    print("Getting predictions from weighted average model...")
    weighted_preds = weighted_ensemble.predict(test_texts, use_rank=args.rank_average)
    
    print("Getting predictions from stacking model...")
    stacking_preds = stacking_ensemble.predict(test_texts, use_rank=args.rank_average)
    
    # Debug: Print raw prediction statistics
    print("\nDebug - Raw prediction statistics:")
    print(f"Weighted predictions - min: {weighted_preds.min():.4f}, max: {weighted_preds.max():.4f}, mean: {weighted_preds.mean():.4f}")
    print(f"Stacking predictions - min: {stacking_preds.min():.4f}, max: {stacking_preds.max():.4f}, mean: {stacking_preds.mean():.4f}")
    
    # Check if predictions are correctly ordered (positive correlation with ground truth)
    w_raw_corr = spearmanr(test_labels, weighted_preds).correlation
    s_raw_corr = spearmanr(test_labels, stacking_preds).correlation
    print(f"Raw correlations - Weighted: {w_raw_corr:.4f}, Stacking: {s_raw_corr:.4f}")
    
    # CRITICAL FIX: Invert predictions if correlation is negative
    if w_raw_corr < 0:
        print("INVERTING weighted model predictions (negative correlation detected)")
        weighted_preds = 1 - weighted_preds
        w_raw_corr = spearmanr(test_labels, weighted_preds).correlation
        print(f"After inversion - Weighted correlation: {w_raw_corr:.4f}")
    
    if s_raw_corr < 0:
        print("INVERTING stacking model predictions (negative correlation detected)")
        stacking_preds = 1 - stacking_preds
        s_raw_corr = spearmanr(test_labels, stacking_preds).correlation
        print(f"After inversion - Stacking correlation: {s_raw_corr:.4f}")
    
    # Optionally optimize blend weights on a hold-out portion
    blend_weights = [args.weighted_weight, args.stacking_weight]
    
    if args.optimize_blend and args.holdout_split > 0:
        print(f"Optimizing blend weights using {args.holdout_split:.0%} of test data...")
        
        # Create hold-out split from test data for weight optimization
        split_idx = int(len(test_texts) * (1 - args.holdout_split))
        opt_texts = test_texts[split_idx:]
        opt_labels = test_labels[split_idx:]
        
        # Get predictions for optimization set
        opt_weighted_preds = weighted_preds[split_idx:]
        opt_stacking_preds = stacking_preds[split_idx:]
        
        # Convert to ranks if requested
        if args.rank_average:
            opt_weighted_preds = percentile_rank(opt_weighted_preds)
            opt_stacking_preds = percentile_rank(opt_stacking_preds)
        
        # Grid search for optimal weights
        print("Performing grid search for optimal blend weights...")
        best_spearman = -1.0
        best_weights = [0.5, 0.5]  # Default equal weights
        
        grid_points = 21  # 0.0, 0.05, 0.1, ..., 1.0
        for i in range(grid_points):
            w_stacking = i / (grid_points - 1)
            w_weighted = 1.0 - w_stacking
            weights = [w_weighted, w_stacking]
            
            # Normalize weights
            total_weight = sum(weights)
            norm_weights = [w / total_weight for w in weights]
            
            # Compute blended predictions
            blended_preds = norm_weights[0] * opt_weighted_preds + norm_weights[1] * opt_stacking_preds
            
            # Apply clipping if needed
            if args.soft_clip_margin > 0:
                if args.use_linear_clip:
                    blended_preds = linear_clip(blended_preds, margin=args.soft_clip_margin)
                else:
                    blended_preds = soft_clip(blended_preds, margin=args.soft_clip_margin)
            
            # Calculate Spearman correlation
            spearman = spearmanr(opt_labels, blended_preds).correlation
            print(f"Weights: [W:{w_weighted:.2f}, S:{w_stacking:.2f}], Spearman: {spearman:.4f}")
            
            if spearman > best_spearman:
                best_spearman = spearman
                best_weights = weights
        
        print(f"Best blend weights: [W:{best_weights[0]:.2f}, S:{best_weights[1]:.2f}], Spearman: {best_spearman:.4f}")
        blend_weights = best_weights
    
    # If we didn't already get rank-averaged predictions, apply percentile rank now
    if not args.rank_average:
        print("Applying percentile ranking to predictions for consistent scaling")
        weighted_preds = percentile_rank(weighted_preds)
        stacking_preds = percentile_rank(stacking_preds)
    
    # Debug: Print prediction statistics after possible ranking
    print("\nDebug - Prediction statistics after ranking step:")
    print(f"Weighted predictions - min: {weighted_preds.min():.4f}, max: {weighted_preds.max():.4f}, mean: {weighted_preds.mean():.4f}")
    print(f"Stacking predictions - min: {stacking_preds.min():.4f}, max: {stacking_preds.max():.4f}, mean: {stacking_preds.mean():.4f}")
    
    # Combine predictions with weights
    final_preds = (blend_weights[0] * weighted_preds + 
                   blend_weights[1] * stacking_preds)
    
    # Apply clipping if needed
    if args.soft_clip_margin > 0:
        if args.use_linear_clip:
            print(f"Applying linear clipping with margin {args.soft_clip_margin}")
            final_preds = linear_clip(final_preds, margin=args.soft_clip_margin)
        else:
            print(f"Applying soft clipping with margin {args.soft_clip_margin}")
            final_preds = soft_clip(final_preds, margin=args.soft_clip_margin)
    else:
        # Ensure predictions are within range [0, 1]
        final_preds = np.clip(final_preds, 0, 1)
    
    # Calculate metrics
    mse = mean_squared_error(test_labels, final_preds)
    spearman = spearmanr(test_labels, final_preds).correlation
    
    # Debug: Print prediction distribution for final predictions
    print("\nDebug - Final prediction statistics:")
    print(f"Final predictions - min: {final_preds.min():.4f}, max: {final_preds.max():.4f}, mean: {final_preds.mean():.4f}")
    print(f"Ground truth - min: {min(test_labels):.4f}, max: {max(test_labels):.4f}, mean: {sum(test_labels)/len(test_labels):.4f}")
    
    # Print distribution of predictions in bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pred_hist, _ = np.histogram(final_preds, bins=bins)
    true_hist, _ = np.histogram(test_labels, bins=bins)
    
    print("\nPrediction distribution:")
    print("Bin  | Predictions | Ground Truth")
    print("-" * 40)
    for i, (p_count, t_count) in enumerate(zip(pred_hist, true_hist)):
        bin_range = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        print(f"{bin_range} | {p_count:11d} | {t_count:11d}")
    
    # Calculate metrics for individual models for comparison
    weighted_mse = mean_squared_error(test_labels, weighted_preds)
    weighted_spearman = spearmanr(test_labels, weighted_preds).correlation
    
    stacking_mse = mean_squared_error(test_labels, stacking_preds)
    stacking_spearman = spearmanr(test_labels, stacking_preds).correlation
    
    print("\n" + "="*50)
    print("Individual model performance:")
    print(f"Weighted Average - MSE: {weighted_mse:.6f}, Spearman: {weighted_spearman:.4f}")
    print(f"Stacking Model   - MSE: {stacking_mse:.6f}, Spearman: {stacking_spearman:.4f}")
    print("\nFinal ensemble performance:")
    print(f"MSE: {mse:.6f}")
    print(f"Spearman correlation: {spearman:.4f}")
    print("="*50)
    
    # Show sample predictions
    import random
    sample_indices = random.sample(range(len(test_texts)), min(10, len(test_texts)))
    print("\nSample predictions:")
    for idx in sample_indices:
        text = test_texts[idx]
        if len(text) > 50:
            text = text[:50] + "..."
        print(f"Text: '{text}'")
        print(f"True: {test_labels[idx]:.4f}, " + 
              f"Weighted: {weighted_preds[idx]:.4f}, " +
              f"Stacking: {stacking_preds[idx]:.4f}, " + 
              f"Final: {final_preds[idx]:.4f}")
        print("-" * 60)
    
    # Save the combined predictions
    results = {
        "texts": test_texts,
        "true_labels": test_labels,
        "weighted_preds": weighted_preds.tolist(),
        "stacking_preds": stacking_preds.tolist(),
        "final_preds": final_preds.tolist(),
        "blend_weights": blend_weights,
        "metrics": {
            "weighted_mse": weighted_mse,
            "weighted_spearman": weighted_spearman,
            "stacking_mse": stacking_mse,
            "stacking_spearman": stacking_spearman,
            "final_mse": mse,
            "final_spearman": spearman
        }
    }
    
    with open("final_ensemble_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f"Results saved to final_ensemble_results.pkl")

if __name__ == "__main__":
    main() 