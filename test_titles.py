#!/usr/bin/env python
"""
Test new YouTube titles using trained ensemble models.
"""
import argparse
import pickle
import numpy as np
from scipy.stats import rankdata
from viral_titles.utils.ensemble import EnsembleViralPredictor
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

def percentile_rank(scores):
    """Convert raw scores to percentiles (0-100 range)"""
    # Convert to percentile ranks (0-1 range)
    ranks = rankdata(scores, "average") / len(scores)
    # Scale to 0-100 for easier interpretation
    return ranks * 100

def apply_sigmoid_scaling(scores, k=5):
    """Apply sigmoid scaling to spread out scores more evenly"""
    scores = np.array(scores)
    # Normalize to 0-1 range first
    min_score = scores.min()
    max_score = scores.max()
    if max_score > min_score:
        normalized = (scores - min_score) / (max_score - min_score)
    else:
        normalized = np.zeros_like(scores)
    
    # Apply sigmoid transformation to spread out mid-range values
    # k controls steepness - higher k gives more spread
    return 1 / (1 + np.exp(-k * (normalized - 0.5)))

def main():
    parser = argparse.ArgumentParser(description="Test new YouTube titles using trained ensemble models")
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
    parser.add_argument("--titles", type=str, nargs="+", required=True,
                        help="List of titles to test")
    parser.add_argument("--rank_average", action="store_true",
                        help="Use rank-based averaging for final predictions")
    parser.add_argument("--soft_clip_margin", type=float, default=0.1,
                        help="Margin for soft clipping (set to 0 to disable)")
    parser.add_argument("--use_only_weighted", action="store_true",
                        help="Use only the weighted average model (skip stacking)")
    parser.add_argument("--show_raw_scores", action="store_true",
                        help="Show raw prediction scores alongside percentiles")
    parser.add_argument("--apply_sigmoid", action="store_true",
                        help="Apply sigmoid scaling to spread out scores")
    parser.add_argument("--sigmoid_k", type=float, default=5.0,
                        help="Steepness parameter for sigmoid scaling")
    
    args = parser.parse_args()
    
    # Configure Windows console if needed
    configure_windows_console()
    
    # Create model configs
    model_configs = []
    for path in args.model_paths:
        model_configs.append({
            "path": path,
            "weight": 1.0
        })
    
    # Load the weighted average model
    print(f"Loading weighted average model from {args.weighted_model}")
    weighted_ensemble = load_ensemble(args.weighted_model, model_configs)
    
    # Get predictions from weighted average model
    print("Getting predictions from weighted average model...")
    weighted_preds = weighted_ensemble.predict(args.titles, use_rank=args.rank_average)
    
    if args.use_only_weighted:
        # If only using weighted model, use its predictions directly
        final_preds = weighted_preds
    else:
        try:
            # Try to load and use stacking model
            print(f"Loading stacking model from {args.stacking_model}")
            stacking_ensemble = load_ensemble(args.stacking_model, model_configs)
            
            print("Getting predictions from stacking model...")
            stacking_preds = stacking_ensemble.predict(args.titles, use_rank=args.rank_average)
            
            # Combine predictions with weights
            final_preds = (args.weighted_weight * weighted_preds + 
                          args.stacking_weight * stacking_preds)
        except Exception as e:
            print(f"\nWarning: Could not use stacking model: {str(e)}")
            print("Falling back to weighted average model only.")
            final_preds = weighted_preds
    
    # Store raw predictions before any post-processing
    raw_preds = final_preds.copy()
    
    # Apply soft clipping if needed
    if args.soft_clip_margin > 0:
        from viral_titles.utils.clipping import soft_clip
        print(f"Applying soft clipping with margin {args.soft_clip_margin}")
        final_preds = soft_clip(final_preds, margin=args.soft_clip_margin)
    else:
        # Ensure predictions are within range [0, 1]
        final_preds = np.clip(final_preds, 0, 1)
    
    # Calculate percentile ranks
    percentiles = percentile_rank(final_preds)
    
    # Apply sigmoid scaling if requested
    if args.apply_sigmoid:
        print(f"Applying sigmoid scaling with k={args.sigmoid_k}")
        sigmoid_scores = apply_sigmoid_scaling(final_preds, k=args.sigmoid_k)
    
    # Print results
    print("\nResults:")
    print("-" * 80)
    
    # Sort titles by score (descending)
    indices = np.argsort(-final_preds)
    
    for idx in indices:
        title = args.titles[idx]
        percentile = percentiles[idx]
        
        print(f"Title: {title}")
        
        if args.show_raw_scores:
            print(f"Raw score: {raw_preds[idx]:.4f}")
            
        print(f"Predicted viral score: {final_preds[idx]:.4f}")
        print(f"Percentile rank: {percentile:.1f}%")
        
        if args.apply_sigmoid:
            print(f"Sigmoid-scaled score: {sigmoid_scores[idx]:.4f}")
            
        print("-" * 80)

if __name__ == "__main__":
    main() 