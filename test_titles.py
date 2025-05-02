#!/usr/bin/env python
"""
Test new YouTube titles using trained ensemble models.
"""
import argparse
import pickle
import numpy as np
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
    
    # Load the ensemble models
    print(f"Loading weighted average model from {args.weighted_model}")
    weighted_ensemble = load_ensemble(args.weighted_model, model_configs)
    
    print(f"Loading stacking model from {args.stacking_model}")
    stacking_ensemble = load_ensemble(args.stacking_model, model_configs)
    
    # Get predictions from both models
    print("Getting predictions from weighted average model...")
    weighted_preds = weighted_ensemble.predict(args.titles, use_rank=args.rank_average)
    
    print("Getting predictions from stacking model...")
    stacking_preds = stacking_ensemble.predict(args.titles, use_rank=args.rank_average)
    
    # Combine predictions with weights
    final_preds = (args.weighted_weight * weighted_preds + 
                   args.stacking_weight * stacking_preds)
    
    # Apply soft clipping if needed
    if args.soft_clip_margin > 0:
        from viral_titles.utils.clipping import soft_clip
        print(f"Applying soft clipping with margin {args.soft_clip_margin}")
        final_preds = soft_clip(final_preds, margin=args.soft_clip_margin)
    else:
        # Ensure predictions are within range [0, 1]
        final_preds = np.clip(final_preds, 0, 1)
    
    # Print results
    print("\nResults:")
    print("-" * 80)
    for title, pred in zip(args.titles, final_preds):
        print(f"Title: {title}")
        print(f"Predicted viral score: {pred:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    main() 