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
    
    if hasattr(config, "feature_selector"):
        ensemble.feature_selector = config["feature_selector"]
    
    if hasattr(config, "pca"):
        ensemble.pca = config["pca"]
    
    if hasattr(config, "scaler"):
        ensemble.scaler = config["scaler"]
    
    return ensemble

def main():
    parser = argparse.ArgumentParser(description="Train final ensemble for viral YouTube titles")
    parser.add_argument("--dataset", type=str, default="hf_dataset_reg_improved",
                        help="Dataset path to use")
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
    
    args = parser.parse_args()
    
    # Configure Windows console if needed
    configure_windows_console()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dsd = DatasetDict.load_from_disk(args.dataset)
    
    # Extract test data
    test_texts = [str(ex["title"]) for ex in dsd["test"]]
    test_labels = [float(ex["viral_score"]) for ex in dsd["test"]]
    
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
    weighted_preds = weighted_ensemble.predict(test_texts)
    
    print("Getting predictions from stacking model...")
    stacking_preds = stacking_ensemble.predict(test_texts)
    
    # Combine predictions with different weights
    final_preds = (args.weighted_weight * weighted_preds + 
                   args.stacking_weight * stacking_preds)
    
    # Ensure predictions are within range [0, 1]
    final_preds = np.clip(final_preds, 0, 1)
    
    # Calculate metrics
    mse = mean_squared_error(test_labels, final_preds)
    spearman = spearmanr(test_labels, final_preds).correlation
    
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