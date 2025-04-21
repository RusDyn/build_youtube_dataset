#!/usr/bin/env python
"""
Viral YouTube Titles Ensemble Training

This script trains an ensemble model that combines multiple transformer models with
OpenAI embeddings to predict viral scores for YouTube titles.

Usage:
  python train_viral_ensemble.py --dataset hf_dataset_reg_improved --ensemble_type stacking
"""
import os
import argparse
import numpy as np
import random
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from datasets import DatasetDict
from tqdm import tqdm

from viral_titles.utils.ensemble import EnsembleViralPredictor
from viral_titles.config import MAX_LEN_TITLE
from viral_titles import configure_windows_console

def main():
    parser = argparse.ArgumentParser(description="Train ensemble model for viral YouTube titles")
    parser.add_argument("--dataset", type=str, default="hf_dataset_reg_improved",
                        help="Dataset path to use")
    parser.add_argument("--ensemble_type", type=str, default="stacking",
                        choices=["weighted_average", "stacking"],
                        help="Type of ensemble to use")
    parser.add_argument("--use_openai", action="store_true",
                        help="Include OpenAI embeddings in the ensemble")
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API key (if not set in environment)")
    parser.add_argument("--model_paths", type=str, nargs="+",
                        default=["title_reg_ckpt"],
                        help="Paths to trained models to include in ensemble")
    parser.add_argument("--model_weights", type=float, nargs="+",
                        help="Weights for each model (if using weighted_average)")
    parser.add_argument("--target", type=str, default="title",
                        choices=["title", "description"],
                        help="Target field to predict from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Configure Windows console if needed
    configure_windows_console()
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set OpenAI API key if provided
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    # Check if OpenAI API key is set
    if args.use_openai and not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not set in environment. OpenAI embeddings will not be used.")
        args.use_openai = False
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    dsd = DatasetDict.load_from_disk(args.dataset)
    
    # Extract text and labels
    train_texts = [str(ex[args.target]) for ex in dsd["train"]]
    train_labels = [float(ex["viral_score"]) for ex in dsd["train"]]
    test_texts = [str(ex[args.target]) for ex in dsd["test"]]
    test_labels = [float(ex["viral_score"]) for ex in dsd["test"]]
    
    print(f"Loaded {len(train_texts)} training examples and {len(test_texts)} test examples")
    
    # Create model configs
    model_configs = []
    for i, path in enumerate(args.model_paths):
        weight = 1.0
        if args.model_weights and i < len(args.model_weights):
            weight = args.model_weights[i]
        
        model_configs.append({
            "path": path,
            "weight": weight
        })
    
    # Create ensemble predictor
    ensemble = EnsembleViralPredictor(
        models_config=model_configs,
        ensemble_type=args.ensemble_type,
        use_openai=args.use_openai
    )
    
    # For stacking ensemble, train the meta-model
    if args.ensemble_type == "stacking":
        print("Training meta-model for stacking ensemble...")
        ensemble.fit_meta_model(
            train_texts, 
            train_labels, 
            max_length=MAX_LEN_TITLE if args.target == "title" else 256,
            openai_cache_file=f"openai_embeddings_{args.target}_train.json"
        )
    
    # Make predictions on test set
    print("Making predictions on test set...")
    test_predictions = ensemble.predict(
        test_texts,
        max_length=MAX_LEN_TITLE if args.target == "title" else 256,
        openai_cache_file=f"openai_embeddings_{args.target}_test.json"
    )
    
    # Calculate metrics
    mse = mean_squared_error(test_labels, test_predictions)
    spearman = spearmanr(test_labels, test_predictions).correlation
    
    print("\n" + "="*50)
    print(f"Ensemble model performance ({args.ensemble_type}):")
    print(f"MSE: {mse:.6f}")
    print(f"Spearman correlation: {spearman:.4f}")
    print("="*50)
    
    # Save ensemble model
    ensemble_path = f"ensemble_{args.target}_{args.ensemble_type}_model.pkl"
    ensemble.save(ensemble_path)
    print(f"Ensemble model saved to {ensemble_path}")
    
    # Show sample predictions
    sample_indices = random.sample(range(len(test_texts)), min(10, len(test_texts)))
    print("\nSample predictions:")
    for idx in sample_indices:
        text = test_texts[idx]
        if len(text) > 50:
            text = text[:50] + "..."
        print(f"Text: '{text}'")
        print(f"True: {test_labels[idx]:.4f}, Pred: {test_predictions[idx]:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    main() 