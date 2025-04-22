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
    parser.add_argument("--score_field", type=str, default="viral_score",
                        help="Field name containing the target score in the dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--rank_average", action="store_true",
                        help="Use rank-based scaling for model predictions")
    parser.add_argument("--soft_clip_margin", type=float, default=0.1,
                        help="Margin for soft clipping (0 to disable)")
    parser.add_argument("--holdout_split", type=float, default=0.1,
                        help="Percentage of training data to use for weight optimization (if ensemble_type=weighted_average)")
    
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
    
    # Debug: Print available fields in the first example
    print("Available fields in the first training example:")
    print(list(dsd["train"][0].keys()))
    
    # Extract text and labels
    train_texts = [str(ex[args.target]) for ex in dsd["train"]]
    train_labels = [float(ex[args.score_field]) for ex in dsd["train"]]
    test_texts = [str(ex[args.target]) for ex in dsd["test"]]
    test_labels = [float(ex[args.score_field]) for ex in dsd["test"]]
    
    print(f"Loaded {len(train_texts)} training examples and {len(test_texts)} test examples")
    
    # For weighted average, optionally perform weight optimization on a holdout set
    if args.ensemble_type == "weighted_average" and not args.model_weights and len(args.model_paths) > 1:
        # Create hold-out split for weight optimization
        if args.holdout_split > 0:
            print(f"Creating {args.holdout_split:.0%} holdout split for weight optimization...")
            split_idx = int(len(train_texts) * (1 - args.holdout_split))
            holdout_texts = train_texts[split_idx:]
            holdout_labels = train_labels[split_idx:]
            train_texts_subset = train_texts[:split_idx]
            train_labels_subset = train_labels[:split_idx]
            print(f"Using {len(train_texts_subset)} examples for training and {len(holdout_texts)} for weight optimization")
            
            # Create model configs for grid search
            model_configs = []
            for path in args.model_paths:
                model_configs.append({
                    "path": path,
                    "weight": 1.0
                })
            
            # Create base ensemble for predictions
            base_ensemble = EnsembleViralPredictor(
                models_config=model_configs,
                ensemble_type="weighted_average",
                use_openai=False  # No need for OpenAI during weight optimization
            )
            
            # Get individual model predictions on holdout set
            print("Getting individual model predictions on holdout set for weight optimization...")
            model_predictions = []
            for i, (model, tokenizer) in enumerate(zip(base_ensemble.models, base_ensemble.tokenizers)):
                print(f"Getting predictions from model {i+1}/{len(base_ensemble.models)}")
                preds = base_ensemble.get_model_predictions(model, tokenizer, holdout_texts)
                
                # Apply rank transformation if requested
                if args.rank_average:
                    from viral_titles.utils.ensemble import percentile_rank
                    preds = percentile_rank(preds)
                    
                model_predictions.append(preds)
            
            # Grid search for optimal weights
            print("Performing grid search for optimal weights...")
            best_spearman = -1.0
            best_weights = [1.0] * len(args.model_paths)
            
            # Simple grid search for 2 models (weight for model 2, model 1 is 1-weight)
            if len(args.model_paths) == 2:
                grid_points = 21  # 0.0, 0.05, 0.1, ..., 1.0
                for i in range(grid_points):
                    w2 = i / (grid_points - 1)
                    w1 = 1.0 - w2
                    weights = [w1, w2]
                    
                    # Normalize weights
                    total_weight = sum(weights)
                    norm_weights = [w / total_weight for w in weights]
                    
                    # Compute weighted predictions
                    weighted_preds = np.zeros(len(holdout_texts))
                    for j, preds in enumerate(model_predictions):
                        weighted_preds += norm_weights[j] * preds
                    
                    # Apply soft clipping if needed
                    if args.soft_clip_margin > 0:
                        def soft_clip(x, margin=args.soft_clip_margin):
                            return 1 / (1 + np.exp(-(np.log(margin) / margin) * (x - 0.5) * 12))
                        weighted_preds = soft_clip(weighted_preds)
                    
                    # Calculate Spearman correlation
                    spearman = spearmanr(holdout_labels, weighted_preds).correlation
                    print(f"Weights: [{w1:.2f}, {w2:.2f}], Spearman: {spearman:.4f}")
                    
                    if spearman > best_spearman:
                        best_spearman = spearman
                        best_weights = weights
            else:
                print("Weight optimization for >2 models not implemented, using equal weights")
            
            print(f"Best weights: {best_weights}, Spearman: {best_spearman:.4f}")
            args.model_weights = best_weights
    
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
            openai_cache_file=f"openai_embeddings_{args.target}_train.json",
            use_rank=args.rank_average
        )
    
    # Make predictions on test set
    print("Making predictions on test set...")
    test_predictions = ensemble.predict(
        test_texts,
        max_length=MAX_LEN_TITLE if args.target == "title" else 256,
        openai_cache_file=f"openai_embeddings_{args.target}_test.json",
        use_rank=args.rank_average,
        soft_clip_margin=args.soft_clip_margin if args.soft_clip_margin > 0 else None
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
    print(f"Ensemble configuration saved to {ensemble_path}")
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