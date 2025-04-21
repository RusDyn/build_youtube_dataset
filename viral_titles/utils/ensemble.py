"""
Ensemble models for combining predictions from multiple sources.
"""
import os
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .openai_embeddings import batch_get_embeddings

class EnsembleViralPredictor:
    """
    Ensemble model that combines predictions from transformer models and OpenAI embeddings.
    """
    def __init__(self, models_config, ensemble_type="weighted_average", use_openai=True):
        """
        Initialize the ensemble predictor.
        
        Args:
            models_config: List of dictionaries with model configs
                Each dict should have 'path', 'weight', and optionally 'tokenizer_path'
            ensemble_type: Type of ensemble - "weighted_average" or "stacking" 
            use_openai: Whether to include OpenAI embeddings in the ensemble
        """
        self.models = []
        self.tokenizers = []
        self.weights = []
        self.ensemble_type = ensemble_type
        self.use_openai = use_openai
        self.meta_model = None
        self.label_stats = None  # Will store (mean, std) for normalization
        
        print(f"Initializing ensemble with {len(models_config)} models")
        
        # Load all models
        for config in models_config:
            model_path = config["path"]
            print(f"Loading model from {model_path}")
            
            # Load the model
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()  # Set to evaluation mode
            self.models.append(model)
            
            # Load tokenizer
            tok_path = config.get("tokenizer_path", model_path)
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizers.append(tokenizer)
            
            # Store weight
            self.weights.append(config.get("weight", 1.0))
        
        # Normalize weights if using weighted average
        if self.ensemble_type == "weighted_average":
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
            print(f"Normalized weights: {self.weights}")
    
    def fit_meta_model(self, texts, labels, max_length=64, openai_cache_file="openai_embeddings_cache.json"):
        """
        Fit the meta-model for stacked ensemble.
        
        Args:
            texts: List of text samples
            labels: Corresponding labels/targets
            max_length: Max sequence length for tokenizers
            openai_cache_file: File to cache OpenAI embeddings
        """
        if self.ensemble_type != "stacking":
            print("No need to fit meta model for weighted average ensemble")
            return
        
        print("Collecting base model predictions for meta-model training...")
        all_predictions = []
        
        # Get predictions from transformer models
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            print(f"Getting predictions from model {i+1}/{len(self.models)}")
            predictions = []
            
            # Process in batches
            batch_size = 32
            for j in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[j:j+batch_size]
                inputs = tokenizer(batch_texts, padding="max_length", truncation=True, 
                                 max_length=max_length, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_preds = outputs.logits.squeeze().cpu().numpy()
                    predictions.extend(batch_preds)
            
            all_predictions.append(predictions)
        
        # Get OpenAI embeddings if needed
        if self.use_openai:
            print("Getting OpenAI embeddings...")
            openai_embeddings = batch_get_embeddings(
                texts, cache_file=openai_cache_file
            )
            print(f"Obtained {len(openai_embeddings)} OpenAI embeddings")
        
        # Prepare features for meta-model
        X_meta = np.column_stack(all_predictions)
        
        if self.use_openai:
            # Apply dimensionality reduction to OpenAI embeddings
            openai_features = np.array(openai_embeddings)
            
            # Add basic statistical features from embeddings
            emb_mean = np.mean(openai_features, axis=1, keepdims=True)
            emb_std = np.std(openai_features, axis=1, keepdims=True)
            emb_max = np.max(openai_features, axis=1, keepdims=True)
            emb_min = np.min(openai_features, axis=1, keepdims=True)
            
            # Add compressed features from embeddings
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            emb_pca = pca.fit_transform(openai_features)
            
            # Combine all features
            X_meta = np.hstack([X_meta, emb_mean, emb_std, emb_max, emb_min, emb_pca])
        
        print(f"Meta-model input shape: {X_meta.shape}")
        
        # Compute label statistics for normalization
        label_mean = np.mean(labels)
        label_std = np.std(labels)
        self.label_stats = (label_mean, label_std)
        
        # Normalize labels
        y_meta = (np.array(labels) - label_mean) / label_std
        
        # Try both Ridge and GradientBoosting, select the best
        ridge = Ridge(alpha=1.0)
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        
        ridge_scores = cross_val_score(ridge, X_meta, y_meta, cv=5, 
                                      scoring=lambda est, X, y: spearmanr(
                                          est.predict(X), y
                                      ).correlation)
        
        gb_scores = cross_val_score(gb, X_meta, y_meta, cv=5, 
                                   scoring=lambda est, X, y: spearmanr(
                                       est.predict(X), y
                                   ).correlation)
        
        print(f"Ridge CV Spearman: {np.mean(ridge_scores):.4f} ± {np.std(ridge_scores):.4f}")
        print(f"GradientBoosting CV Spearman: {np.mean(gb_scores):.4f} ± {np.std(gb_scores):.4f}")
        
        # Select the better model
        if np.mean(ridge_scores) > np.mean(gb_scores):
            print("Using Ridge as meta-model")
            self.meta_model = Ridge(alpha=1.0)
        else:
            print("Using GradientBoosting as meta-model")
            self.meta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        
        # Fit the selected model on all training data
        self.meta_model.fit(X_meta, y_meta)
        print("Meta-model fitted successfully")
    
    def predict(self, texts, max_length=64, openai_cache_file="openai_embeddings_cache.json"):
        """
        Make predictions with the ensemble.
        
        Args:
            texts: List of text samples
            max_length: Max sequence length for tokenizers
            openai_cache_file: File to cache OpenAI embeddings
            
        Returns:
            Numpy array of predictions
        """
        all_predictions = []
        
        # Get predictions from transformer models
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            print(f"Getting predictions from model {i+1}/{len(self.models)}")
            predictions = []
            
            # Process in batches
            batch_size = 32
            for j in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[j:j+batch_size]
                inputs = tokenizer(batch_texts, padding="max_length", truncation=True, 
                                 max_length=max_length, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_preds = outputs.logits.squeeze().cpu().numpy()
                    predictions.extend(batch_preds)
            
            all_predictions.append(predictions)
        
        # For weighted average ensemble
        if self.ensemble_type == "weighted_average":
            # Combine predictions using weighted average
            ensemble_preds = np.zeros(len(texts))
            for i, preds in enumerate(all_predictions):
                ensemble_preds += np.array(preds) * self.weights[i]
            
            return ensemble_preds
        
        # For stacking ensemble
        elif self.ensemble_type == "stacking":
            # Get OpenAI embeddings if needed
            if self.use_openai:
                print("Getting OpenAI embeddings...")
                openai_embeddings = batch_get_embeddings(
                    texts, cache_file=openai_cache_file
                )
            
            # Prepare features for meta-model
            X_meta = np.column_stack(all_predictions)
            
            if self.use_openai:
                # Apply dimensionality reduction to OpenAI embeddings
                openai_features = np.array(openai_embeddings)
                
                # Add basic statistical features from embeddings
                emb_mean = np.mean(openai_features, axis=1, keepdims=True)
                emb_std = np.std(openai_features, axis=1, keepdims=True)
                emb_max = np.max(openai_features, axis=1, keepdims=True)
                emb_min = np.min(openai_features, axis=1, keepdims=True)
                
                # Add compressed features from embeddings
                from sklearn.decomposition import PCA
                pca = PCA(n_components=50)
                emb_pca = pca.fit_transform(openai_features)
                
                # Combine all features
                X_meta = np.hstack([X_meta, emb_mean, emb_std, emb_max, emb_min, emb_pca])
            
            # Make predictions with meta-model
            meta_preds = self.meta_model.predict(X_meta)
            
            # Denormalize predictions if label stats available
            if self.label_stats:
                label_mean, label_std = self.label_stats
                meta_preds = meta_preds * label_std + label_mean
            
            return meta_preds
        
    def save(self, path):
        """Save the ensemble model configuration"""
        import pickle
        
        config = {
            "ensemble_type": self.ensemble_type,
            "use_openai": self.use_openai,
            "weights": self.weights,
            "label_stats": self.label_stats,
            "meta_model": self.meta_model if self.meta_model else None,
        }
        
        with open(path, "wb") as f:
            pickle.dump(config, f)
        
        print(f"Ensemble configuration saved to {path}")
    
    @classmethod
    def load(cls, path, models_config):
        """Load an ensemble model configuration"""
        import pickle
        
        with open(path, "rb") as f:
            config = pickle.load(f)
        
        ensemble = cls(
            models_config=models_config,
            ensemble_type=config["ensemble_type"],
            use_openai=config["use_openai"]
        )
        
        ensemble.weights = config["weights"]
        ensemble.label_stats = config["label_stats"]
        ensemble.meta_model = config["meta_model"]
        
        return ensemble 