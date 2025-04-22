"""
Ensemble models for combining predictions from multiple sources.
"""
import os
import numpy as np
import torch
from sklearn.linear_model import Ridge, ElasticNet, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.svm import SVR
# Add more advanced features
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
            # Try multiple meta-models and select the best
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import re
import unicodedata

from .openai_embeddings import batch_get_embeddings

def percentile_rank(vec):
    """
    Convert a vector to its percentile ranks (0-1 range).
    This preserves the rank order while normalizing the scale.
    
    Args:
        vec: numpy array of predictions
        
    Returns:
        Percentile ranks of the same shape as input
    """
    # Get the ranks (1-indexed)
    ranks = np.array([sorted(vec).index(x) + 1 for x in vec])
    # Convert to percentiles (0-1 range)
    return (ranks - 1) / (len(vec) - 1)

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
    
    def extract_text_features(self, text):
        """Extract additional features from the text itself"""
        # Normalize text
        text = str(text)
        
        # Text length
        length = len(text)
        
        # Count special characters
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        uppercase_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        
        # Detect script/language features
        has_cyrillic = bool(re.search('[\u0400-\u04FF]', text))
        has_cjk = bool(re.search('[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]', text))
        has_arabic = bool(re.search('[\u0600-\u06FF]', text))
        has_devanagari = bool(re.search('[\u0900-\u097F]', text))
        
        # Create feature dictionary
        features = {
            'length': length,
            'question_marks': question_marks,
            'exclamation_marks': exclamation_marks,
            'uppercase_ratio': uppercase_count / max(1, length),
            'digit_ratio': digit_count / max(1, length),
            'has_cyrillic': int(has_cyrillic),
            'has_cjk': int(has_cjk),
            'has_arabic': int(has_arabic),
            'has_devanagari': int(has_devanagari)
        }
        
        return np.array(list(features.values()))
    
    def get_model_predictions(self, model, tokenizer, texts, max_length=64, batch_size=32):
        """
        Get predictions from a model more efficiently using GPU if available.
        """
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        predictions = []
        
        # Process in batches
        for j in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[j:j+batch_size]
            inputs = tokenizer(batch_texts, padding="max_length", truncation=True, 
                             max_length=max_length, return_tensors="pt")
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_preds = outputs.logits.squeeze().cpu().numpy()
                
                # Handle case where batch size is 1
                if batch_size == 1 or len(batch_texts) == 1:
                    batch_preds = np.array([batch_preds])
                    
                predictions.extend(batch_preds)
        
        return predictions
    
    def fit_meta_model(self, texts, labels, max_length=64, openai_cache_file="openai_embeddings_cache.json", use_rank=False):
        """
        Fit the meta-model for stacked ensemble.
        
        Args:
            texts: List of text samples
            labels: Corresponding labels/targets
            max_length: Max sequence length for tokenizers
            openai_cache_file: File to cache OpenAI embeddings
            use_rank: Whether to use percentile rank scaling for predictions
        """
        if self.ensemble_type != "stacking":
            print("No need to fit meta model for weighted average ensemble")
            return
        
        print("Collecting base model predictions for meta-model training...")
        all_predictions = []
        
        # Get predictions from transformer models
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            print(f"Getting predictions from model {i+1}/{len(self.models)}")
            # Use the optimized prediction method
            predictions = self.get_model_predictions(model, tokenizer, texts, max_length)
            
            # Apply rank transformation if requested
            if use_rank:
                predictions = percentile_rank(predictions)
                
            all_predictions.append(predictions)
        
        # Prepare features for meta-model (base model predictions)
        X_meta = np.column_stack(all_predictions)
        
        if self.use_openai:
            # Extract text features and include in meta-model input
            print("Extracting text features...")
            text_features = np.array([self.extract_text_features(text) for text in texts])
            print(f"Extracted text features with shape: {text_features.shape}")
            X_meta = np.hstack([X_meta, text_features])
            
            # Get OpenAI embeddings if needed
            print("Getting OpenAI embeddings...")
            openai_embeddings = batch_get_embeddings(
                texts, cache_file=openai_cache_file
            )
            print(f"Obtained {len(openai_embeddings)} OpenAI embeddings")
            
            # Apply dimensionality reduction and extract features from OpenAI embeddings
            openai_features = np.array(openai_embeddings)
            
            # Add basic statistical features from embeddings - more selectively
            emb_mean = np.mean(openai_features, axis=1, keepdims=True)
            emb_std = np.std(openai_features, axis=1, keepdims=True)
            
            # Standardize features
            scaler = StandardScaler()
            openai_scaled = scaler.fit_transform(openai_features)
            
            # Apply dimensionality reduction
            n_components = min(30, openai_features.shape[1], openai_features.shape[0]-1)
            try:
                pca = PCA(n_components=n_components)
                emb_pca = pca.fit_transform(openai_scaled)
                explained_var = sum(pca.explained_variance_ratio_)
                print(f"PCA explained variance: {explained_var:.4f}")
            except Exception as e:
                print(f"PCA failed: {e}, using SVD instead")
                # Fallback to SVD
                svd = TruncatedSVD(n_components=n_components)
                emb_pca = svd.fit_transform(openai_scaled)
                explained_var = sum(svd.explained_variance_ratio_)
                print(f"SVD explained variance: {explained_var:.4f}")
            
            # Select the most important embedding features for predicting the target
            # First, concatenate the transformer predictions with PCA features
            initial_features = np.column_stack([np.column_stack(all_predictions), emb_pca])
            
            # Compute label statistics for normalization
            label_mean = np.mean(labels)
            label_std = np.std(labels)
            self.label_stats = (label_mean, label_std)
            
            # Normalize labels
            y_meta = (np.array(labels) - label_mean) / label_std
            
            # Use feature selection to keep only the most predictive features
            selector = SelectKBest(f_regression, k=min(40, initial_features.shape[1]))
            selected_features = selector.fit_transform(initial_features, y_meta)
            
            # Keep track of which features were selected
            self.feature_selector = selector
            self.pca = pca if 'pca' in locals() else svd
            self.scaler = scaler
            
            # Use the selected features
            X_meta = selected_features
            
            print(f"Selected {selected_features.shape[1]} features out of {initial_features.shape[1]}")
        else:
            # Without OpenAI and without text features, X_meta remains only base predictions
            
            # Store label statistics for later use in prediction
            label_mean = np.mean(labels)
            label_std = np.std(labels)
            self.label_stats = (label_mean, label_std)
            
            # Normalize labels for training
            y_meta = (np.array(labels) - label_mean) / label_std
            
        print(f"Meta-model input shape: {X_meta.shape}")
        
        # Initialize models to try with stronger regularization
        models = {
            "Ridge": Ridge(alpha=10.0),
            "RidgeCV": RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0, 100.0], cv=5),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, 
                                                         max_depth=3, min_samples_split=20),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5)
        }
        
        # Evaluate each model using cross-validation
        best_score = -float('inf')
        best_model = None
        best_model_name = None
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_meta, y_meta, cv=5, scoring=lambda est, X, y: spearmanr(
                est.predict(X), y
            ).correlation)
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            print(f"{name} CV Spearman: {mean_score:.4f} ± {std_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_model_name = name
        
        print(f"Using {best_model_name} as meta-model with score {best_score:.4f}")
        
        # Fit the best model on all training data
        best_model.fit(X_meta, y_meta)
        self.meta_model = best_model
        print("Meta-model fitted successfully")
    
    def predict(self, texts, max_length=64, openai_cache_file="openai_embeddings_cache.json", use_rank=False, soft_clip_margin=0.1):
        """
        Make predictions using the ensemble model.
        
        Args:
            texts: List of text samples
            max_length: Max sequence length for tokenizers
            openai_cache_file: File to cache OpenAI embeddings
            use_rank: Whether to use percentile rank scaling before combining predictions
            soft_clip_margin: Margin for soft clipping (set to None to disable)
            
        Returns:
            numpy array of predictions
        """
        print("Making predictions on test set...")
        all_predictions = []
        
        # Get predictions from transformer models
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            print(f"Getting predictions from model {i+1}/{len(self.models)}")
            predictions = self.get_model_predictions(model, tokenizer, texts, max_length)
            all_predictions.append(predictions)
            
        # For weighted average ensemble
        if self.ensemble_type == "weighted_average":
            # Apply rank transformation if requested
            if use_rank:
                ranked_predictions = [percentile_rank(preds) for preds in all_predictions]
                all_predictions = ranked_predictions
                
            # Weighted average of all model predictions
            predictions = np.zeros(len(texts))
            for i, preds in enumerate(all_predictions):
                predictions += self.weights[i] * np.array(preds)
                
            # Apply soft clipping if margin is provided
            if soft_clip_margin is not None and soft_clip_margin > 0:
                def soft_clip(x, margin=soft_clip_margin):
                    """Soft clip to [0, 1] range with specified margin"""
                    return 1 / (1 + np.exp(-(np.log(margin) / margin) * (x - 0.5) * 12))
                
                predictions = soft_clip(predictions)
                
            return predictions
        
        # For stacking ensemble
        elif self.ensemble_type == "stacking":
            # Prepare features for meta-model prediction
            X_meta = np.column_stack(all_predictions)
            base_shape = X_meta.shape
            
            if self.use_openai:
                # Include text features when using OpenAI embeddings
                print("Extracting text features...")
                text_features = np.array([self.extract_text_features(text) for text in texts])
                X_meta = np.hstack([X_meta, text_features])
                # Add OpenAI embeddings
                print("Getting OpenAI embeddings...")
                openai_embeddings = batch_get_embeddings(
                    texts, cache_file=openai_cache_file
                )
                
                # Process OpenAI embeddings the same way as in training
                openai_features = np.array(openai_embeddings)
                
                # Apply the same transformations as during training
                openai_scaled = self.scaler.transform(openai_features)
                emb_pca = self.pca.transform(openai_scaled)
                
                # Combine with transformer predictions
                initial_features = np.column_stack([np.column_stack(all_predictions), emb_pca])
                
                # Apply feature selection
                X_meta = self.feature_selector.transform(initial_features)
                
                print(f"Feature matrix shape: {X_meta.shape}")
            else:
                # Without OpenAI embeddings, use transformer predictions only
                print(f"Using transformer predictions only. Feature matrix shape: {base_shape}")
            
            # Apply the meta-model
            predictions = self.meta_model.predict(X_meta)
            
            # Denormalize predictions
            if self.label_stats is not None:
                label_mean, label_std = self.label_stats
                predictions = predictions * label_std + label_mean
                
            # Apply soft clipping if margin is provided
            if soft_clip_margin is not None and soft_clip_margin > 0:
                def soft_clip(x, margin=soft_clip_margin):
                    """Soft clip to [0, 1] range with specified margin"""
                    return 1 / (1 + np.exp(-(np.log(margin) / margin) * (x - 0.5) * 12))
                
                predictions = soft_clip(predictions)
                
            return predictions
    
    def save(self, path):
        """
        Save the ensemble model to disk.
        
        Args:
            path: Path to save the model
        """
        import pickle
        
        # Prepare the config to save
        config = {
            "ensemble_type": self.ensemble_type,
            "weights": self.weights,
            "label_stats": self.label_stats,
            "meta_model": self.meta_model,
            "use_openai": self.use_openai
        }
        
        # Save additional attributes if they exist
        if hasattr(self, "feature_selector"):
            config["feature_selector"] = self.feature_selector
            
        if hasattr(self, "pca"):
            config["pca"] = self.pca
            
        if hasattr(self, "scaler"):
            config["scaler"] = self.scaler
        
        # Save the config
        with open(path, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Ensemble configuration saved to {path}")
    
    @classmethod
    def load(cls, path, models_config):
        """Load an ensemble model configuration"""
        import pickle
        
        with open(path, 'rb') as f:
            config = pickle.load(f)
        
        ensemble = cls(
            models_config=models_config,
            ensemble_type=config["ensemble_type"],
            use_openai=config["use_openai"]
        )
        
        ensemble.weights = config["weights"]
        ensemble.label_stats = config["label_stats"]
        ensemble.meta_model = config["meta_model"]
        
        # Load additional attributes if they exist
        if "feature_selector" in config:
            ensemble.feature_selector = config["feature_selector"]
        
        if "pca" in config:
            ensemble.pca = config["pca"]
        
        if "scaler" in config:
            ensemble.scaler = config["scaler"]
        
        return ensemble 