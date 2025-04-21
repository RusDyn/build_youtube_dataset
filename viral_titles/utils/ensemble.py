"""
Ensemble models for combining predictions from multiple sources.
"""
import os
import numpy as np
import torch
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.svm import SVR
# Add more advanced features
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
            # Try multiple meta-models and select the best
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import re
import unicodedata

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
        
        # Extract text features for all texts
        print("Extracting text features...")
        text_features = np.array([self.extract_text_features(text) for text in texts])
        print(f"Extracted text features with shape: {text_features.shape}")
        
        # Get OpenAI embeddings if needed
        if self.use_openai:
            print("Getting OpenAI embeddings...")
            openai_embeddings = batch_get_embeddings(
                texts, cache_file=openai_cache_file
            )
            print(f"Obtained {len(openai_embeddings)} OpenAI embeddings")
        
        # Prepare features for meta-model
        X_meta = np.column_stack(all_predictions)
        
        # Add text features to meta-model input
        X_meta = np.hstack([X_meta, text_features])
        
        if self.use_openai:
            # Apply dimensionality reduction and extract features from OpenAI embeddings
            openai_features = np.array(openai_embeddings)
            
            # Add basic statistical features from embeddings
            emb_mean = np.mean(openai_features, axis=1, keepdims=True)
            emb_std = np.std(openai_features, axis=1, keepdims=True)
            emb_max = np.max(openai_features, axis=1, keepdims=True)
            emb_min = np.min(openai_features, axis=1, keepdims=True)
            emb_median = np.median(openai_features, axis=1, keepdims=True)
            emb_q25 = np.percentile(openai_features, 25, axis=1, keepdims=True)
            emb_q75 = np.percentile(openai_features, 75, axis=1, keepdims=True)
            emb_skew = np.nan_to_num(((emb_mean - emb_median) * 3) / (emb_std + 1e-8))  # Approximate skewness
            emb_kurtosis = np.nan_to_num(((emb_q75 - emb_q25) / (2 * (emb_std + 1e-8))))  # Approximate kurtosis
            

            # Standardize features for better decomposition
            scaler = StandardScaler()
            openai_scaled = scaler.fit_transform(openai_features)
            
            # Apply PCA 
            pca = PCA(n_components=min(50, openai_features.shape[1], openai_features.shape[0]))
            try:
                emb_pca = pca.fit_transform(openai_scaled)
                print(f"PCA explained variance: {sum(pca.explained_variance_ratio_):.4f}")
            except Exception as e:
                print(f"PCA failed: {e}, using SVD instead")
                # Fallback to SVD if PCA fails
                svd = TruncatedSVD(n_components=min(50, openai_features.shape[1], openai_features.shape[0]-1))
                emb_pca = svd.fit_transform(openai_scaled)
                print(f"SVD explained variance: {sum(svd.explained_variance_ratio_):.4f}")
            
            # Combine all features
            openai_meta_features = np.hstack([
                emb_mean, emb_std, emb_max, emb_min, emb_median, 
                emb_q25, emb_q75, emb_skew, emb_kurtosis, emb_pca
            ])
            
            print(f"OpenAI meta features shape: {openai_meta_features.shape}")
            
            # Combine transformer predictions with OpenAI features
            X_meta = np.hstack([X_meta, openai_meta_features])
        
        print(f"Meta-model input shape: {X_meta.shape}")
        
        # Compute label statistics for normalization
        label_mean = np.mean(labels)
        label_std = np.std(labels)
        self.label_stats = (label_mean, label_std)
        
        # Normalize labels
        y_meta = (np.array(labels) - label_mean) / label_std
        


        # Initialize models to try
        models = {
            "Ridge": Ridge(alpha=1.0),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5),
            "SVR": SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # Evaluate all models
        best_score = -1
        best_model_name = None
        scores = {}
        
        for name, model in models.items():
            try:
                scores[name] = cross_val_score(model, X_meta, y_meta, cv=5, 
                                              scoring=lambda est, X, y: spearmanr(
                                                  est.predict(X), y
                                              ).correlation)
                
                mean_score = np.mean(scores[name])
                std_score = np.std(scores[name])
                
                print(f"{name} CV Spearman: {mean_score:.4f} ± {std_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        # Select the best model
        print(f"Using {best_model_name} as meta-model with score {best_score:.4f}")
        self.meta_model = models[best_model_name]
        
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
            # First, normalize each model's predictions to [0,1] range
            normalized_predictions = []
            for preds in all_predictions:
                # Calculate min/max for each model's predictions
                preds_array = np.array(preds)
                preds_min = np.min(preds_array)
                preds_max = np.max(preds_array)
                
                # Handle the case where all predictions are the same
                if preds_max == preds_min:
                    normalized_predictions.append(preds_array)
                else:
                    # Normalize to [0,1]
                    norm_preds = (preds_array - preds_min) / (preds_max - preds_min)
                    normalized_predictions.append(norm_preds)
            
            # Combine normalized predictions using weighted average
            ensemble_preds = np.zeros(len(texts))
            for i, norm_preds in enumerate(normalized_predictions):
                ensemble_preds += norm_preds * self.weights[i]
            
            # Apply softer clipping - use sigmoid-like function to compress extremes
            # but retain more gradation near the limits
            def soft_clip(x, margin=0.1):
                """Soft clipping function that preserves more detail at extremes"""
                # Apply sigmoid-like compression at the extremes
                lower_mask = x < margin
                upper_mask = x > (1.0 - margin)
                
                # Keep middle values unchanged
                result = x.copy()
                
                # Apply soft transformation for values below margin
                if np.any(lower_mask):
                    result[lower_mask] = margin * np.exp(
                        (x[lower_mask] - margin) / margin
                    )
                
                # Apply soft transformation for values above (1-margin)
                if np.any(upper_mask):
                    result[upper_mask] = 1.0 - margin * np.exp(
                        ((1.0 - margin) - x[upper_mask]) / margin
                    )
                
                return result
            
            # Apply soft clipping instead of hard clipping
            ensemble_preds = soft_clip(ensemble_preds, margin=0.05)
            
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
            
            # Add text features to meta-model input
            print("Extracting text features...")
            text_features = np.array([self.extract_text_features(text) for text in texts])
            X_meta = np.hstack([X_meta, text_features])
            
            if self.use_openai:
                # Apply the same feature extraction as in fit_meta_model
                openai_features = np.array(openai_embeddings)
                
                # Add basic statistical features from embeddings
                emb_mean = np.mean(openai_features, axis=1, keepdims=True)
                emb_std = np.std(openai_features, axis=1, keepdims=True)
                emb_max = np.max(openai_features, axis=1, keepdims=True)
                emb_min = np.min(openai_features, axis=1, keepdims=True)
                emb_median = np.median(openai_features, axis=1, keepdims=True)
                emb_q25 = np.percentile(openai_features, 25, axis=1, keepdims=True)
                emb_q75 = np.percentile(openai_features, 75, axis=1, keepdims=True)
                emb_skew = np.nan_to_num(((emb_mean - emb_median) * 3) / (emb_std + 1e-8))  # Approximate skewness
                emb_kurtosis = np.nan_to_num(((emb_q75 - emb_q25) / (2 * (emb_std + 1e-8))))  # Approximate kurtosis
                
    
                # Standardize features
                scaler = StandardScaler()
                openai_scaled = scaler.fit_transform(openai_features)
                
                # Apply PCA with fallback to SVD
                try:
                    pca = PCA(n_components=min(50, openai_features.shape[1], openai_features.shape[0]))
                    emb_pca = pca.fit_transform(openai_scaled)
                except Exception as e:
                    print(f"PCA failed: {e}, using SVD instead")
                    svd = TruncatedSVD(n_components=min(50, openai_features.shape[1], openai_features.shape[0]-1))
                    emb_pca = svd.fit_transform(openai_scaled)
                
                # Combine all features
                openai_meta_features = np.hstack([
                    emb_mean, emb_std, emb_max, emb_min, emb_median, 
                    emb_q25, emb_q75, emb_skew, emb_kurtosis, emb_pca
                ])
                
                # Combine transformer predictions with OpenAI features
                X_meta = np.hstack([X_meta, openai_meta_features])
            
            # Make predictions with meta-model
            meta_preds = self.meta_model.predict(X_meta)
            
            # Denormalize predictions if label stats available
            if self.label_stats:
                label_mean, label_std = self.label_stats
                meta_preds = meta_preds * label_std + label_mean
            
            # Apply soft clipping to ensure values are within [0, 1] but preserve more detail
            def soft_clip(x, margin=0.1):
                """Soft clipping function that preserves more detail at extremes"""
                # Apply sigmoid-like compression at the extremes
                lower_mask = x < 0
                upper_mask = x > 1.0
                mid_lower_mask = (x >= 0) & (x < margin)
                mid_upper_mask = (x <= 1.0) & (x > (1.0 - margin))
                
                # Keep middle values unchanged
                result = x.copy()
                
                # Hard clip severe outliers (less than 0 or greater than 1)
                if np.any(lower_mask):
                    result[lower_mask] = 0
                if np.any(upper_mask):
                    result[upper_mask] = 1.0
                
                # Apply soft transformation for borderline values
                if np.any(mid_lower_mask):
                    result[mid_lower_mask] = margin * (x[mid_lower_mask] / margin) ** 2
                
                if np.any(mid_upper_mask):
                    overflow = x[mid_upper_mask] - (1.0 - margin)
                    result[mid_upper_mask] = (1.0 - margin) + margin * (1 - (1 - overflow/margin) ** 2)
                
                return result
            
            # Apply soft clipping
            meta_preds = soft_clip(meta_preds, margin=0.05)
            
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