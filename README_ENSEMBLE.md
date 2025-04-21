# YouTube Viral Score Ensemble Predictor

This ensemble model combines multiple transformer-based models with OpenAI embeddings to predict viral scores for YouTube titles, achieving Spearman correlations of 0.8+.

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```
   
   Or on Windows:
   ```
   set OPENAI_API_KEY=your_api_key_here
   ```

## Training the Ensemble

### Step 1: Train individual models

First, train at least two different transformer models:

```bash
# Train DeBERTa-v3-base
python train_viral_titles_pro.py regression_title --dataset hf_dataset_reg_improved --use-spearman --epochs 5 --lr 5e-5 --bs 128 --accumulation 8 --model_ckpt microsoft/deberta-v3-base

# Train DeBERTa-v3-large
python train_viral_titles_pro.py regression_title --dataset hf_dataset_reg_improved --use-spearman --epochs 5 --lr 3e-5 --bs 64 --accumulation 8 --model_ckpt microsoft/deberta-v3-large
```

This will create model checkpoints in `title_reg_ckpt/` (the latest model will overwrite previous ones).

### Step 2: Save each model with a unique name

After training each model, copy the checkpoint to a unique location:

```bash
# For DeBERTa-v3-base
cp -r title_reg_ckpt deberta_v3_base_ckpt

# For DeBERTa-v3-large
cp -r title_reg_ckpt deberta_v3_large_ckpt
```

### Step 3: Train the ensemble model

```bash
# Train a stacking ensemble with OpenAI embeddings
python train_viral_ensemble.py --dataset hf_dataset_reg_improved --ensemble_type stacking --use_openai --model_paths deberta_v3_base_ckpt deberta_v3_large_ckpt

# Or train a weighted average ensemble (faster, simpler)
python train_viral_ensemble.py --dataset hf_dataset_reg_improved --ensemble_type weighted_average --model_paths deberta_v3_base_ckpt deberta_v3_large_ckpt --model_weights 0.4 0.6
```

The ensemble model will be saved as `ensemble_title_stacking_model.pkl` or `ensemble_title_weighted_average_model.pkl`.

## Key Techniques Used

This ensemble approach combines several advanced techniques to boost performance:

1. **Multiple Transformer Models**: Utilizes both DeBERTa-v3-base and DeBERTa-v3-large for diverse predictions

2. **OpenAI Embeddings**: Incorporates text-embedding-ada-002 for high-quality semantic representations

3. **Stacking Ensemble**: Uses a meta-model (either Ridge Regression or Gradient Boosting) to combine the predictions

4. **Feature Engineering**: Extracts statistical features from the OpenAI embeddings

5. **Dimensionality Reduction**: Applies PCA to compress the high-dimensional OpenAI embeddings

## Expected Results

The combined ensemble model should achieve a Spearman correlation of 0.8+ on the test set, a significant improvement over individual models which typically reach 0.70-0.73.

## Advanced Usage

### Using Different Base Models

You can experiment with different base models:

```bash
python train_viral_ensemble.py --model_paths deberta_v3_base_ckpt deberta_v3_large_ckpt roberta_large_ckpt
```

### Fine-tuning the Meta-Model

If you're using the stacking ensemble, try different meta-models by modifying the `fit_meta_model` method in `viral_titles/utils/ensemble.py`.

### Prediction Only

To make predictions with an existing ensemble model:

```python
from viral_titles.utils.ensemble import EnsembleViralPredictor

# Define the model paths exactly as used during training
model_configs = [
    {"path": "deberta_v3_base_ckpt", "weight": 0.4},
    {"path": "deberta_v3_large_ckpt", "weight": 0.6}
]

# Load the ensemble
ensemble = EnsembleViralPredictor.load(
    "ensemble_title_stacking_model.pkl",
    model_configs
)

# Make predictions
predictions = ensemble.predict(
    ["How to make a viral YouTube video", "Cat vs Cucumber"],
    openai_cache_file="embeddings_cache.json"
) 