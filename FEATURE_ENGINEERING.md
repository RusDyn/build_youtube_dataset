# Feature Engineering Documentation

## Recent Improvements

As of April 20, 2025, the following improvements have been made to the viral score calculation process:

### 1. Fixed Data Issues
- **Negative Value Clipping**: Values are now clipped to be >= 0 before applying log1p transformation, eliminating "invalid value encountered in log1p" warnings.
- **Recency Value Capping**: The recency_n metric is now capped at 1.0 to ensure all normalized metrics stay within the expected [0,1] range.

### 2. Schema Versioning
- Implemented a versioning system for feature engineering to ensure consistency between training and inference.
- All transformation details, weights, and constraints are documented in `youtube_dataset/processing/schema_versions.py`.
- Version information is logged during the data preparation process.

## Viral Score Algorithm

The viral score calculation includes the following steps:

1. **Time-based metrics**:
   - Calculate views, likes, and comments per hour based on video age
   - Apply log transformation (log1p) to handle range and distribution
   - Normalize using sigmoid transformation for better distribution

2. **Engagement metrics**:
   - Rank score (inverted and normalized)
   - Recency boost using exponential decay
   - Engagement ratio (likes/views)
   - Early viral potential (combination of views and likes)

3. **Weighted combination**:
   - Apply weights to each normalized metric
   - Convert to percentile rank
   - Apply power transformation for better differentiation
   - Add small controlled random noise to ensure uniqueness

## Usage

To regenerate training data with these improvements:

```bash
python prepare_training_data.py
```

Then train the model:

```bash
python train_viral_titles_pro.py --stage regression_title --enhanced --dataset hf_dataset_reg_improved
```

## Future Considerations

When implementing inference:
- Always use the same version of feature engineering as was used during training
- Refer to the schema version documentation to ensure consistency
- Use the same normalization and transformation steps 