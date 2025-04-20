"""
Schema and feature engineering version documentation.

This file documents the different versions of data processing and feature 
engineering used in the YouTube viral score calculation, to ensure consistency
between training and inference.
"""

# Current schema version
CURRENT_SCHEMA_VERSION = "1.0.0"

# Feature engineering settings
FEATURE_ENGINEERING = {
    "1.0.0": {
        "description": "Initial feature engineering scheme with improved viral scores",
        "transformations": [
            "log1p on viewsPerHr, likesPerHr, commentsPerHr",
            "sigmoid normalization for _log metrics",
            "recency with exp(-age/24) capped at 1.0",
            "engagement ratio with sigmoid normalization",
            "earlyViral score based on views and likes",
            "power transformation (alpha=1.5) for viral percentile",
            "5% random noise to break ties"
        ],
        "weights": {
            "viewsPerHr_n": 0.35,
            "likesPerHr_n": 0.25,
            "commentsPerHr_n": 0.15,
            "rankScore_n": 0.15,
            "recency_n": 0.10,
            "engagementRatio_n": 0.10,
            "earlyViral_n": 0.05
        },
        "constraints": {
            "all_metrics_capped": True,  # All normalized metrics capped at [0,1]
            "negative_values_clipped": True  # Negative values clipped before log1p
        },
        "date_implemented": "2025-04-20"
    }
}

def get_schema_version():
    """Returns the current schema version."""
    return CURRENT_SCHEMA_VERSION

def get_feature_engineering_config(version=None):
    """
    Returns the feature engineering configuration for a specific version.
    
    Args:
        version (str, optional): Schema version. Defaults to current version.
        
    Returns:
        dict: Feature engineering configuration
        
    Raises:
        ValueError: If requested version doesn't exist
    """
    if version is None:
        version = CURRENT_SCHEMA_VERSION
    
    if version not in FEATURE_ENGINEERING:
        raise ValueError(f"Unknown schema version: {version}")
    
    return FEATURE_ENGINEERING[version] 