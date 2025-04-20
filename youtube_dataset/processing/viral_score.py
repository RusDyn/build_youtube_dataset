#!/usr/bin/env python3
"""
Data processing module for YouTube dataset builder.
Handles data transformation and viral score calculations.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
from youtube_dataset.config import KEEP_REGIONS

from youtube_dataset.config import NOW_UTC

def add_viral(df):
    """
    Add viral score metrics to the DataFrame with enhanced diversity.
    Uses non-linear transformations and controlled randomness to create
    a more continuous distribution of viral scores.
    
    Args:
        df (pd.DataFrame): Input DataFrame with video data
        
    Returns:
        pd.DataFrame: DataFrame with added viral score columns
    """
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert date columns to datetime (force UTC)
    df.loc[:, "publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
    df.loc[:, "trendingDate"] = pd.to_datetime(df["trendingDate"], errors="coerce", utc=True)

    # Ensure both columns are datetime64[ns, UTC]
    if df["publishedAt"].dtype != "datetime64[ns, UTC]":
        df.loc[:, "publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
    if df["trendingDate"].dtype != "datetime64[ns, UTC]":
        df.loc[:, "trendingDate"] = pd.to_datetime(df["trendingDate"], errors="coerce", utc=True)
    
    # Use trending date if available, otherwise use current time
    # NOW_UTC is already a timezone-aware datetime
    now_ts = pd.Timestamp(NOW_UTC)
    # Convert to datetime series with UTC timezone to ensure proper type
    snap = pd.to_datetime(df["trendingDate"], utc=True)
    snap = snap.fillna(now_ts)
    
    # Calculate age in hours, with minimum 1 hour to avoid division by zero
    # Ensure both operands are timezone-aware datetime
    age = (snap - pd.to_datetime(df["publishedAt"], utc=True)).dt.total_seconds() / 3600
    age = age.replace(0, 1).fillna(24)
    
    # Calculate engagement per hour metrics
    df.loc[:, "viewsPerHr"] = df["viewCount"] / age
    df.loc[:, "likesPerHr"] = df["likeCount"] / age
    df.loc[:, "commentsPerHr"] = df["commentCount"] / age
    
    # Use non-linear transformations for better distribution
    # Log transformation helps with heavy-tailed metrics and creates more diversity
    for c in ["viewsPerHr", "likesPerHr", "commentsPerHr"]:
        # Add 1 to avoid log(0) and apply log transformation
        df.loc[:, c + "_log"] = np.log1p(df[c])
    
    # Non-linear normalization using sigmoid/tanh functions
    # This spreads values more evenly across the range
    for c in ["viewsPerHr_log", "likesPerHr_log", "commentsPerHr_log"]:
        # Get statistics for robust scaling
        median = df[c].median()
        q75 = df[c].quantile(0.75)
        q25 = df[c].quantile(0.25)
        scale = (q75 - q25) if q75 > q25 else 1
        
        # Scale using median and IQR for robustness to outliers
        scaled = (df[c] - median) / scale
        
        # Apply sigmoid to get values between 0-1 with better distribution
        df.loc[:, c.replace("_log", "_n")] = 1 / (1 + np.exp(-scaled))
    
    # Calculate rank score using non-linear transformation
    rank = df["rank"].fillna(50)
    # Inverse and normalize using sigmoid instead of linear scaling
    rank_normalized = 1 - (rank / 50)  # Higher ranks get lower scores (0 to 1)
    # Apply sigmoid for non-linear transformation
    df.loc[:, "rankScore_n"] = 1 / (1 + np.exp(-6 * (rank_normalized - 0.5)))
    
    # Calculate recency boost using exponential decay
    # Creates more natural falloff with age
    recency_boost = np.exp(-age / 24)  # Decay constant set to 24 hours
    df.loc[:, "recency_n"] = recency_boost
    
    # Calculate engagement ratio (likes/views) as additional signal
    # Helps distinguish truly engaging content
    likes_to_views = df["likeCount"] / df["viewCount"].replace(0, 1)
    df.loc[:, "engagementRatio_n"] = 1 / (1 + np.exp(-10 * (likes_to_views - 0.05)))
    
    # Calculate viral potential score based on early performance
    early_viral = np.sqrt(df["viewsPerHr_n"] * df["likesPerHr_n"])
    df.loc[:, "earlyViral_n"] = early_viral
    
    # Apply more complex weighting with additional metrics
    raw_score = (
        0.35 * df["viewsPerHr_n"] +
        0.25 * df["likesPerHr_n"] +
        0.15 * df["commentsPerHr_n"] +
        0.15 * df["rankScore_n"] +
        0.10 * df["recency_n"] +
        0.10 * df["engagementRatio_n"] +
        0.05 * df["earlyViral_n"]
    )
    
    # Normalize the raw score using percentiles for better distribution
    viral_percentile = raw_score.rank(pct=True)
    
    # Apply non-linear transformation to emphasize differences and create smoother distribution
    # Power transformation with small exponent helps differentiate videos
    alpha = 1.5  # Values > 1 emphasize high viral scores
    viral_score_base = viral_percentile ** alpha
    
    # Add controlled random noise (unique per row) to break ties and create diversity
    # Use 5% max noise based on rank order to maintain overall ranking
    np.random.seed(42)  # For reproducibility
    noise = np.random.rand(len(df)) * 0.05 * (1 - viral_score_base)
    
    # Final viral score with noise but no rounding to avoid clustering
    df.loc[:, "viral_score"] = np.clip(viral_score_base + noise, 0, 1)
    
    # Print distribution statistics for each normalized metric
    print("\n📊 Distribution of normalized metrics used in viral score:")
    
    normalized_metrics = [
        "viewsPerHr_n", "likesPerHr_n", "commentsPerHr_n", 
        "rankScore_n", "recency_n", "engagementRatio_n", "earlyViral_n"
    ]
    
    # Get basic statistics for each metric
    stats_df = df[normalized_metrics].describe().T
    stats_df['null_count'] = df[normalized_metrics].isna().sum()
    stats_df['null_pct'] = (df[normalized_metrics].isna().sum() / len(df)) * 100
    stats_df['zeros'] = (df[normalized_metrics] == 0).sum()
    stats_df['zeros_pct'] = (df[normalized_metrics] == 0).sum() / len(df) * 100
    
    # Print formatted statistics
    print(f"{'Metric':<20} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'25%':>8} | {'50%':>8} | {'75%':>8} | {'Max':>8} | {'Nulls %':>8} | {'Zeros %':>8}")
    print("-" * 120)
    
    for metric in normalized_metrics:
        print(f"{metric:<20} | {stats_df.loc[metric, 'mean']:>8.4f} | {stats_df.loc[metric, 'std']:>8.4f} | "
              f"{stats_df.loc[metric, 'min']:>8.4f} | {stats_df.loc[metric, '25%']:>8.4f} | "
              f"{stats_df.loc[metric, '50%']:>8.4f} | {stats_df.loc[metric, '75%']:>8.4f} | "
              f"{stats_df.loc[metric, 'max']:>8.4f} | {stats_df.loc[metric, 'null_pct']:>8.2f} | "
              f"{stats_df.loc[metric, 'zeros_pct']:>8.2f}")
    
    # Calculate and print correlation matrix between metrics
    print("\n📈 Correlation matrix between normalized metrics:")
    corr_matrix = df[normalized_metrics].corr()
    
    # Print formatted correlation matrix
    print(f"{'Metric':<20} |", end="")
    for metric in normalized_metrics:
        print(f" {metric[:8]:>8} |", end="")
    print()
    print("-" * 120)
    
    for row_metric in normalized_metrics:
        print(f"{row_metric:<20} |", end="")
        for col_metric in normalized_metrics:
            print(f" {corr_matrix.loc[row_metric, col_metric]:>8.4f} |", end="")
        print()
    
    # Analyze the distribution of the final viral score
    print("\n🏆 Final viral_score distribution:")
    
    # Create bins and count values in each bin
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_counts = pd.cut(df['viral_score'], bins=bins).value_counts().sort_index()
    
    # Calculate percentages
    bin_percentages = (bin_counts / len(df)) * 100
    
    # Print histogram
    for i, (bin_range, count) in enumerate(bin_counts.items()):
        percentage = bin_percentages[bin_range]
        bar_length = int(percentage)
        print(f"{bin_range}: {count:,} ({percentage:.1f}%) {'█' * bar_length}")
    
    # Log information about the score distribution for monitoring
    logging.info(f"Viral score diversity: unique values: {df['viral_score'].nunique()}/{len(df)}")
    
    return df

def clean_data(df):
    """
    Clean and normalize the combined dataset.
    Adds detailed logging to show how many rows are dropped at each step.
    
    Args:
        df (pd.DataFrame): Combined DataFrame with video data
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("clean_data")

    logger.info(f"Initial: {len(df):,} rows")
    logger.info(f"  - Missing title: {(df['title'].isna() | (df['title'] == '')).sum():,}")
    logger.info(f"  - Missing description: {(df['description'].isna() | (df['description'] == '')).sum():,}")
    logger.info(f"  - Missing publishedAt: {df['publishedAt'].isna().sum():,}")
    logger.info(f"  - Unique videoId: {df['videoId'].nunique():,}")

    # Print group count by region
    #region_counts = df['region'].value_counts(dropna=False)
    #logger.info("Rows per region before region filter:")
    #for region, count in region_counts.items():
    #    logger.info(f"  {region}: {count:,}")


    # Filter by regions if needed
    df_filtered = df.copy()
    if "region" in df.columns:
        
        before = len(df_filtered)
        df_filtered = df.query("region.isna() or region in @KEEP_REGIONS").copy()
        logger.info(f"After region filter: {len(df_filtered):,} rows (dropped {before - len(df_filtered):,})")
        
    # Remove duplicates
    before = len(df_filtered)
    df_deduped = df_filtered.drop_duplicates(
        subset=["videoId", "trendingDate", "region"], 
        keep="last"
    ).copy()
    logger.info(f"After deduplication: {len(df_deduped):,} rows (dropped {before - len(df_deduped):,})")

    # Convert numeric columns
    for n in ["viewCount", "likeCount", "commentCount", "rank"]:
        nulls_before = df_deduped[n].isna().sum()
        df_deduped.loc[:, n] = pd.to_numeric(df_deduped[n], errors="coerce").fillna(0).astype(int)
        nulls_after = df_deduped[n].isna().sum()
        logger.info(f"  - {n}: {nulls_before:,} nulls before, {nulls_after:,} after conversion")

    # Log after numeric conversion
    logger.info(f"After numeric conversion: {len(df_deduped):,} rows")

    # Calculate viral scores
    before = len(df_deduped)
    df_with_scores = add_viral(df_deduped)
    logger.info(f"After viral score calculation: {len(df_with_scores):,} rows (should be unchanged)")

    # Log viral_score statistics
    vs = df_with_scores['viral_score']
    logger.info(f"viral_score stats: min={vs.min():.4f}, max={vs.max():.4f}, mean={vs.mean():.4f}, median={vs.median():.4f}")
    for thresh in [0.05, 0.10, 0.20, 0.40]:
        logger.info(f"  - Rows with viral_score >= {thresh:.2f}: {(vs >= thresh).sum():,}")

    # Log missing critical fields after all processing
    logger.info(f"Final: {len(df_with_scores):,} rows")
    logger.info(f"  - Missing title: {(df_with_scores['title'].isna() | (df_with_scores['title'] == '')).sum():,}")
    logger.info(f"  - Missing description: {(df_with_scores['description'].isna() | (df_with_scores['description'] == '')).sum():,}")
    logger.info(f"  - Missing publishedAt: {df_with_scores['publishedAt'].isna().sum():,}")
    logger.info(f"  - Unique videoId: {df_with_scores['videoId'].nunique():,}")

    return df_with_scores 