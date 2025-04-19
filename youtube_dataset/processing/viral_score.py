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
    Add viral score metrics to the DataFrame.
    
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
    
    # Normalize engagement metrics by the maximum value
    for c in ["viewsPerHr", "likesPerHr", "commentsPerHr"]:
        m = df[c].max()
        df.loc[:, c + "_n"] = df[c] / m if m else 0
    
    # Calculate rank score (inverted and normalized)
    rank_score = (51 - df["rank"].fillna(50)) / 50
    df.loc[:, "rankScore_n"] = rank_score.clip(0, 1)
    
    # Calculate recency boost
    recency_boost = 1 / (1 + age)
    
    # Calculate final viral score as weighted sum of normalized metrics
    df.loc[:, "viral_score"] = (
        0.45 * df["viewsPerHr_n"] +
        0.25 * df["likesPerHr_n"] +
        0.10 * df["commentsPerHr_n"] +
        0.20 * df["rankScore_n"] +
        0.10 * recency_boost
    ).round(4).fillna(0)
    
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