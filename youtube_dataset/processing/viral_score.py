#!/usr/bin/env python3
"""
Data processing module for YouTube dataset builder.
Handles data transformation and viral score calculations.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone

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
    
    # Calculate final viral score as weighted sum of normalized metrics
    df.loc[:, "viral_score"] = (
        0.45 * df["viewsPerHr_n"] +
        0.25 * df["likesPerHr_n"] +
        0.10 * df["commentsPerHr_n"] +
        0.20 * df["rankScore_n"]
    ).round(4).fillna(0)
    
    return df

def clean_data(df):
    """
    Clean and normalize the combined dataset.
    
    Args:
        df (pd.DataFrame): Combined DataFrame with video data
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Filter by regions if needed
    df_filtered = df.copy()
    if "region" in df.columns:
        from youtube_dataset.config import KEEP_REGIONS
        df_filtered = df.query("region.isna() or region in @KEEP_REGIONS").copy()
    
    # Remove duplicates
    df_deduped = df_filtered.drop_duplicates(
        subset=["videoId", "trendingDate", "region"], 
        keep="last"
    ).copy()
    
    # Convert numeric columns
    for n in ["viewCount", "likeCount", "commentCount", "rank"]:
        df_deduped.loc[:, n] = pd.to_numeric(df_deduped[n], errors="coerce").fillna(0).astype(int)
    
    # Calculate viral scores
    df_with_scores = add_viral(df_deduped)
    
    return df_with_scores 