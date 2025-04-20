#!/usr/bin/env python
"""
Prepare training data with improved viral scores.

This script:
1. Runs the updated viral score calculation on the entire YouTube dataset
2. Analyzes viral score metrics distribution 
3. Creates a balanced dataset using stratified sampling
4. Saves it in the format needed for regression model training
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
import duckdb
import random
from pathlib import Path

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now attempt imports with the modified path
from datasets import Dataset, DatasetDict
try:
    # Try importing from the package
    from youtube_dataset.processing.viral_score import add_viral
    from viral_titles.config import DB_PATH, SEED
    from viral_titles.utils import fetch_duckdb

    # Import schema version info if available
    from youtube_dataset.processing.schema_versions import get_schema_version
    SCHEMA_VERSION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Package import failed: {str(e)}")
    
    # Fallback: try direct imports from local files
    # Adjust these paths based on your actual file structure
    import importlib.util
    
    # Load viral_score
    viral_score_path = os.path.join(project_root, 'youtube_dataset', 'processing', 'viral_score.py')
    if os.path.exists(viral_score_path):
        spec = importlib.util.spec_from_file_location("viral_score", viral_score_path)
        viral_score_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(viral_score_module)
        add_viral = viral_score_module.add_viral
    else:
        raise ImportError(f"Could not find viral_score.py at {viral_score_path}")
    
    # Load config
    from viral_titles.config import DB_PATH, SEED
    
    # Define a simple fetch_duckdb function if the import fails
    def fetch_duckdb():
        """Fallback function to check DB existence"""
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Could not find database at {DB_PATH}")
        return True

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import visualization packages but make them optional
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib and/or Seaborn not installed. Visualization will be disabled.")
    VISUALIZATION_AVAILABLE = False

def load_full_dataset(min_views=100, max_rows=None):
    """
    Load data from the DuckDB database with basic filtering.
    
    Args:
        min_views: Minimum view count (default: 100)
        max_rows: Maximum number of rows to load (None = all)
        
    Returns:
        DataFrame with YouTube video data
    """
    logger.info("Loading YouTube data from DuckDB")
    
    # Make sure we have the database file
    fetch_duckdb()
    
    if not os.path.exists(DB_PATH):
        logger.error(f"Database file not found at {DB_PATH}")
        raise FileNotFoundError(f"Database file not found at {DB_PATH}")
    
    # Connect to the database
    conn = duckdb.connect(str(DB_PATH))
    
    # Basic filters for quality data
    where_clause = f"""
        title IS NOT NULL 
        AND description IS NOT NULL
        AND publishedAt IS NOT NULL
        AND viewCount >= {min_views}
    """
    
    # Count total rows matching criteria
    total_count = conn.execute(f"SELECT COUNT(*) FROM youtube_videos WHERE {where_clause}").fetchone()[0]
    logger.info(f"Total matching rows in database: {total_count:,}")
    
    # Limit clause if max_rows specified
    limit_clause = f"LIMIT {max_rows}" if max_rows else ""
    
    # Query to get videos with basic filters
    query = f"""
    SELECT 
        videoId, title, description, publishedAt, trendingDate, 
        viewCount, likeCount, commentCount, rank, region
    FROM youtube_videos
    WHERE {where_clause}
    ORDER BY RANDOM() -- Randomize to avoid biases
    {limit_clause}
    """
    
    # Execute the query and load into a DataFrame
    rows_to_fetch = max_rows if max_rows else total_count
    logger.info(f"Fetching up to {rows_to_fetch:,} rows with min_views={min_views}")
    df = conn.execute(query).df()
    conn.close()
    
    # Convert date columns to datetime objects
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce', utc=True)
    df['trendingDate'] = pd.to_datetime(df['trendingDate'], errors='coerce', utc=True)
    
    # Log basic stats
    logger.info(f"Loaded {len(df):,} rows from database")
    date_range = (df['publishedAt'].min(), df['publishedAt'].max())
    view_range = (df['viewCount'].min(), df['viewCount'].max())
    logger.info(f"Date range: {date_range[0]} to {date_range[1]}")
    logger.info(f"View count range: {view_range[0]:,} to {view_range[1]:,}")
    
    return df

def calculate_viral_scores(df):
    """
    Calculate improved viral scores using the enhanced algorithm.
    
    Args:
        df: DataFrame with YouTube video data
        
    Returns:
        DataFrame with added viral scores
    """
    logger.info(f"Calculating viral scores for {len(df):,} videos")
    
    # Apply the improved viral score calculation
    df_with_scores = add_viral(df)
    
    # Log viral score statistics
    vs = df_with_scores['viral_score']
    logger.info(f"Viral score stats - min: {vs.min():.4f}, max: {vs.max():.4f}, mean: {vs.mean():.4f}, median: {vs.median():.4f}")
    logger.info(f"Unique viral scores: {vs.nunique():,}/{len(df):,}")
    
    return df_with_scores

def analyze_viral_distribution(df_with_scores):
    """
    Analyze the distribution of viral scores in detail.
    
    Args:
        df_with_scores: DataFrame with calculated viral scores
    """
    logger.info("Analyzing viral score distribution")
    
    # Log basic statistics of viral metrics
    metrics = ['viral_score', 'viral_factor', 'view_performance', 'engagement_ratio', 'trending_score']
    for metric in metrics:
        if metric in df_with_scores.columns:
            values = df_with_scores[metric]
            logger.info(f"{metric} - min: {values.min():.4f}, max: {values.max():.4f}, "
                        f"mean: {values.mean():.4f}, median: {values.median():.4f}")
    
    # Distribution analysis of viral score
    bins = 10
    bin_edges = np.linspace(0, 1, bins+1)
    df_with_scores['bin'] = pd.cut(df_with_scores['viral_score'], 
                                    bins=bin_edges,
                                    labels=[f"bin_{i}" for i in range(bins)])
    
    # Count videos in each bin
    bin_counts = df_with_scores['bin'].value_counts().sort_index()
    logger.info("Viral score distribution across bins:")
    for bin_label, count in bin_counts.items():
        bin_idx = int(bin_label.split('_')[1])
        range_start = bin_edges[bin_idx]
        range_end = bin_edges[bin_idx+1]
        percentage = 100 * count / len(df_with_scores)
        logger.info(f"  {range_start:.2f}-{range_end:.2f}: {count:,} videos ({percentage:.1f}%)")
    
    # Check for any data issues in normalized metrics
    for metric in metrics:
        if metric in df_with_scores.columns:
            values = df_with_scores[metric]
            if values.min() < 0 or values.max() > 1:
                logger.warning(f"Potential data issue: {metric} has values outside [0,1] range")
                
    # Correlation analysis
    correlation_metrics = ['viral_score', 'viral_factor', 'view_performance', 'engagement_ratio', 
                           'trending_score', 'viewCount', 'likeCount', 'commentCount']
    available_metrics = [m for m in correlation_metrics if m in df_with_scores.columns]
    
    if len(available_metrics) > 1:
        corr_matrix = df_with_scores[available_metrics].corr()
        logger.info("Correlation matrix of key metrics:")
        for metric in available_metrics:
            corr_values = ", ".join([f"{other_metric}: {corr_matrix.loc[metric, other_metric]:.2f}" 
                                     for other_metric in available_metrics if other_metric != metric])
            logger.info(f"  {metric} correlates with: {corr_values}")
            
    # Only generate visualizations if matplotlib and seaborn are available
    if VISUALIZATION_AVAILABLE:
        try:
            # Create directory for plots if it doesn't exist
            os.makedirs("analysis", exist_ok=True)
            
            # Plot viral score distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(df_with_scores['viral_score'], bins=50, kde=True)
            plt.title('Distribution of Viral Scores')
            plt.xlabel('Viral Score')
            plt.ylabel('Count')
            plt.savefig('analysis/viral_score_distribution.png')
            logger.info("Saved viral score distribution plot to analysis/viral_score_distribution.png")
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Matrix of Viral Metrics')
            plt.tight_layout()
            plt.savefig('analysis/correlation_heatmap.png')
            logger.info("Saved correlation heatmap to analysis/correlation_heatmap.png")
        except Exception as e:
            logger.warning(f"Could not generate plots: {str(e)}")
    else:
        logger.info("Skipping visualization generation as matplotlib/seaborn are not available.")
        
    # Return binned DataFrame for further analysis
    df_with_scores.drop('bin', axis=1, inplace=True)

def create_stratified_dataset(df, bins=10, samples_per_bin=5000):
    """
    Create a balanced dataset with stratified sampling across viral score ranges.
    
    Args:
        df: DataFrame with calculated viral scores
        bins: Number of bins to divide viral scores into
        samples_per_bin: Maximum samples to take from each bin
        
    Returns:
        DataFrame with balanced samples
    """
    logger.info(f"Creating stratified dataset with {bins} bins, up to {samples_per_bin} samples per bin")
    
    # Create bins for viral scores
    bin_edges = np.linspace(0, 1, bins+1)
    df['bin'] = pd.cut(df['viral_score'], 
                       bins=bin_edges,
                       labels=[f"bin_{i}" for i in range(bins)])
    
    # Log counts per bin
    bin_counts = df['bin'].value_counts().sort_index()
    logger.info("Initial distribution across bins:")
    for bin_label, count in bin_counts.items():
        bin_idx = int(bin_label.split('_')[1])
        range_start = bin_edges[bin_idx]
        range_end = bin_edges[bin_idx+1]
        logger.info(f"  {range_start:.2f}-{range_end:.2f}: {count:,} videos")
    
    # Sample from each bin - use natural sort order for bin labels like "bin_0", "bin_1", etc.
    sampled_dfs = []
    
    # Get sorted unique bin labels with proper handling for NaN
    bin_labels = sorted([label for label in df['bin'].unique() if pd.notna(label)], 
                        key=lambda x: int(x.split('_')[1]))
    
    for bin_label in bin_labels:
        bin_df = df[df['bin'] == bin_label]
        if len(bin_df) > samples_per_bin:
            # If bin has more samples than needed, randomly sample
            bin_sample = bin_df.sample(samples_per_bin, random_state=SEED)
        else:
            # If bin has fewer samples, use all of them
            bin_sample = bin_df
        
        sampled_dfs.append(bin_sample)
        
    # Combine all samples
    balanced_df = pd.concat(sampled_dfs, ignore_index=True)
    logger.info(f"Created balanced dataset with {len(balanced_df):,} samples")
    
    # Drop the bin column
    balanced_df = balanced_df.drop('bin', axis=1)
    
    return balanced_df

def create_huggingface_dataset(df):
    """
    Create a Hugging Face dataset for regression training.
    
    Args:
        df: DataFrame with balanced samples
        
    Returns:
        Path to saved dataset
    """
    logger.info("Creating Hugging Face dataset for regression training")
    
    # Keep only necessary columns
    df_clean = df[['videoId', 'title', 'description', 'viral_score']].copy()
    
    # Create dataset
    ds = Dataset.from_pandas(df_clean)
    
    # Split into train and test
    dataset_dict = ds.train_test_split(test_size=0.1, seed=SEED)
    
    # Save to disk
    output_path = "hf_dataset_reg_improved"
    dataset_dict.save_to_disk(output_path)
    
    # Log stats
    logger.info(f"Dataset saved to {output_path}/")
    logger.info(f"Train set: {len(dataset_dict['train']):,} examples")
    logger.info(f"Test set: {len(dataset_dict['test']):,} examples")
    
    return output_path

def export_sample_dataset(df, output_file="sample_with_scores.csv", sample_size=1000):
    """
    Export a sample of the dataset with scores for analysis.
    
    Args:
        df: DataFrame with calculated viral scores
        output_file: Path to export the CSV file
        sample_size: Number of rows to sample
    """
    # Create the analysis directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Select a random sample
    sample_df = df.sample(min(sample_size, len(df)), random_state=SEED)
    
    # Select relevant columns
    cols_to_keep = ['videoId', 'title', 'publishedAt', 'viewCount', 'likeCount', 
                    'commentCount', 'viral_score', 'viral_factor', 
                    'view_performance', 'engagement_ratio']
    
    # Keep only columns that exist
    export_cols = [col for col in cols_to_keep if col in sample_df.columns]
    sample_df = sample_df[export_cols]
    
    # Export to CSV
    sample_df.to_csv(output_file, index=False)
    logger.info(f"Exported {len(sample_df)} rows to {output_file}")

def main():
    """Main function to run the entire process."""
    logger.info("Starting training data preparation with improved viral scores")
    
    # Load data from database
    df = load_full_dataset(min_views=100, max_rows=100000)  # Limit for testing, remove for full dataset
    
    # Calculate viral scores
    df_with_scores = calculate_viral_scores(df)
    
    # Analyze viral score distribution
    analyze_viral_distribution(df_with_scores)
    
    # Make sure the analysis directory exists
    os.makedirs("analysis", exist_ok=True)
    
    # Export sample for analysis
    export_sample_dataset(df_with_scores, "analysis/sample_with_scores.csv", 1000)
    
    # Create balanced dataset
    balanced_df = create_stratified_dataset(df_with_scores, bins=10, samples_per_bin=5000)
    
    # Create and save HuggingFace dataset
    dataset_path = create_huggingface_dataset(balanced_df)
    
    # Export sample for inspection
    sample_path = "analysis/sample_training_data.csv"
    balanced_df.sample(min(1000, len(balanced_df))).to_csv(sample_path, index=False)
    logger.info(f"Sample data exported to {sample_path}")
    
    logger.info("Training data preparation complete!")
    
    # Log schema version info
    if 'SCHEMA_VERSION_AVAILABLE' in globals() and SCHEMA_VERSION_AVAILABLE:
        try:
            version = get_schema_version()
            logger.info(f"Feature engineering schema version: {version}")
            logger.info("This version is documented in youtube_dataset/processing/schema_versions.py")
        except Exception as e:
            logger.warning(f"Could not retrieve schema version: {e}")
    
    logger.info("To train the regression model, run: python train_viral_titles_pro.py --stage regression_title --enhanced --dataset hf_dataset_reg_improved")
    
if __name__ == "__main__":
    main() 