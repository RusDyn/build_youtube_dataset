#!/usr/bin/env python3
"""
YouTube Dataset Builder

Render‑ready cron script:
  1.  downloads youtube_dataset.duckdb from S3  (if it exists)
  2.  merges Kaggle + YouTube‑API data
  3.  writes /tmp/youtube_dataset.duckdb
  4.  uploads it back to S3 (overwrite)

Env‑vars required:
  S3_BUCKET             e.g.  my‑yt‑datasets
  S3_KEY                e.g.  warehouse/youtube_dataset.duckdb
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  YT_API_KEY            (YouTube Data API key)
  KAGGLE_USERNAME, KAGGLE_KEY
"""
import os, sys, argparse
import pandas as pd
from pathlib import Path

# Import modules from the youtube_dataset package
from youtube_dataset.config import (
    S3_BUCKET, S3_KEY, LOCAL_DB, 
    KAGGLE_SLUGS, YOUTUBE_REGIONS,
    check_environment
)
from youtube_dataset.s3 import s3_handler
from youtube_dataset.kaggle import download_kaggle, load_csvs
from youtube_dataset.api import fetch_api
from youtube_dataset.processing import clean_data
from youtube_dataset.db import save_to_duckdb, analyze_database

def main():
    """Main function to build the YouTube dataset"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build YouTube dataset from Kaggle and YouTube API data')
    parser.add_argument('--dry-run', action='store_true', help='Run without connecting to external services')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing database without updating')
    parser.add_argument('--force-download', action='store_true', help='Force re-download of Kaggle datasets even if cached')
    args = parser.parse_args()
    
    # Check environment variables
    try:
        check_environment(args.dry_run)
    except EnvironmentError as e:
        sys.exit(str(e))
    
    # Create temp directories if needed
    os.makedirs("/tmp", exist_ok=True)
    
    # 0. Pull existing warehouse, if any
    s3_handler.download(S3_BUCKET, S3_KEY, LOCAL_DB, args.dry_run)

    # If analyze mode, just analyze the existing database
    if args.analyze:
        analyze_database(LOCAL_DB)
        return

    workdir = Path("/tmp/kaggle")
    
    # Add diagnostic for Kaggle download
    print("\n📥 Checking Kaggle datasets:")
    for tag, slug in KAGGLE_SLUGS.items():
        download_kaggle(slug, workdir/tag, args.dry_run, args.force_download)
        
        # Check if files were downloaded - recursively search for files
        csv_files = []
        csv_gz_files = []
        for root, dirs, files in os.walk(workdir/tag):
            root_path = Path(root)
            csv_files.extend([root_path / f for f in files if f.endswith('.csv')])
            csv_gz_files.extend([root_path / f for f in files if f.endswith('.csv.gz')])
        
        if csv_files or csv_gz_files:
            print(f"  ✓ {tag} dataset: Found {len(csv_files)} CSV files and {len(csv_gz_files)} compressed CSV files")
            
            # Show directory structure
            print(f"    Directory structure:")
            for subdir, _, _ in os.walk(workdir/tag):
                rel_path = Path(subdir).relative_to(workdir/tag)
                if str(rel_path) != '.':  # Don't show the root directory
                    print(f"    - {rel_path}/")
            
            # Show sample of files (max 10)
            if csv_files:
                print(f"    Sample CSV files:")
                for csv_file in csv_files[:5]:
                    rel_path = csv_file.relative_to(workdir/tag)
                    print(f"    - {rel_path} ({csv_file.stat().st_size/1024/1024:.1f} MB)")
                if len(csv_files) > 5:
                    print(f"    - ... and {len(csv_files) - 5} more CSV files")
            
            # Show sample of compressed files
            if csv_gz_files:
                print(f"    Sample compressed CSV files:")
                for csv_gz_file in csv_gz_files[:5]:
                    rel_path = csv_gz_file.relative_to(workdir/tag)
                    print(f"    - {rel_path} ({csv_gz_file.stat().st_size/1024/1024:.1f} MB compressed)")
                if len(csv_gz_files) > 5:
                    print(f"    - ... and {len(csv_gz_files) - 5} more compressed files")
        else:
            print(f"  ❌ {tag} dataset: No CSV files found in {workdir/tag}")
            # list all files in the directory
            print(f"    - {workdir/tag}")
            print(list((workdir/tag).glob("*")))
    
    print("\n🔄 Loading data:")
    # Track each dataset's contribution and store DataFrames
    dataset_dfs = {}
    
    for tag, slug in KAGGLE_SLUGS.items():
        df = load_csvs(workdir/tag, f"kaggle_{tag}")
        # Ensure DataFrame is properly structured before storing
        if isinstance(df, pd.DataFrame):
            # Reset index and ensure no duplicate column names
            print(f"    Columns: {list(df.columns)}")
            df = df.reset_index(drop=True)
            df.columns = pd.Index(df.columns).drop_duplicates(keep='first')
            dataset_dfs[f"kaggle_{tag}"] = df
            print(f"  ✓ Loaded {len(df):,} rows from {tag} dataset")
            
        else:
            print(f"  ⚠️ Warning: No valid DataFrame loaded from {tag} dataset")
    
    # Combine all Kaggle datasets
    if dataset_dfs:
        print("\n  🔄 Concatenating datasets...")
        try:
            df_all = pd.concat(
                list(dataset_dfs.values()),
                ignore_index=True,
                sort=False,
                copy=True
            )
            print(f"  ✓ Combined dataset: {len(df_all):,} rows")
        except Exception as e:
            print(f"  ❌ Error concatenating datasets: {str(e)}")
            # Try alternative concatenation method
            try:
                print("  🔄 Trying alternative concatenation method...")
                dfs_list = list(dataset_dfs.values())
                df_all = dfs_list[0].copy()
                for df in dfs_list[1:]:
                    df_all = pd.concat([df_all, df], ignore_index=True)
                print(f"  ✓ Combined dataset: {len(df_all):,} rows")
            except Exception as e2:
                print(f"  ❌ Alternative method also failed: {str(e2)}")
                raise
    else:
        print("  ⚠️ No datasets to combine")
        df_all = pd.DataFrame()
    
    # Load YouTube API data
    api_frames = [fetch_api(r, args.dry_run) for r in YOUTUBE_REGIONS]
    if api_frames:
        valid_frames = [df.reset_index(drop=True) for df in api_frames if isinstance(df, pd.DataFrame)]
        if valid_frames:
            try:
                df_all = pd.concat([df_all] + valid_frames, ignore_index=True, sort=False)
            except Exception as e:
                print(f"  ❌ Error adding API data: {str(e)}")
                # Try adding one frame at a time
                for frame in valid_frames:
                    try:
                        df_all = pd.concat([df_all, frame], ignore_index=True)
                    except Exception as e2:
                        print(f"  ⚠️ Skipping problematic API frame: {str(e2)}")

    # Clean and process data
    df_all = clean_data(df_all)
    
    # Add diagnostics on date coverage
    date_coverage = df_all.groupby('source')['publishedAt'].apply(
        lambda x: f"{x.notna().sum():,}/{len(x):,} ({100*x.notna().sum()/len(x):.1f}%)")
    print("\n📅 Date coverage by source:")
    for source, coverage in date_coverage.items():
        print(f"  {source}: {coverage}")

    # Save to DuckDB
    save_to_duckdb(df_all, LOCAL_DB, args.dry_run)
    
    # Upload to S3
    s3_handler.upload(S3_BUCKET, S3_KEY, LOCAL_DB, args.dry_run)
    
    print("✅ cron run complete")

if __name__ == "__main__":
    main() 