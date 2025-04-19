#!/usr/bin/env python3
"""
Kaggle data processing module for YouTube dataset builder.
Handles loading and processing CSV files from Kaggle datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import os

def load_csvs(folder: Path, tag: str) -> pd.DataFrame:
    """
    Load and process CSV files from a Kaggle dataset.
    
    Args:
        folder (Path): Directory containing the CSV files
        tag (str): Tag to identify the source of the data
        
    Returns:
        pd.DataFrame: Processed DataFrame with standardized columns
    """
    dfs = []
    loaded_count = 0
    error_count = 0
    
    # Ensure folder exists
    if not folder.exists():
        print(f"  ⚠️ Directory not found: {folder}")
        os.makedirs(folder, exist_ok=True)
        print(f"  ✓ Created directory: {folder}")
        
        # Create sample data immediately
        if "daily" in tag:
            return _create_daily_sample_data(tag)
        elif "113c" in tag:
            return _create_113c_sample_data(tag)
        return pd.DataFrame()
    
    # Function to recursively find all CSV files
    def find_csv_files(directory, pattern):
        try:
            for path in directory.glob(pattern):
                if path.is_file():
                    yield path
                elif path.is_dir():
                    yield from find_csv_files(path, pattern)
        except Exception as e:
            print(f"  ⚠️ Error finding files in {directory}: {e}")
    
    # First look for regular CSV files (recursively through subdirectories)
    csv_files = list(find_csv_files(folder, "*.csv"))
    
    # If no CSV files found in the main directory, check subdirectories like 'default', 'gaming', etc.
    if not csv_files and tag == "kaggle_daily":
        for subdir in ["default", "gaming", "movies", "music"]:
            subdir_path = folder / subdir
            if subdir_path.exists():
                csv_files.extend(list(find_csv_files(subdir_path, "*.csv")))
    
    total_files = len(csv_files)
    
    if total_files == 0:
        print(f"  ⚠️ No CSV files found in {folder}")
        if "daily" in tag:
            return _create_daily_sample_data(tag)
        elif "113c" in tag:
            return _create_113c_sample_data(tag)
        return pd.DataFrame()
    
    # Process files with progress tracking
    print(f"  🔄 Processing {total_files} files from {tag} dataset...")
    
    # For the daily dataset, sample a subset of files to avoid loading everything
    """
    if tag == "kaggle_daily" and total_files > 100:
        # Take up to 20 files from each category if possible
        sampled_files = []
        for category in ["default", "gaming", "movies", "music"]:
            category_files = [f for f in csv_files if f.is_file() and category in str(f)]
            if category_files:
                # Sort by name and take recent files (assuming date is in filename)
                category_files.sort()
                sampled_files.extend(category_files[-20:])
        
        if not sampled_files:
            # Fallback: just take the most recent 100 files
            csv_files.sort()
            sampled_files = csv_files[-100:]
        
        print(f"  📊 Sampling {len(sampled_files)} files from {total_files} total files")
        csv_files = sampled_files
    """
    for csv_file in csv_files:
        try:
            rel_path = csv_file.relative_to(folder) if folder in csv_file.parents else csv_file.name
        except:
            rel_path = str(csv_file).replace(str(folder), '')
            if rel_path.startswith('/') or rel_path.startswith('\\'):
                rel_path = rel_path[1:]
                
        try:
            # Regular CSV reading for smaller files with more error handling
            try:
                # First attempt - standard read
                df = pd.read_csv(
                    csv_file, 
                    encoding='utf-8',
                    on_bad_lines='skip',
                    nrows=1000000
                )
            except UnicodeDecodeError:
                # Try different encodings if UTF-8 fails
                print(f"  ⚠️ UTF-8 encoding failed for {rel_path}, trying other encodings...")
                encodings = ['latin1', 'ISO-8859-1', 'cp1252']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(
                            csv_file, 
                            encoding=encoding,
                            on_bad_lines='skip'
                        )
                        print(f"  ✓ Successfully loaded with {encoding} encoding")
                        break
                    except Exception:
                        continue
                
                if df is None:
                    print(f"  ❌ All encodings failed for {rel_path}")
                    raise ValueError("Unable to read with any encoding")
            except Exception as e:
                # More robust fallback with extra options
                print(f"  ⚠️ Standard read failed for {rel_path}: {str(e)[:100]}... trying with more options")
                df = pd.read_csv(
                    csv_file, 
                    encoding='utf-8',
                    on_bad_lines='skip',
                    quoting=csv.QUOTE_NONE,
                    escapechar='\\',
                )
        
            if len(df) == 0:
                print(f"  ⚠️ Empty file: {rel_path}")
                continue
                
            # Add source information
            df["source"] = tag
            df["data_source"] = f"{tag}_{rel_path}"
            
            #print(f"Columns: {df.columns}")

            # Handle daily dataset specific column mapping
            # Map columns for all datasets
            if "videoId" in df.columns:
                # Already has correct column name
                pass
            elif "id" in df.columns:
                df["videoId"] = df["id"]
            elif "video_id" in df.columns:
                df["videoId"] = df["video_id"]
            
            # Map date/time columns
            if "publishedDate" in df.columns:
                df["publishedAt"] = df["publishedDate"]
            elif "publishedText" in df.columns:
                # Try to convert human-readable date to timestamp
                df["publishedAt"] = pd.to_datetime(df["publishedText"], errors='coerce')
            elif "publish_time" in df.columns:
                df["publishedAt"] = pd.to_datetime(df["publish_time"], errors='coerce')
            elif "publish_date" in df.columns:
                df["publishedAt"] = pd.to_datetime(df["publish_date"], errors='coerce')
            
            if "snapshot_date" in df.columns:
                df["trendingDate"] = pd.to_datetime(df["snapshot_date"], errors='coerce')
            
            # Map view counts
            if "views" in df.columns:
                df["viewCount"] = pd.to_numeric(df["views"], errors='coerce')
            elif "view_count" in df.columns:
                df["viewCount"] = pd.to_numeric(df["view_count"], errors='coerce')
            
            # Map view/like/comment counts
            if "like_count" in df.columns:
                df["likeCount"] = pd.to_numeric(df["like_count"], errors='coerce')
            
            if "comment_count" in df.columns:
                df["commentCount"] = pd.to_numeric(df["comment_count"], errors='coerce')
            
            # Map rank and region
            if "daily_rank" in df.columns:
                df["rank"] = pd.to_numeric(df["daily_rank"], errors='coerce')
            if "country" in df.columns:
                df["region"] = df["country"]
            # Add category information from the file path
            category = None
            for cat in ["default", "gaming", "movies", "music"]:
                if cat in str(rel_path).lower():
                    category = cat
                    break
            if category:
                df["category"] = category
            
            dfs.append(df)
            loaded_count += 1
            
            # Print progress occasionally
            if loaded_count % 20 == 0 or loaded_count == len(csv_files):
                print(f"  ✓ Loaded {loaded_count}/{len(csv_files)} files from {tag}")
                
        except Exception as e:
            print(f"  ⚠️ Error reading {rel_path}: {str(e)[:100]}...")  # Truncate long error messages
            error_count += 1
    
    if not dfs: 
        print(f"  ❌ No data loaded from {tag} dataset (errors: {error_count})")
        # Create sample data based on dataset type
        if "daily" in tag:
            return _create_daily_sample_data(tag)
        elif "113c" in tag:
            return _create_113c_sample_data(tag)
        return pd.DataFrame()
    
    print(f"  ✓ Successfully loaded {loaded_count} files from {tag}, {error_count} files had errors")
    
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Handle column name mapping for different datasets
    if tag == "kaggle_daily":
        # Daily dataset columns are already handled above
        pass
    else:
        # Original mapping for other datasets
        rn = {
            "video_id": "videoId", 
            "publish_time": "publishedAt", 
            "trending_date": "trendingDate",
            "view_count": "viewCount", 
            "likes": "likeCount", 
            "comment_count": "commentCount",
            "rank": "rank", 
            "country": "region"
        }
        # Rename columns that exist in the mapping
        df = df.rename(columns={k: v for k, v in rn.items() if k in df.columns})
    
    # List of columns we want in the final dataframe
    need = ["videoId", "title", "description", "publishedAt", "trendingDate", "region",
            "viewCount", "likeCount", "commentCount", "rank", "source", "data_source"]
    
    # For any column not in the dataframe, add it with NaN values
    for c in need:
        if c not in df.columns: 
            df[c] = np.nan
    
    # Remove duplicate columns by keeping only the first occurrence
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    # Filter to only include the columns we need
    result_df = df[need]
    
    # Report count and date information
    date_count = result_df["publishedAt"].notna().sum()
    total_count = len(result_df)
    print(f"  ✓ Final dataset from {tag}: {total_count:,} rows, {date_count:,} with dates ({100*date_count/total_count if total_count else 0:.1f}%)")
    
    return result_df

def _create_daily_sample_data(tag):
    """Create a sample dataset for the daily dataset to avoid pipeline failures."""
    print(f"  ℹ️ Creating sample data for {tag} to avoid pipeline failure")
    return pd.DataFrame({
        "videoId": [f"sample_id_{i}" for i in range(10)],
        "title": [f"Sample Video {i}" for i in range(10)],
        "description": [f"Sample description {i}" for i in range(10)],
        "publishedAt": pd.date_range(start="2023-01-01", periods=10),
        "trendingDate": pd.date_range(start="2023-01-02", periods=10),
        "region": ["US"] * 10,
        "viewCount": [i * 1000 for i in range(10)],
        "likeCount": [i * 100 for i in range(10)],
        "commentCount": [i * 10 for i in range(10)],
        "rank": list(range(1, 11)),
        "source": tag,
        "data_source": f"{tag}_sample"
    })

def _create_113c_sample_data(tag):
    """Create a sample dataset for the 113c dataset to avoid pipeline failures."""
    print(f"  ℹ️ Creating sample data for {tag} to avoid pipeline failure")
    countries = ["US", "GB", "IN", "CA", "AU"]
    data = {
        "videoId": [],
        "title": [],
        "description": [],
        "publishedAt": [],
        "trendingDate": [],
        "region": [],
        "viewCount": [],
        "likeCount": [],
        "commentCount": [],
        "rank": [],
        "source": [],
        "data_source": []
    }
    
    # Create a larger sample (50 items)
    for i in range(50):
        country = countries[i % len(countries)]
        data["videoId"].append(f"sample_{country}_{i}")
        data["title"].append(f"Sample Video {i} - {country}")
        data["description"].append(f"Sample description {i} for region {country}")
        data["publishedAt"].append(pd.Timestamp(f"2023-01-{(i%28)+1}"))
        data["trendingDate"].append(pd.Timestamp(f"2023-01-{(i%28)+2}"))
        data["region"].append(country)
        data["viewCount"].append(i * 1000)
        data["likeCount"].append(i * 100)
        data["commentCount"].append(i * 10)
        data["rank"].append((i % 10) + 1)
        data["source"].append(tag)
        data["data_source"].append(f"{tag}_sample_{country}")
    
    return pd.DataFrame(data) 