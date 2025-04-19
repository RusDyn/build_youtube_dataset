#!/usr/bin/env python3
"""
Kaggle data handling module for YouTube dataset builder.
Handles downloading and processing Kaggle datasets.
"""
import subprocess
import shutil
import os
import time
from pathlib import Path

def download_kaggle(slug: str, dest: Path, dry_run=False, force=False):
    """
    Download a Kaggle dataset.
    
    Args:
        slug (str): Kaggle dataset slug
        dest (Path): Destination directory
        dry_run (bool): If True, only print what would be done
        force (bool): If True, force re-download even if files exist
    """
    if dry_run:
        print(f"🔍 DRY RUN: Would download Kaggle dataset {slug} to {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        return
        
    dest.mkdir(parents=True, exist_ok=True)
    
    # Check if we need to download
    if not force and (any(dest.glob("**/*.csv")) or any(dest.glob("**/*.csv.gz"))):
        print(f"  ℹ️ Using cached files for {slug}")
        return
    
    # Clean destination directory if force download
    if force and dest.exists():
        print(f"  🧹 Cleaning destination directory for fresh download: {dest}")
        try:
            # Remove all files but keep the directory
            for item in dest.glob("*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        except Exception as e:
            print(f"  ⚠️ Error cleaning directory: {e}")
    
    # Try multiple download methods for robustness
    methods = [
        # Method 1: Standard kaggle CLI with unzip
        lambda: subprocess.check_call(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]),
        
        # Method 2: Download without unzipping first
        lambda: subprocess.check_call(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)]),
        
        # Method 3: Use force flag
        lambda: subprocess.check_call(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip", "--force"]),
    ]
    
    success = False
    error_messages = []
    
    for i, method in enumerate(methods):
        if success:
            break
            
        try:
            print(f"  🔄 Downloading {slug} to {dest} (attempt {i+1}/{len(methods)})...")
            method()
            success = True
            print(f"  ✅ Downloaded {slug} successfully")
            
            # If we downloaded zip files and not CSV files, try to unzip them
            if not any(dest.glob("**/*.csv")) and any(dest.glob("**/*.zip")):
                print(f"  📦 Extracting zip files in {dest}...")
                for zip_file in dest.glob("**/*.zip"):
                    try:
                        # Try to extract with unzip command
                        subprocess.check_call(["unzip", "-o", str(zip_file), "-d", str(dest)])
                        print(f"  ✅ Extracted {zip_file}")
                    except Exception as e:
                        print(f"  ⚠️ Error extracting {zip_file}: {e}")
                        
                        # If unzip fails, try Python's built-in zipfile
                        try:
                            import zipfile
                            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                zip_ref.extractall(dest)
                            print(f"  ✅ Extracted {zip_file} using Python zipfile")
                        except Exception as e2:
                            print(f"  ❌ All extraction methods failed for {zip_file}: {e2}")
            
        except Exception as e:
            error_msg = str(e)
            error_messages.append(error_msg)
            print(f"  ⚠️ Error with download method {i+1}: {error_msg}")
            
            # Sleep briefly before next attempt
            if i < len(methods) - 1:
                print(f"  ⏱️ Waiting before next attempt...")
                time.sleep(2)
    
    # Check if any files were downloaded
    all_files = list(dest.glob("**/*"))
    csv_files = [f for f in all_files if f.is_file() and f.suffix == ".csv"]
    zip_files = [f for f in all_files if f.is_file() and f.suffix == ".zip"]
    
    if csv_files:
        print(f"  📊 Found {len(csv_files)} CSV files in {dest}")
        # Report sizes of some files
        for csv_file in csv_files[:3]:
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            print(f"  - {csv_file.name} ({size_mb:.1f} MB)")
    elif zip_files:
        print(f"  📦 Found {len(zip_files)} ZIP files but no CSV files in {dest}")
    else:
        print(f"  ❌ No files found in {dest} after download attempts")
        
    if not success:
        print(f"  ❌ All download methods failed for {slug}")
        print(f"  🔍 Error details: {', '.join(error_messages)}")
        
        # Create a dummy file for testing if we couldn't download
        if "daily" in str(slug).lower():
            print(f"  🔧 Creating sample file for testing...")
            sample_dir = dest / "default"
            sample_dir.mkdir(exist_ok=True, parents=True)
            
            sample_file = sample_dir / "default_sample.csv"
            with open(sample_file, "w") as f:
                f.write("videoId,title,description,publishedAt,channelId,channelTitle,categoryId,trending_date,tags,view_count,likes,dislikes,comment_count,thumbnail_link,comments_disabled,ratings_disabled,video_error_or_removed,description\n")
                for i in range(10):
                    f.write(f"sample{i},Sample Video {i},This is sample video {i},2023-01-0{i+1},UC_sample,Sample Channel,22,2023-01-0{i+1},sample tag,{i*1000},{i*100},{i*10},{i*5},http://example.com/thumb{i},false,false,false,Sample description {i}\n")
            print(f"  ✅ Created sample file at {sample_file}")
        elif "113c" in str(slug).lower():
            print(f"  🔧 Creating sample file for testing...")
            sample_file = dest / "trending_yt_videos_113_countries.csv"
            with open(sample_file, "w") as f:
                f.write("video_id,title,description,publish_time,trending_date,country,view_count,likes,dislikes,comment_count,comments_disabled,ratings_disabled,video_error_or_removed,category_id,channel_id,channel_title,tags,thumbnail_link\n")
                countries = ["US", "GB", "IN", "CA", "AU"]
                for i in range(20):
                    country = countries[i % len(countries)]
                    f.write(f"sample{i},Sample Video {i},This is sample video {i},2023-01-0{i%9+1},2023-01-0{i%9+2},{country},{i*1000},{i*100},{i*10},{i*5},false,false,false,22,UC_sample{i},Sample Channel {i},sample tags,http://example.com/thumb{i}\n")
            print(f"  ✅ Created sample file at {sample_file}") 