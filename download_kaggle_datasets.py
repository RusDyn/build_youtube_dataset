#!/usr/bin/env python3
"""
Script to manually download Kaggle datasets for YouTube trend analysis.
Use this to troubleshoot Kaggle dataset download issues.
"""
import os
import sys
import subprocess
from pathlib import Path

# Import necessary configuration
try:
    from youtube_dataset.config import KAGGLE_SLUGS
except ImportError:
    # Fallback if config isn't available
    KAGGLE_SLUGS = {
        "daily": "pyuser11/youtube-trending-videos-updated-daily",
        "113c": "asaniczka/trending-youtube-videos-113-countries",
    }

def setup_kaggle_credentials():
    """Check and setup Kaggle API credentials."""
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    
    if not kaggle_username or not kaggle_key:
        print("❌ Kaggle credentials not found in environment variables.")
        print("Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        
        # Check if kaggle.json exists
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            print(f"✓ Found Kaggle credentials at {kaggle_json}")
            return True
        
        # Prompt for credentials if not found
        print("\nWould you like to enter your Kaggle credentials now? (y/n)")
        response = input().lower()
        
        if response.startswith("y"):
            username = input("Enter your Kaggle username: ")
            key = input("Enter your Kaggle API key: ")
            
            # Set environment variables
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key
            
            # Create kaggle.json file
            kaggle_dir = Path.home() / ".kaggle"
            kaggle_dir.mkdir(exist_ok=True)
            
            with open(kaggle_dir / "kaggle.json", "w") as f:
                f.write(f'{{"username":"{username}","key":"{key}"}}')
            
            # Set appropriate permissions
            if os.name != "nt":  # Not Windows
                os.chmod(kaggle_dir / "kaggle.json", 0o600)
            
            print("✓ Kaggle credentials saved successfully.")
            return True
        else:
            return False
    
    return True

def download_dataset(slug, dest_dir, force=False):
    """
    Download a Kaggle dataset.
    
    Args:
        slug (str): Kaggle dataset slug (username/dataset-name)
        dest_dir (Path): Destination directory
        force (bool): Whether to force download even if files exist
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files already exist
    if not force and (any(dest_dir.glob("*.csv")) or any(dest_dir.glob("*.csv.gz"))):
        print(f"ℹ️ Files already exist in {dest_dir}. Use --force to re-download.")
        return
    
    try:
        # Ensure kaggle command is available
        try:
            subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("❌ Kaggle CLI not found. Please install it with: pip install kaggle")
            return
        
        print(f"🔄 Downloading dataset: {slug}")
        print(f"   Destination: {dest_dir}")
        
        # Execute kaggle download command
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest_dir), "--unzip"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully downloaded {slug}")
            
            # List downloaded files
            files = list(dest_dir.glob("**/*"))
            csv_files = [f for f in files if f.is_file() and f.suffix == ".csv"]
            
            print(f"📁 Downloaded {len(csv_files)} CSV files:")
            for csv_file in csv_files[:5]:  # Show first 5 files
                rel_path = csv_file.relative_to(dest_dir)
                size_mb = csv_file.stat().st_size / (1024 * 1024)
                print(f"   - {rel_path} ({size_mb:.2f} MB)")
            
            if len(csv_files) > 5:
                print(f"   - ... and {len(csv_files) - 5} more files")
        else:
            print(f"❌ Failed to download dataset: {slug}")
            print(f"Error: {result.stderr}")
    
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")

def main():
    """Main function to download Kaggle datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Kaggle datasets for YouTube trend analysis")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--dataset", choices=list(KAGGLE_SLUGS.keys()), help="Download specific dataset only")
    parser.add_argument("--dest", default="/tmp/kaggle", help="Destination directory")
    
    args = parser.parse_args()
    
    # Setup Kaggle credentials
    if not setup_kaggle_credentials():
        sys.exit("❌ Kaggle credentials setup failed.")
    
    # Create destination directory
    dest_path = Path(args.dest)
    
    # Download datasets
    if args.dataset:
        # Download specific dataset
        if args.dataset in KAGGLE_SLUGS:
            slug = KAGGLE_SLUGS[args.dataset]
            download_dataset(slug, dest_path / args.dataset, args.force)
        else:
            print(f"❌ Unknown dataset: {args.dataset}")
    else:
        # Download all datasets
        for name, slug in KAGGLE_SLUGS.items():
            download_dataset(slug, dest_path / name, args.force)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main() 