#!/usr/bin/env python3
"""
Configuration module for YouTube dataset builder.
Handles environment variables and constants.
"""
import os
from pathlib import Path
from datetime import datetime, timezone

# Try to load environment variables from .env or test_env.txt
try:
    from dotenv import load_dotenv
    if Path('.env').exists():
        load_dotenv('.env')
    elif Path('test_env.txt').exists():
        load_dotenv('test_env.txt')
except ImportError:
    pass  # dotenv is optional

# S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY = os.getenv("S3_KEY", "warehouse/youtube_dataset.duckdb")
LOCAL_DB = Path("/tmp/youtube_dataset.duckdb")

# YouTube API Configuration
YT_API_KEY = os.getenv("YT_API_KEY")
YOUTUBE_REGIONS = ["US", "GB", "IN"]
MAX_RESULTS_API = 50

# Kaggle Configuration
KAGGLE_SLUGS = {
    "daily": "pyuser11/youtube-trending-videos-updated-daily",
    "113c": "asaniczka/trending-youtube-videos-113-countries",
}
# Expanded KEEP_REGIONS: EU countries + US, CA, AU, NZ, GB, CH, NO, IS
KEEP_REGIONS = {
    # EU countries
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    # Other similar countries
    "US", "CA", "AU", "NZ", "GB", "CH", "NO", "IS",
    # Add more if needed
}

# Time Configuration
NOW_UTC = datetime.now(timezone.utc)

def check_environment(dry_run=False):
    """
    Check if required environment variables are set.
    
    Args:
        dry_run (bool): If True, skip environment checks
    """
    if dry_run:
        print("🔍 Running in DRY RUN mode - will not connect to external services")
        return
        
    for var in ("S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        if not os.getenv(var):
            raise EnvironmentError(f"❌  {var} env-var is missing – aborting.") 