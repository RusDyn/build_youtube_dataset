# YouTube Dataset Builder

This script does the following:
1. Downloads YouTube dataset data from S3 (if exists)
2. Merges Kaggle and YouTube API data
3. Writes the database to a local DuckDB file
4. Uploads the file back to S3

## Setup

1. Create a virtual environment:
```
python -m venv venv
```

2. Activate the virtual environment:
- Windows: `.\venv\Scripts\Activate.ps1` or `.\venv\Scripts\activate.bat`
- Unix/Mac: `source venv/bin/activate`

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Create a `.env` file with the following environment variables:
```
# S3 Configuration
S3_BUCKET=your_bucket_name
S3_KEY=warehouse/youtube_dataset.duckdb

# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=your_aws_region

# YouTube API Key
YT_API_KEY=your_youtube_api_key

# Kaggle Credentials
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

## Usage

Run the script:
```
python build_youtube_dataset.py
```

To test the script without connecting to external services, use the dry-run option:
```
python build_youtube_dataset.py --dry-run
```

## Requirements

- Python 3.6+
- AWS credentials with S3 access
- YouTube Data API key
- Kaggle API credentials