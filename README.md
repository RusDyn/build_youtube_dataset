# YouTube Dataset Builder

This package builds a dataset of trending YouTube videos by:
1. Downloads YouTube dataset data from S3 (if exists)
2. Merges Kaggle and YouTube API data
3. Writes the database to a local DuckDB file
4. Uploads the file back to S3

## Project Structure

The codebase has been refactored into a modular architecture:

```
youtube_dataset/
├── __init__.py            # Package initialization
├── config.py              # Configuration settings and environment variables
├── api/                   # YouTube API integration
│   ├── __init__.py
│   └── fetch.py           # Functions to fetch data from YouTube API
├── db/                    # Database operations
│   ├── __init__.py
│   └── operations.py      # DuckDB operations
├── kaggle/                # Kaggle data handling
│   ├── __init__.py
│   ├── downloader.py      # Functions to download Kaggle datasets
│   └── processor.py       # Functions to process Kaggle CSV files
├── processing/            # Data processing
│   ├── __init__.py
│   └── viral_score.py     # Functions to calculate viral scores
└── s3/                    # S3 operations
    ├── __init__.py
    └── operations.py      # Functions to interact with S3

youtube_dataset_builder.py   # Main script that orchestrates the process
download_kaggle_datasets.py  # Helper script to download Kaggle datasets
download_datasets.ps1        # PowerShell script to automate dataset downloads on Windows
```

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

## Dataset Download

Before running the main script, you should download the Kaggle datasets. This can take a significant amount of time depending on your connection speed, especially for the large 113-countries dataset.

### Windows Users
Run the PowerShell script:
```
.\download_datasets.ps1
```

### Manual Download
You can manually download the datasets using the helper script:
```
# Download all datasets
python download_kaggle_datasets.py

# Or download a specific dataset
python download_kaggle_datasets.py --dataset daily
python download_kaggle_datasets.py --dataset 113c

# Force re-download even if files exist
python download_kaggle_datasets.py --force
```

### Known Dataset Issues and Fixes

This package includes fixes for common issues with the Kaggle datasets:

1. **Daily dataset loading failures**: The package now creates sample data if the daily dataset files can't be loaded, ensuring the pipeline continues to run. It also checks multiple subdirectories for CSV files to handle different download structures.

2. **Large 113-countries dataset corruption**: The large CSV file (over 1GB) often has EOF errors and buffer overflow issues. The enhanced CSV reader now attempts multiple methods, including:
   - Reading with specialized dtype options
   - Chunked reading to limit memory usage
   - Fallback to read the first 500,000 rows only
   - Direct CSV reading as a last resort
   
3. **File encoding issues**: The processor now tries multiple encodings (UTF-8, latin1, ISO-8859-1, cp1252) if the default encoding fails.

## Usage

Run the main script:
```
python youtube_dataset_builder.py
```

### Command Line Options

The script supports several command line options:

- `--dry-run`: Test the script without connecting to external services
  ```
  python youtube_dataset_builder.py --dry-run
  ```

- `--analyze`: Analyze the existing database without updating it
  ```
  python youtube_dataset_builder.py --analyze
  ```

- `--force-download`: Force re-download of Kaggle datasets even if cached
  ```
  python youtube_dataset_builder.py --force-download
  ```

## Preparing Training Data

The package includes functionality to prepare training data for viral title prediction models:

```
python prepare_training_data.py
```

This script:
1. Loads YouTube video data from the DuckDB database
2. Calculates improved viral scores using enhanced algorithms
3. Analyzes viral score distribution and creates visualization plots
4. Creates a balanced dataset using stratified sampling across viral score ranges
5. Exports the dataset in Hugging Face format for regression model training

### Training Data Outputs

The script generates several outputs:

- `analysis/viral_score_distribution.png`: Distribution plot of viral scores
- `analysis/correlation_heatmap.png`: Correlation matrix of viral metrics
- `analysis/sample_with_scores.csv`: Sample dataset with calculated scores
- `analysis/sample_training_data.csv`: Sample of the balanced training data
- `hf_dataset_reg_improved/`: Hugging Face dataset for regression training

### Next Steps

After preparing the training data, you can train a regression model using:

```
python train_viral_titles_pro.py --stage regression_title --enhanced --dataset hf_dataset_reg_improved
```

## Requirements

- Python 3.6+
- AWS credentials with S3 access
- YouTube Data API key
- Kaggle API credentials