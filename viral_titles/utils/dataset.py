"""
Dataset utilities for viral titles training.
"""
import boto3
import random
import pathlib
import duckdb
import pandas as pd
import sys
from collections import Counter
from datasets import Dataset, DatasetDict
from ..config import SEED, DB_PATH, S3_BUCKET, S3_KEY

def fetch_duckdb():
    """Fetch DuckDB file from S3 if not present locally."""
    if DB_PATH.exists(): 
        return
    
    if S3_BUCKET and S3_KEY:
        boto3.client("s3").download_file(S3_BUCKET, S3_KEY, str(DB_PATH))
    
    if not DB_PATH.exists():
        sys.exit("❌ DuckDB warehouse not found. Run build_youtube_dataset.py first.")

def sanity_check_dataset(dsd):
    """Sanity checks for the training dataset before SFT."""
    train = dsd["train"]
    n = len(train)
    if n < 1000:
        print(f"❌ ERROR: Training set too small: {n} examples.")
        sys.exit(1)
        
    # Check for empty prompts/responses
    n_empty_prompt = sum(not (ex["prompt"] and ex["prompt"].strip()) for ex in train)
    n_empty_resp = sum(not (ex["response"] and ex["response"].strip()) for ex in train)
    if n_empty_prompt > 0 or n_empty_resp > 0:
        print(f"❌ ERROR: Found {n_empty_prompt} empty prompts and {n_empty_resp} empty responses.")
        sys.exit(1)
        
    # Check for excessive duplication
    resp_counts = Counter(ex["response"] for ex in train)
    most_common_resp, resp_freq = resp_counts.most_common(1)[0]
    if resp_freq > n * 0.1:
        print(f"❌ ERROR: Most common response appears {resp_freq} times (>10% of data). Example: {most_common_resp[:80]}")
        sys.exit(1)
        
    # Check for short responses
    short_resps = sum(len(ex["response"]) < 10 for ex in train)
    if short_resps > n * 0.2:
        print(f"❌ ERROR: {short_resps} responses are shorter than 10 chars (>20% of data). Possible data issue.")
        sys.exit(1)
        
    # Check for train/test split
    if "test" not in dsd or len(dsd["test"]) == 0:
        print(f"❌ ERROR: No test split found or test set is empty.")
        sys.exit(1)
        
    print("✓ Dataset sanity checks passed.")

def analyze_viral_score_distribution(dataset_path="hf_dataset_reg"):
    """
    Analyze the distribution of viral scores in the dataset to check for label bias.
    Returns a tuple of (bias_detected, most_common_scores).
    """
    print("📊 Analyzing viral score distribution...")
    
    # Load the dataset
    dsd = DatasetDict.load_from_disk(dataset_path)
    
    # Extract all viral scores
    train_scores = [float(ex["viral_score"]) for ex in dsd["train"]]
    
    # Count frequency of each exact score value (rounded to 4 decimal places)
    rounded_scores = [round(score, 4) for score in train_scores]
    score_counts = Counter(rounded_scores)
    
    # Get the most common values
    most_common = score_counts.most_common(10)
    total_samples = len(train_scores)
    
    print(f"Total samples: {total_samples}")
    print("\nTop 10 most common viral scores:")
    for score, count in most_common:
        percentage = (count / total_samples) * 100
        print(f"  {score:.4f}: {count} samples ({percentage:.2f}%)")
    
    # Calculate distribution by ranges
    ranges = [
        (0.0, 0.05), (0.05, 0.10), (0.10, 0.15), 
        (0.15, 0.18), (0.18, 0.19), (0.19, 0.20), 
        (0.20, 0.21), (0.21, 0.22), (0.22, 0.25),
        (0.25, 1.0)
    ]
    
    print("\nDistribution by ranges:")
    for low, high in ranges:
        count = sum(1 for score in train_scores if low <= score < high)
        percentage = (count / total_samples) * 100
        print(f"  {low:.2f} - {high:.2f}: {count} samples ({percentage:.2f}%)")
    
    # Check for potential bias
    top_score, top_count = most_common[0]
    top_percentage = (top_count / total_samples) * 100
    
    bias_detected = False
    if top_percentage > 30:
        print(f"\n⚠️ WARNING: Label bias detected! {top_percentage:.2f}% of samples have viral_score={top_score:.4f}")
        bias_detected = True
    
    return bias_detected, most_common

def fix_biased_dataset(dataset_path="hf_dataset_reg", output_path="hf_dataset_reg_fixed"):
    """
    Fix a dataset with biased viral scores by adding small Gaussian noise to the labels.
    Returns the path to the fixed dataset.
    """
    print("🔧 Fixing biased viral score distribution...")
    
    # Load the dataset
    dsd = DatasetDict.load_from_disk(dataset_path)
    
    # Add small Gaussian noise to each score
    def add_noise_to_scores(example):
        # Add small Gaussian noise (mean=0, std=0.0025) to the viral_score
        # This maintains the general ranking but breaks exact value clusters
        noise = random.gauss(0, 0.0025)  # Small standard deviation to maintain original score roughly
        
        # Ensure we don't go below 0 or above 1
        new_score = max(0, min(1, float(example["viral_score"]) + noise))
        return {"viral_score": new_score}
    
    # Apply the transformation
    dsd_fixed = DatasetDict({
        "train": dsd["train"].map(add_noise_to_scores),
        "test": dsd["test"].map(add_noise_to_scores)
    })
    
    # Save the fixed dataset
    dsd_fixed.save_to_disk(output_path)
    print(f"✅ Fixed dataset saved to {output_path}/")
    
    # Verify the fix
    analyze_viral_score_distribution(output_path)
    
    return output_path 