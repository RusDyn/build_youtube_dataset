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
    Fix a dataset with biased viral scores using multiple techniques:
    1. Stratified resampling - undersample over-represented bins and oversample rare bins
    2. Add controlled noise to break clusters but preserve overall ranking
    3. Ensure more uniform distribution across the viral score range
    
    Returns the path to the fixed dataset.
    """
    print("🔧 Fixing biased viral score distribution...")
    
    # Load the dataset
    dsd = DatasetDict.load_from_disk(dataset_path)
    
    # Get all train examples
    train_examples = list(dsd["train"])
    total_examples = len(train_examples)
    
    # Analyze the distribution before fixing
    train_scores = [float(ex["viral_score"]) for ex in train_examples]
    
    # Define score bins for stratification
    bins = [
        (0.0, 0.05),
        (0.05, 0.10),
        (0.10, 0.15),
        (0.15, 0.18),
        (0.18, 0.20),
        (0.20, 0.205),
        (0.205, 0.21),
        (0.21, 0.25),
        (0.25, 1.0)
    ]
    
    # Group examples by bin
    binned_examples = {i: [] for i in range(len(bins))}
    
    for ex in train_examples:
        score = float(ex["viral_score"])
        for i, (low, high) in enumerate(bins):
            if low <= score < high:
                binned_examples[i].append(ex)
                break
    
    # Print initial distribution
    print("\nCurrent distribution by bin:")
    for i, (low, high) in enumerate(bins):
        bin_count = len(binned_examples[i])
        bin_pct = 100 * bin_count / total_examples
        print(f"  Bin {i} ({low:.3f}-{high:.3f}): {bin_count} examples ({bin_pct:.2f}%)")
    
    # Target counts for each bin - more balanced but still preserving natural distribution
    # We want to undersample the dominant bins but still keep the distribution somewhat natural
    target_counts = {}
    
    # Identify overrepresented bins (typically around 0.20)
    dominant_bins = []
    for i, examples in binned_examples.items():
        if len(examples) > total_examples * 0.15:  # If bin has >15% of all examples
            dominant_bins.append(i)
    
    # Define target distribution based on observed distribution
    # For overrepresented bins, cap at a percentage of total samples
    # For underrepresented bins, ensure minimum representation
    total_target = 0
    for i, examples in binned_examples.items():
        if i in dominant_bins:
            # Cap dominant bins at 15% 
            target_counts[i] = min(len(examples), int(total_examples * 0.15))
        else:
            # For other bins, ensure at least 2000 examples or what's available
            target_counts[i] = max(min(len(examples), 8000), 2000)
        
        # Update if bin doesn't have enough examples
        if target_counts[i] > len(examples):
            target_counts[i] = len(examples)
        
        total_target += target_counts[i]
    
    # Create resampled examples
    resampled_examples = []
    
    print("\nResampling to target distribution:")
    for i, (low, high) in enumerate(bins):
        bin_examples = binned_examples[i]
        target = target_counts[i]
        
        # Skip empty bins
        if not bin_examples:
            print(f"  Bin {i} ({low:.3f}-{high:.3f}): Empty bin, skipping")
            continue
        
        # If target > available examples, sample with replacement
        if target > len(bin_examples):
            sampled = random.choices(bin_examples, k=target)
        # If target < available examples, sample without replacement
        else:
            sampled = random.sample(bin_examples, k=target)
        
        # Report on sampling
        print(f"  Bin {i} ({low:.3f}-{high:.3f}): {len(bin_examples)} → {len(sampled)} examples")
        
        # Add sampled examples to result list
        resampled_examples.extend(sampled)
    
    # Apply controlled noise to break clusters while preserving ranking
    def add_controlled_noise(examples):
        result = []
        for ex in examples:
            # Copy the example to avoid modifying the original
            new_ex = dict(ex)
            
            # Get bin information for appropriate noise scaling
            score = float(ex["viral_score"])
            bin_idx = None
            bin_width = None
            
            for i, (low, high) in enumerate(bins):
                if low <= score < high:
                    bin_idx = i
                    bin_width = high - low
                    break
            
            if bin_idx is not None:
                # Scale noise based on bin width, with added stochasticity
                # Use smaller noise for narrow bins, larger for wide bins
                # Max noise is 30% of bin width to avoid distorting overall ranking
                noise_scale = bin_width * (0.15 + random.random() * 0.15)
                
                # Add noise with controlled sign to maintain distribution shape
                # For the dominant 0.20 bin, bias noise slightly negative to reduce clustering
                if 0.199 < score < 0.201:
                    noise = random.random() * noise_scale * -1.0  # Slight negative bias
                else:
                    noise = (random.random() - 0.5) * noise_scale * 2.0  # Balanced noise
                
                # Apply noise and ensure within valid range [0, 1]
                new_score = max(0.0, min(1.0, score + noise))
                
                # Ensure we don't introduce new clusters
                if abs(new_score - round(new_score, 4)) < 0.0001:
                    new_score += random.random() * 0.0001  # Break exact values
                
                new_ex["viral_score"] = float(new_score)
            
            result.append(new_ex)
        return result
    
    # Apply the noise transformation
    print("\nApplying controlled noise to break clusters...")
    noisy_examples = add_controlled_noise(resampled_examples)
    
    # Create new datasets
    new_train_dataset = dsd["train"].from_list(noisy_examples)
    
    # Also apply minimal noise to test set to avoid exact values
    test_examples = list(dsd["test"])
    noisy_test = add_controlled_noise(test_examples)
    new_test_dataset = dsd["test"].from_list(noisy_test)
    
    # Create the fixed dataset
    dsd_fixed = DatasetDict({
        "train": new_train_dataset,
        "test": new_test_dataset
    })
    
    # Save the fixed dataset
    dsd_fixed.save_to_disk(output_path)
    print(f"✅ Fixed dataset saved to {output_path}/")
    
    # Verify the fix
    analyze_viral_score_distribution(output_path)
    
    return output_path 