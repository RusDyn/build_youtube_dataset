"""
Dataset preparation stage functions.
"""
import duckdb
import pandas as pd
from datasets import Dataset
from ..utils import fetch_duckdb
from ..config import SEED, DB_PATH

def stage_prep():
    """
    Prepare the main dataset for the training pipeline.
    """
    fetch_duckdb()
    con = duckdb.connect(DB_PATH)
    
    # Run diagnostic queries to understand data distribution
    print("📊 Data Distribution Analysis:")
    
    # Check total count
    total_count = con.execute("SELECT COUNT(*) FROM youtube_videos").fetchone()[0]
    print(f"  Total videos: {total_count:,}")
    
    # Check viral score distribution
    viral_score_dist = con.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE viral_score >= 0.20) as vs_20_plus,
            COUNT(*) FILTER (WHERE viral_score >= 0.205) as vs_205_plus,
            COUNT(*) FILTER (WHERE viral_score >= 0.21) as vs_21_plus,
            COUNT(*) FILTER (WHERE viral_score >= 0.215) as vs_215_plus,
            COUNT(*) FILTER (WHERE viral_score >= 0.22) as vs_22_plus
        FROM youtube_videos
    """).fetchone()
    
    print(f"  Viral score distribution:")
    print(f"    ≥ 0.20: {viral_score_dist[0]:,}")
    print(f"    ≥ 0.205: {viral_score_dist[1]:,}")
    print(f"    ≥ 0.21: {viral_score_dist[2]:,}")
    print(f"    ≥ 0.215: {viral_score_dist[3]:,}")
    print(f"    ≥ 0.22: {viral_score_dist[4]:,}")
    
    # Check date distribution
    date_dist = con.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE publishedAt >= (CURRENT_DATE - INTERVAL '1 years')) as last_1yr,
            COUNT(*) FILTER (WHERE publishedAt >= (CURRENT_DATE - INTERVAL '3 years')) as last_3yr,
            COUNT(*) FILTER (WHERE publishedAt >= (CURRENT_DATE - INTERVAL '5 years')) as last_5yr,
            COUNT(*) FILTER (WHERE publishedAt IS NOT NULL) as with_date,
            COUNT(*) FILTER (WHERE publishedAt IS NULL) as without_date
        FROM youtube_videos
    """).fetchone()
    
    print(f"  Date distribution:")
    print(f"    Last 1 year: {date_dist[0]:,}")
    print(f"    Last 3 years: {date_dist[1]:,}")
    print(f"    Last 5 years: {date_dist[2]:,}")
    print(f"    With date: {date_dist[3]:,}")
    print(f"    Without date: {date_dist[4]:,}")
    
    # Analyze combined filters
    combined_counts = con.execute("""
        SELECT
            COUNT(*) FILTER (
                WHERE viral_score >= 0.10 
                AND title IS NOT NULL AND description IS NOT NULL
                AND (publishedAt >= (CURRENT_DATE - INTERVAL '5 years') OR publishedAt IS NULL)
            ) as current_filter,
            COUNT(*) FILTER (
                WHERE viral_score >= 0.05
                AND title IS NOT NULL AND description IS NOT NULL
            ) as relaxed_filter
        FROM youtube_videos
    """).fetchone()
    
    print(f"  Combined filters:")
    print(f"    Current filter (VS≥0.10, with date handling): {combined_counts[0]:,}")
    print(f"    Relaxed filter (VS≥0.05, no date filter): {combined_counts[1]:,}")
    
    # Modified main query
    df = con.execute("""
        SELECT title, description, viral_score
        FROM youtube_videos
        WHERE title IS NOT NULL 
          AND description IS NOT NULL
        ORDER BY random()
    """).df()
    con.close()
    print(f"✓ loaded {len(df):,} rows for training (full range)")
    
    # Sanity check for duplicates
    n_dupes = df.duplicated(subset=["title", "description"]).sum()
    if n_dupes > 0:
        print(f"❌ WARNING: Found {n_dupes} duplicate rows (by title+description). Dropping duplicates.")
        df = df.drop_duplicates(subset=["title", "description"]).reset_index(drop=True)
        print(f"  After dropping duplicates: {len(df):,} rows remain.")
    else:
        print("✓ No duplicate rows found (by title+description).")
        
    # build prompts
    data = []
    for _, r in df.iterrows():
        title = r['title'] or ""
        desc  = (r['description'] or "").strip()[:300]

        prompt = (
            "### Instruction\n"  # keep format consistent across tasks
            "Write a viral YouTube title and a 300‑character description.\n\n"
            "### Input\n{\n  \"topic\": \"PLACEHOLDER\"\n}\n\n"
            "### Response\nTitle:"
        )

        resp = title + (f"\nDescription: {desc}" if desc else "")
        data.append({
            "prompt": prompt,
            "response": resp,
            "score": float(r['viral_score']),
            "title": title,
            "description": desc
        })
        
    ds = Dataset.from_list(data)
    ds = ds.shuffle(SEED)
    split = ds.train_test_split(test_size=0.1, seed=SEED)
    split.save_to_disk("hf_dataset")
    print("✅ Dataset saved ➜ hf_dataset/")
    
    return True

def stage_prep_regression():
    """
    Prepare a dataset specifically for regression models with full range of viral scores.
    Uses stratified sampling to ensure better distribution across viral score ranges.
    """
    print("▶️ Preparing regression dataset with balanced viral score distribution")
    fetch_duckdb()
    con = duckdb.connect(DB_PATH)
    
    # Count distribution of viral scores
    viral_dist = con.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE viral_score < 0.05) as vs_lt_05,
            COUNT(*) FILTER (WHERE viral_score >= 0.05 AND viral_score < 0.10) as vs_05_10,
            COUNT(*) FILTER (WHERE viral_score >= 0.10 AND viral_score < 0.15) as vs_10_15,
            COUNT(*) FILTER (WHERE viral_score >= 0.15 AND viral_score < 0.20) as vs_15_20,
            COUNT(*) FILTER (WHERE viral_score >= 0.20 AND viral_score < 0.21) as vs_20_21,
            COUNT(*) FILTER (WHERE viral_score >= 0.21) as vs_21_plus,
            COUNT(*) FILTER (WHERE viral_score IS NOT NULL) as total_with_score
        FROM youtube_videos
        WHERE title IS NOT NULL AND description IS NOT NULL
    """).fetchone()
    
    print(f"  Original viral score distribution:")
    print(f"    < 0.05: {viral_dist[0]:,}")
    print(f"    0.05 - 0.10: {viral_dist[1]:,}")
    print(f"    0.10 - 0.15: {viral_dist[2]:,}")
    print(f"    0.15 - 0.20: {viral_dist[3]:,}")
    print(f"    0.20 - 0.21: {viral_dist[4]:,}")
    print(f"    ≥ 0.21: {viral_dist[5]:,}")
    print(f"    Total with scores: {viral_dist[6]:,}")
    
    # Define target number of samples per bucket (aiming for more balanced distribution)
    # We'll collect a maximum of samples_per_bucket from each range
    # For the 0.20-0.21 bucket that dominates the dataset, we'll undersample
    samples_per_bucket = 8000
    
    # Query data in stratified manner to get more balanced distribution
    df_segments = []
    
    # Define viral score buckets and fetch balanced samples
    viral_buckets = [
        (0.00, 0.05),
        (0.05, 0.10),
        (0.10, 0.15),
        (0.15, 0.18),
        (0.18, 0.20),
        (0.20, 0.205),  # Split the dominant 0.20-0.21 range
        (0.205, 0.21),
        (0.21, 0.25),
        (0.25, 1.00)
    ]
    
    print("\n  Collecting stratified samples:")
    
    for low, high in viral_buckets:
        # Query segment with specific viral score range
        segment_df = con.execute(f"""
            SELECT title, description, viral_score
            FROM youtube_videos
            WHERE title IS NOT NULL 
              AND description IS NOT NULL
              AND viral_score >= {low}
              AND viral_score < {high}
            ORDER BY random()
            LIMIT {samples_per_bucket}
        """).df()
        
        df_segments.append(segment_df)
        print(f"    {low:.3f} - {high:.3f}: {len(segment_df):,} samples")
    
    # Combine all segments
    df = pd.concat(df_segments, ignore_index=True)
    con.close()
    
    print(f"\n✓ Created balanced dataset with {len(df):,} rows for regression training")
    
    # Report distribution in final dataset
    df['bucket'] = pd.cut(df['viral_score'], 
                          bins=[0, 0.05, 0.10, 0.15, 0.20, 0.21, 0.25, 1.0],
                          labels=['0.00-0.05', '0.05-0.10', '0.10-0.15', '0.15-0.20', 
                                 '0.20-0.21', '0.21-0.25', '0.25-1.00'])
    
    bucket_counts = df['bucket'].value_counts().sort_index()
    print("\n  Final viral score distribution:")
    for bucket, count in bucket_counts.items():
        percentage = 100 * count / len(df)
        print(f"    {bucket}: {count:,} samples ({percentage:.2f}%)")
    
    # Drop the temporary bucket column
    df = df.drop(columns=['bucket'])
    
    # Sanity check for duplicates
    n_dupes = df.duplicated(subset=["title", "description"]).sum()
    if n_dupes > 0:
        print(f"❌ WARNING: Found {n_dupes} duplicate rows (by title+description). Dropping duplicates.")
        df = df.drop_duplicates(subset=["title", "description"]).reset_index(drop=True)
        print(f"  After dropping duplicates: {len(df):,} rows remain.")
    else:
        print("✓ No duplicate rows found (by title+description).")
    
    # Create the dataset
    ds = Dataset.from_pandas(df)
    split = ds.train_test_split(test_size=0.1, seed=SEED)
    split.save_to_disk("hf_dataset_reg")
    print("✅ Regression dataset saved ➜ hf_dataset_reg/")
    
    return True 