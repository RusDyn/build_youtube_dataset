#!/usr/bin/env python3
"""
Database operations module for YouTube dataset builder.
Handles DuckDB operations for storing and retrieving data.
"""
import pandas as pd
import duckdb
from pathlib import Path

def save_to_duckdb(df: pd.DataFrame, db_path: Path, dry_run=False):
    """
    Save the DataFrame to a DuckDB database.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        db_path (Path): Path to the DuckDB file
        dry_run (bool): If True, only print what would be done
    """
    if dry_run:
        print(f"🔍 DRY RUN: Would write {len(df)} rows to DuckDB at {db_path}")
        print(f"Data sample:\n{df.head()}")
        return
    
    con = duckdb.connect(str(db_path))
    
    # Check if table exists and if it has data_source column
    table_exists = con.execute("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = 'youtube_videos'
    """).fetchone()[0] > 0
    
    if table_exists:
        # Check if data_source column exists
        data_source_exists = con.execute("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_name = 'youtube_videos' AND column_name = 'data_source'
        """).fetchone()[0] > 0
        
        if not data_source_exists:
            print("⚠️ Adding data_source column to existing table")
            con.execute("ALTER TABLE youtube_videos ADD COLUMN data_source VARCHAR")
            
        # Get existing schema to ensure correct type matching
        schema_info = con.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns
            WHERE table_name = 'youtube_videos'
        """).df()
        print(f"📋 Current schema types: {schema_info.set_index('column_name')['data_type'].to_dict()}")
        
        # Insert with explicit column mapping to avoid type mismatches
        columns = con.execute("SELECT * FROM youtube_videos LIMIT 0").description
        column_names = [col[0] for col in columns]
        
        # Only use columns that exist in the target table
        valid_columns = [col for col in df.columns if col in column_names]
        print(f"✓ Inserting data with {len(valid_columns)} columns: {', '.join(valid_columns)}")
        
        # Register dataframe and insert with explicit column list
        con.register("df_input", df)
        con.execute(f"INSERT INTO youtube_videos ({', '.join(valid_columns)}) SELECT {', '.join(valid_columns)} FROM df_input")
    else:
        # Creating new table - let DuckDB infer types from the dataframe
        print("✓ Creating new youtube_videos table")
        con.execute("CREATE TABLE youtube_videos AS SELECT * FROM df_input", {"df_input": df})
    
    con.close()
    print(f"✓ Successfully saved data to {db_path}")

def analyze_database(db_path: Path):
    """
    Analyze the existing database and print statistics.
    
    Args:
        db_path (Path): Path to the DuckDB file
    
    Returns:
        bool: True if analysis was successful, False otherwise
    """
    if not db_path.exists():
        print("❌ No local database found to analyze")
        return False
        
    print("📊 Analyzing existing database...")
    con = duckdb.connect(str(db_path))
    
    # Count total records
    total = con.execute("SELECT COUNT(*) FROM youtube_videos").fetchone()[0]
    print(f"Total videos: {total:,}")
    
    # Analyze date availability by source
    date_stats = con.execute("""
        SELECT 
            source,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE publishedAt IS NOT NULL) as with_date,
            COUNT(*) FILTER (WHERE publishedAt IS NULL) as without_date,
            ROUND(100.0 * COUNT(*) FILTER (WHERE publishedAt IS NOT NULL) / COUNT(*), 2) as date_percentage
        FROM youtube_videos
        GROUP BY source
        ORDER BY total DESC
    """).df()
    
    print("\nDate availability by source:")
    print(date_stats)
    
    # Check if data_source column exists before querying it
    data_source_exists = con.execute("""
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_name = 'youtube_videos' AND column_name = 'data_source'
    """).fetchone()[0] > 0
    
    if data_source_exists:
        # Analyze more detailed data sources
        source_stats = con.execute("""
            SELECT 
                data_source,
                COUNT(*) as count,
                COUNT(*) FILTER (WHERE publishedAt IS NOT NULL) as with_date,
                MIN(publishedAt) as earliest_date,
                MAX(publishedAt) as latest_date
            FROM youtube_videos
            GROUP BY data_source
            ORDER BY count DESC
            LIMIT 20
        """).df()
        
        print("\nTop 20 specific data sources:")
        print(source_stats)
    else:
        print("\n⚠️ data_source column doesn't exist yet in the database")
    
    # Viral score distribution
    viral_stats = con.execute("""
        SELECT 
            source,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE viral_score >= 0.40) as vs_40_plus,
            COUNT(*) FILTER (WHERE viral_score >= 0.20) as vs_20_plus,
            COUNT(*) FILTER (WHERE viral_score >= 0.10) as vs_10_plus,
            COUNT(*) FILTER (WHERE viral_score >= 0.05) as vs_05_plus
        FROM youtube_videos
        GROUP BY source
        ORDER BY total DESC
    """).df()
    
    print("\nViral score distribution by source:")
    print(viral_stats)
    
    con.close()
    return True 