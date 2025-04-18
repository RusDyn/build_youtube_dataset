import duckdb
import os
import sys

# Use the same DB_PATH as in the training script
db_path = os.getenv("DB_PATH", "youtube_dataset.duckdb")

if not os.path.exists(db_path):
    sys.exit(f"❌ DuckDB file not found: {db_path}")

print(f"📊 Examining DuckDB file: {db_path} ({os.path.getsize(db_path) / (1024*1024):.2f} MB)")

# Connect to database
con = duckdb.connect(db_path)

# Check tables in database
tables = con.execute("SHOW TABLES").fetchall()
print(f"\n📋 Tables in database:")
for table in tables:
    print(f"  - {table[0]}")

# If we have youtube_videos table, analyze it
if any(table[0] == 'youtube_videos' for table in tables):
    # Get row count
    total_rows = con.execute("SELECT COUNT(*) FROM youtube_videos").fetchone()[0]
    print(f"\n🔍 Table 'youtube_videos' has {total_rows:,} total rows")
    
    # Check for viral_score
    has_viral_score = False
    try:
        columns = con.execute("PRAGMA table_info(youtube_videos)").fetchall()
        print(f"\n📊 Columns in youtube_videos:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
            if col[1] == 'viral_score':
                has_viral_score = True
                
        if has_viral_score:
            # Check viral score distribution
            print("\n📈 Viral score statistics:")
            stats = con.execute("""
                SELECT 
                    MIN(viral_score) as min_score,
                    MAX(viral_score) as max_score,
                    AVG(viral_score) as avg_score,
                    COUNT(*) as count
                FROM youtube_videos
                WHERE viral_score IS NOT NULL
            """).fetchone()
            print(f"  - Min: {stats[0]:.4f}")
            print(f"  - Max: {stats[1]:.4f}")
            print(f"  - Avg: {stats[2]:.4f}")
            print(f"  - Count: {stats[3]:,} rows with non-null viral_score")
            
            # Count number of rows with viral_score >= 0.80
            high_viral = con.execute("""
                SELECT COUNT(*) 
                FROM youtube_videos 
                WHERE viral_score >= 0.80
            """).fetchone()[0]
            print(f"  - {high_viral:,} rows with viral_score >= 0.80")
            
            # Check title and description null values
            null_titles = con.execute("SELECT COUNT(*) FROM youtube_videos WHERE title IS NULL").fetchone()[0]
            null_desc = con.execute("SELECT COUNT(*) FROM youtube_videos WHERE description IS NULL").fetchone()[0]
            print(f"  - {null_titles:,} rows with NULL title")
            print(f"  - {null_desc:,} rows with NULL description")
            
            # Check the original query from the train_viral_titles_pro.py script
            matching_rows = con.execute("""
                SELECT COUNT(*)
                FROM youtube_videos
                WHERE viral_score >= 0.80
                  AND title IS NOT NULL AND description IS NOT NULL
            """).fetchone()[0]
            print(f"\n🎯 {matching_rows:,} rows match all criteria in the query")
            
            # If no rows match, suggest adjustment
            if matching_rows == 0:
                print("\n⚠️ No rows match the current criteria. Testing lower viral score thresholds:")
                for threshold in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                    count = con.execute(f"""
                        SELECT COUNT(*)
                        FROM youtube_videos
                        WHERE viral_score >= {threshold}
                          AND title IS NOT NULL AND description IS NOT NULL
                    """).fetchone()[0]
                    print(f"  - viral_score >= {threshold}: {count:,} rows")
    except Exception as e:
        print(f"\n❌ Error analyzing viral_score: {str(e)}")

# Sample a few rows to see what the data looks like
try:
    print("\n📝 Sample rows from youtube_videos:")
    sample = con.execute("SELECT * FROM youtube_videos LIMIT 3").fetchall()
    columns = [col[0] for col in con.description]
    for row in sample:
        print("\n---Row---")
        for i, val in enumerate(row):
            # Truncate long strings
            if isinstance(val, str) and len(val) > 100:
                val = val[:97] + "..."
            print(f"  {columns[i]}: {val}")
except Exception as e:
    print(f"\n❌ Error fetching sample rows: {str(e)}")

con.close()
print("\n✅ Analysis complete") 