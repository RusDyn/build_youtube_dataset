#!/usr/bin/env python3
"""
Render‑ready cron script:
  1.  downloads youtube_dataset.duckdb from S3  (if it exists)
  2.  merges Kaggle + YouTube‑API data
  3.  writes /tmp/youtube_dataset.duckdb
  4.  uploads it back to S3 (overwrite)

Env‑vars required:
  S3_BUCKET             e.g.  my‑yt‑datasets
  S3_KEY                e.g.  warehouse/youtube_dataset.duckdb
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  YT_API_KEY            (YouTube Data API key)
  KAGGLE_USERNAME, KAGGLE_KEY
"""
import os, sys, zipfile, subprocess, tempfile, argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd, numpy as np, requests, duckdb, boto3
from dateutil import parser as dtparser
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    # Try to load environment variables from .env or test_env.txt
    if Path('.env').exists():
        load_dotenv('.env')
    elif Path('test_env.txt').exists():
        load_dotenv('test_env.txt')
except ImportError:
    pass  # dotenv is optional

# ╭──────────────────────────── Config ────────────────────────────╮
S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY    = os.getenv("S3_KEY", "warehouse/youtube_dataset.duckdb")
LOCAL_DB  = Path("/tmp/youtube_dataset.duckdb")

YT_API_KEY = os.getenv("YT_API_KEY")
KAGGLE_SLUGS = {
    "daily": "pyuser11/youtube-trending-videos-updated-daily",
    "113c":  "asaniczka/trending-youtube-videos-113-countries",
}
KEEP_REGIONS      = {"US", "GB", "IN"}
YOUTUBE_REGIONS   = ["US", "GB", "IN"]
MAX_RESULTS_API   = 50
NOW_UTC           = datetime.now(timezone.utc)
# ╰────────────────────────────────────────────────────────────────╯

# ─── Basic sanity checks ──────────────────────────────────────────
def check_environment(dry_run=False):
    if dry_run:
        print("🔍 Running in DRY RUN mode - will not connect to external services")
        return
        
    for var in ("S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        if not os.getenv(var):
            sys.exit(f"❌  {var} env‑var is missing – aborting.")

# ─── S3 helpers ───────────────────────────────────────────────────
s3 = None  # Will be initialized when needed

def initialize_s3():
    global s3
    if s3 is None:
        s3 = boto3.client("s3")
    return s3

def s3_download(bucket, key, dest: Path, dry_run=False):
    if dry_run:
        print(f"🔍 DRY RUN: Would download {key} from S3 to {dest}")
        return
        
    try:
        initialize_s3().download_file(bucket, key, str(dest))
        print(f"✔  downloaded {key} from S3")
    except Exception as e:
        if "NoSuchKey" in str(e):
            print("ℹ  No existing warehouse in S3 – will create new file.")
        else:
            print(f"⚠️  Error downloading from S3: {e}")

def s3_upload(bucket, key, src: Path, dry_run=False):
    if dry_run:
        print(f"🔍 DRY RUN: Would upload {src} to S3 as {key}")
        return
        
    initialize_s3().upload_file(str(src), bucket, key)
    print(f"✔  uploaded {key} to S3")

# ─── Kaggle download util ─────────────────────────────────────────
def download_kaggle(slug: str, dest: Path, dry_run=False):
    if dry_run:
        print(f"🔍 DRY RUN: Would download Kaggle dataset {slug} to {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        return
        
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob("*.csv")):   # cached
        return
    try:
        subprocess.check_call(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"])
    except Exception as e:
        print(f"⚠️  Error downloading from Kaggle: {e}")

# ─── Load & harmonise CSVs (same as before, trimmed) ─────────────
def load_csvs(folder: Path, tag: str) -> pd.DataFrame:
    dfs=[]
    for csv in folder.glob("*.csv"):
        df = pd.read_csv(csv, low_memory=False)
        df["source"]=tag
        dfs.append(df)
    if not dfs: return pd.DataFrame()
    df=pd.concat(dfs,ignore_index=True)
    rn={"video_id":"videoId","publish_time":"publishedAt","trending_date":"trendingDate",
        "view_count":"viewCount","likes":"likeCount","comment_count":"commentCount",
        "rank":"rank","country":"region"}
    df=df.rename(columns={k:v for k,v in rn.items() if k in df.columns})
    need=["videoId","title","description","publishedAt","trendingDate","region",
          "viewCount","likeCount","commentCount","rank","source"]
    for c in need:
        if c not in df.columns: df[c]=np.nan
    return df[need]

# ─── YouTube API pull (same logic) ────────────────────────────────
def fetch_api(region:str, dry_run=False)->pd.DataFrame:
    if not YT_API_KEY or dry_run:
        if dry_run:
            print(f"🔍 DRY RUN: Would fetch YouTube API data for region {region}")
            # Return a minimal dummy dataframe for testing
            rows = [{
                "videoId": f"dummy_video_id_{i}",
                "title": f"Dummy Video {i}",
                "description": f"Description for dummy video {i}",
                "publishedAt": NOW_UTC.isoformat(),
                "trendingDate": NOW_UTC.date().isoformat(),
                "region": region,
                "viewCount": i * 1000,
                "likeCount": i * 100,
                "commentCount": i * 10,
                "rank": i,
                "source": "api"
            } for i in range(1, 6)]
            return pd.DataFrame(rows)
        return pd.DataFrame()
    
    url="https://youtube.googleapis.com/youtube/v3/videos"
    params={"part":"snippet,statistics","chart":"mostPopular","regionCode":region,
            "maxResults":MAX_RESULTS_API,"key":YT_API_KEY}
    try:
        r=requests.get(url,params=params,timeout=30)
        r.raise_for_status()
        rows=[]
        for it in r.json().get("items",[]):
            sn,st=it["snippet"],it["statistics"]
            rows.append({"videoId":it["id"],"title":sn["title"],"description":sn["description"],
                         "publishedAt":sn["publishedAt"],"trendingDate":NOW_UTC.date().isoformat(),
                         "region":region,"viewCount":int(st.get("viewCount",0)),
                         "likeCount":int(st.get("likeCount",0)),
                         "commentCount":int(st.get("commentCount",0)),"rank":np.nan,"source":"api"})
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"⚠️  Error fetching from YouTube API: {e}")
        return pd.DataFrame()

# ─── Viral score function (same formula) ─────────────────────────
def add_viral(df):
    df["publishedAt"]=pd.to_datetime(df["publishedAt"],errors="coerce",utc=True)
    df["trendingDate"]=pd.to_datetime(df["trendingDate"],errors="coerce",utc=True)
    snap=df["trendingDate"].fillna(pd.Timestamp(NOW_UTC))
    age=(snap-df["publishedAt"]).dt.total_seconds()/3600
    age=age.replace(0,1).fillna(24)
    df["viewsPerHr"]=df["viewCount"]/age
    df["likesPerHr"]=df["likeCount"]/age
    df["commentsPerHr"]=df["commentCount"]/age
    for c in ["viewsPerHr","likesPerHr","commentsPerHr"]:
        m=df[c].max(); df[c+"_n"]=df[c]/m if m else 0
    rank_score=(51-df["rank"].fillna(50))/50
    df["rankScore_n"]=rank_score.clip(0,1)
    df["viral_score"]=(
        0.45*df["viewsPerHr_n"]+0.25*df["likesPerHr_n"]+
        0.10*df["commentsPerHr_n"]+0.20*df["rankScore_n"]
    ).round(4).fillna(0)
    return df

# ─── Main job ─────────────────────────────────────────────────────
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build YouTube dataset from Kaggle and YouTube API data')
    parser.add_argument('--dry-run', action='store_true', help='Run without connecting to external services')
    args = parser.parse_args()
    
    # Check environment variables
    check_environment(args.dry_run)
    
    # Create temp directories if needed
    os.makedirs("/tmp", exist_ok=True)
    
    # 0. Pull existing warehouse, if any
    s3_download(S3_BUCKET, S3_KEY, LOCAL_DB, args.dry_run)

    workdir=Path("/tmp/kaggle")
    for tag,slug in KAGGLE_SLUGS.items():
        download_kaggle(slug, workdir/tag, args.dry_run)

    df_all=pd.concat(
        [load_csvs(workdir/tag, f"kaggle_{tag}") for tag in KAGGLE_SLUGS],
        ignore_index=True,sort=False
    )
    api_frames=[fetch_api(r, args.dry_run) for r in YOUTUBE_REGIONS]
    if api_frames: df_all=pd.concat([df_all]+api_frames, ignore_index=True, sort=False)

    df_all=df_all.query("region.isna() or region in @KEEP_REGIONS").drop_duplicates(
        subset=["videoId","trendingDate","region"], keep="last")
    for n in ["viewCount","likeCount","commentCount","rank"]:
        df_all[n]=pd.to_numeric(df_all[n],errors="coerce").fillna(0).astype(int)

    df_all=add_viral(df_all)

    # In dry-run mode, just print some stats instead of writing to DuckDB
    if args.dry_run:
        print(f"🔍 DRY RUN: Would write {len(df_all)} rows to DuckDB")
        print(f"Data sample:\n{df_all.head()}")
        print(f"🔍 DRY RUN: Would upload DuckDB file to S3")
    else:
        con=duckdb.connect(LOCAL_DB)
        con.execute(f"CREATE TABLE IF NOT EXISTS youtube_videos AS SELECT * FROM df_all LIMIT 0")
        con.register("df_all", df_all)
        con.execute("INSERT INTO youtube_videos SELECT * FROM df_all")
        con.close()

        # 6. push back to S3
        s3_upload(S3_BUCKET, S3_KEY, LOCAL_DB, args.dry_run)
    
    print("✅ cron run complete")

if __name__=="__main__":
    main()
