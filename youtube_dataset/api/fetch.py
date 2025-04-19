#!/usr/bin/env python3
"""
YouTube API module for YouTube dataset builder.
Handles fetching data from the YouTube Data API.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from youtube_dataset.config import YT_API_KEY, MAX_RESULTS_API, NOW_UTC

def fetch_api(region: str, dry_run=False) -> pd.DataFrame:
    """
    Fetch trending videos from the YouTube API for a specific region.
    
    Args:
        region (str): Two-letter country code for the region
        dry_run (bool): If True, only print what would be done and return dummy data
        
    Returns:
        pd.DataFrame: DataFrame with video data from the API
    """
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
                "source": "api",
                "data_source": f"youtube_api_{region}_{NOW_UTC.date().isoformat()}"
            } for i in range(1, 6)]
            return pd.DataFrame(rows)
        return pd.DataFrame()
    
    url = "https://youtube.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics",
        "chart": "mostPopular",
        "regionCode": region,
        "maxResults": MAX_RESULTS_API,
        "key": YT_API_KEY
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        
        rows = []
        for it in r.json().get("items", []):
            sn, st = it["snippet"], it["statistics"]
            rows.append({
                "videoId": it["id"],
                "title": sn["title"],
                "description": sn["description"],
                "publishedAt": sn["publishedAt"],
                "trendingDate": NOW_UTC.date().isoformat(),
                "region": region,
                "viewCount": int(st.get("viewCount", 0)),
                "likeCount": int(st.get("likeCount", 0)),
                "commentCount": int(st.get("commentCount", 0)),
                "rank": np.nan,
                "source": "api",
                "data_source": f"youtube_api_{region}_{NOW_UTC.date().isoformat()}"
            })
        
        print(f"✓ Fetched {len(rows)} videos from YouTube API for region {region}")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"⚠️  Error fetching from YouTube API for region {region}: {e}")
        return pd.DataFrame() 