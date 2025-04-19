#!/usr/bin/env python3
"""
S3 operations module for YouTube dataset builder.
Handles downloads and uploads to S3.
"""
import boto3
from pathlib import Path

class S3Handler:
    """S3 operations handler"""
    
    def __init__(self):
        """Initialize S3 client"""
        self.s3 = None
    
    def initialize(self):
        """Initialize the S3 client if it hasn't been initialized yet"""
        if self.s3 is None:
            self.s3 = boto3.client("s3")
        return self.s3
    
    def download(self, bucket, key, dest: Path, dry_run=False):
        """
        Download a file from S3.
        
        Args:
            bucket (str): S3 bucket name
            key (str): S3 object key
            dest (Path): Local destination path
            dry_run (bool): If True, only print what would be done
        """
        if dry_run:
            print(f"🔍 DRY RUN: Would download {key} from S3 to {dest}")
            return
            
        try:
            self.initialize().download_file(bucket, key, str(dest))
            print(f"✔  downloaded {key} from S3")
        except Exception as e:
            if "NoSuchKey" in str(e):
                print("ℹ  No existing warehouse in S3 – will create new file.")
            else:
                print(f"⚠️  Error downloading from S3: {e}")
    
    def upload(self, bucket, key, src: Path, dry_run=False):
        """
        Upload a file to S3.
        
        Args:
            bucket (str): S3 bucket name
            key (str): S3 object key
            src (Path): Local source path
            dry_run (bool): If True, only print what would be done
        """
        if dry_run:
            print(f"🔍 DRY RUN: Would upload {src} to S3 as {key}")
            return
            
        self.initialize().upload_file(str(src), bucket, key)
        print(f"✔  uploaded {key} to S3")

# Create a singleton instance for use across the application
s3_handler = S3Handler() 