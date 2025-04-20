"""
Configuration settings for viral titles training pipeline.
"""
import os
import torch
import random
import pathlib
from transformers import BitsAndBytesConfig

# Seeds for reproducibility
SEED = 42
random.seed(SEED)

# Model and data configuration
BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
DB_PATH = pathlib.Path(os.getenv("DB_PATH", "youtube_dataset.duckdb"))
S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY = os.getenv("S3_KEY")

# Text length limits
MAX_LEN = 32
MAX_LEN_TITLE = 64
MAX_LEN_DESC = 256

# CUDA/PyTorch configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

# Configure Windows console for ANSI
def configure_windows_console():
    """Configure Windows console for proper ANSI handling."""
    import sys
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Enable ANSI escape sequence processing
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception as e:
            print(f"Warning: Could not enable ANSI colors in Windows console: {e}")

# Default BitsAndBytes config for loading models
def get_bnb_config():
    """Get default BitsAndBytes configuration."""
    return BitsAndBytesConfig(
        load_in_8bit=True, 
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=torch.float16
    ) 