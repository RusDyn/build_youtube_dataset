"""
Viral Titles Training Package.
"""

from .config import (
    BASE_MODEL,
    DB_PATH,
    S3_BUCKET,
    S3_KEY,
    SEED,
    MAX_LEN,
    MAX_LEN_TITLE,
    MAX_LEN_DESC,
    configure_windows_console,
    get_bnb_config
)

from .stages import (
    stage_prep,
    stage_prep_regression,
    stage_sft,
    stage_reward,
    stage_rlhf,
    stage_regression
)

from .utils import (
    fetch_duckdb,
    sanity_check_dataset,
    analyze_viral_score_distribution,
    fix_biased_dataset,
    SpearmanCallback,
    PairwiseRankingLoss
)

__all__ = [
    # Config
    'BASE_MODEL',
    'DB_PATH',
    'S3_BUCKET',
    'S3_KEY',
    'SEED',
    'MAX_LEN',
    'MAX_LEN_TITLE',
    'MAX_LEN_DESC',
    'configure_windows_console',
    'get_bnb_config',
    
    # Stages
    'stage_prep',
    'stage_prep_regression',
    'stage_sft',
    'stage_reward',
    'stage_rlhf',
    'stage_regression',
    
    # Utils
    'fetch_duckdb',
    'sanity_check_dataset',
    'analyze_viral_score_distribution',
    'fix_biased_dataset',
    'SpearmanCallback',
    'PairwiseRankingLoss'
] 