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

# Import ensemble and OpenAI embedding utilities
from .utils.ensemble import EnsembleViralPredictor
from .utils.openai_embeddings import get_embedding, batch_get_embeddings

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
    'PairwiseRankingLoss',
    
    # Ensemble and embeddings
    'EnsembleViralPredictor',
    'get_embedding',
    'batch_get_embeddings'
] 