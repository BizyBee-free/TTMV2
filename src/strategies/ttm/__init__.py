"""Trapped Trader Model (TTM) strategy package."""

from src.strategies.ttm.config import (
    TTM_CONFIG,
    build_ttm_live_adaptive_config,
    build_ttm_paper_live_config,
    build_ttm_research_parallel_config,
    is_ttm_adaptive_learning_enabled,
)
from src.strategies.ttm.ttm_alignment import TTMAlignmentError
from src.strategies.ttm.ttm_features import compute_features
from src.strategies.ttm.ttm_score import compute_score, compute_score_v2_alpha, scores_to_probs
from src.strategies.ttm.ttm_signal import generate_ttm_signal, generate_ttm_signal_v1
from src.strategies.ttm.ttm_strategy import TTMStrategy
from src.strategies.ttm.ttm_types import TTMFeatureVector, feature_vector_from_last

__all__ = [
    "TTM_CONFIG",
    "is_ttm_adaptive_learning_enabled",
    "TTMAlignmentError",
    "build_ttm_live_adaptive_config",
    "build_ttm_paper_live_config",
    "build_ttm_research_parallel_config",
    "TTMStrategy",
    "TTMFeatureVector",
    "compute_features",
    "compute_score",
    "compute_score_v2_alpha",
    "feature_vector_from_last",
    "scores_to_probs",
    "generate_ttm_signal",
    "generate_ttm_signal_v1",
]
