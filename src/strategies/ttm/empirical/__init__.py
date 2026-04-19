"""Empirical calibration for TTM V2: rolling feature store, bin curves, alpha (see automate_tunning.md)."""

from src.strategies.ttm.empirical.adaptive_engine import EmpiricalAlphaEngine
from src.strategies.ttm.empirical.alpha import compute_empirical_alpha, empirical_alpha_to_score_delta
from src.strategies.ttm.empirical.calibration import (
    CalibrationState,
    build_curves_from_labeled_rows,
    smooth_curves_ema,
)
from src.strategies.ttm.empirical.execution import (
    ExecutionDecision,
    decide_execution,
    position_size_from_alpha,
)
from src.strategies.ttm.empirical.feature_store import (
    EMPIRICAL_FEATURE_KEYS,
    extract_empirical_features_from_last,
    FeatureStoreRow,
    RollingFeatureStore,
    last_dict_from_decision_features,
)

__all__ = [
    "CalibrationState",
    "EMPIRICAL_FEATURE_KEYS",
    "EmpiricalAlphaEngine",
    "ExecutionDecision",
    "FeatureStoreRow",
    "RollingFeatureStore",
    "build_curves_from_labeled_rows",
    "compute_empirical_alpha",
    "decide_execution",
    "empirical_alpha_to_score_delta",
    "extract_empirical_features_from_last",
    "last_dict_from_decision_features",
    "position_size_from_alpha",
    "smooth_curves_ema",
]
