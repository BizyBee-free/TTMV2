"""HMM regime-detection trading strategy package.

Modules:
    feature_engineer -- Raw OhlcBar → normalised feature matrix (log_ret, range, vol_change)
    regime_model     -- GaussianHMM wrapper (fit, predict_proba, auto state labelling)
    hmm_signal       -- State probability → trade direction with confidence filter

Quick-start::

    from src.hmm import HMMConfig, HMMFeatureEngineer, HMMRegimeModel, HMMSignalGenerator

    cfg    = HMMConfig(k_states=3, n_iter=1000)
    eng    = HMMFeatureEngineer(cfg)
    model  = HMMRegimeModel(cfg)
    X      = eng.compute(bars)
    model.fit(X)
    proba  = model.get_latest_state_proba(X)
    signal = HMMSignalGenerator(model.state_labels)
    direction = signal.get_direction(proba, confidence_threshold=0.65)
"""

from src.hmm.feature_engineer import HMMConfig, HMMFeatureEngineer
from src.hmm.regime_model import HMMRegimeModel, STATE_BULL, STATE_FLAT, STATE_BEAR
from src.hmm.hmm_signal import HMMSignalGenerator, LONG, SHORT, FLAT

__all__ = [
    "HMMConfig",
    "HMMFeatureEngineer",
    "HMMRegimeModel",
    "STATE_BULL",
    "STATE_FLAT",
    "STATE_BEAR",
    "HMMSignalGenerator",
    "LONG",
    "SHORT",
    "FLAT",
]
