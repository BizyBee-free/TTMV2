from src.backtest.data_fetcher import resolve_dnse_symbol
from src.live.hmm_live_runner import _root_cause_from_detail


def test_root_cause_strategy_model():
    assert _root_cause_from_detail("warmup") == "strategy_or_model"
    assert _root_cause_from_detail("model_not_fitted") == "strategy_or_model"


def test_root_cause_system_data_and_ops():
    assert _root_cause_from_detail("fetch_failed") == "system_or_api"
    assert _root_cause_from_detail("basis_align_failed") == "market_data"
    assert _root_cause_from_detail("trading_disabled_external") == "ops_control"


def test_root_cause_risk_and_order_reject():
    assert _root_cause_from_detail("BELOW_MIN_BALANCE") == "risk_or_order_reject"
    assert _root_cause_from_detail("RATE_LIMIT") == "risk_or_order_reject"


def test_resolve_dnse_symbol_friendly_and_krx_nine():
    code, t = resolve_dnse_symbol("VN30F1M")
    assert t == "derivative"
    assert len(code) == 9 and code.startswith("41")
    code9, t9 = resolve_dnse_symbol("41I1F9000")
    assert t9 == "derivative"
    assert code9 == "41I1F9000"
