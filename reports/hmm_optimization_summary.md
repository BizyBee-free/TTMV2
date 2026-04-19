# HMM optimization summary
Config: `config\config.yaml`
Symbol **VN30F1M** | 20250901–20251231

## Objective
Score = 0.3×Calmar + 0.3×Sharpe + 0.4×ProfitFactor (clipped per YAML).

## Features (this run)
- Basis (future vs index): **True** (`index_symbol=VN30`)
- Open interest from cache: **True** (`D:\Chuong\BeeTrade\data\cache\oi\VN30F1M.jsonl`)

## Constraints
- Max drawdown < 25.0%
- Trades > 100
- Profit factor > 1.2
- Increment-shuffle p-value < 0.05

## Top 5 valid configs
*No configuration satisfied all constraints.*


## Top 5 by raw score (constraints ignored)
| # | W | K | TF | Score* | Valid | Trades | MDD% | reason |
|---:|---:|---:|:---|-------:|:-----|-------:|-----:|:-------|
| 1 | 3000 | 4 | 15m | 5.152588 | False | 69 | 13.0255 | trades<=100 |
| 2 | 3000 | 2 | 15m | 5.076592 | False | 604 | 48.7032 | mdd>=25.0 |
| 3 | 1000 | 2 | 15m | 4.933333 | False | 607 | 48.7032 | mdd>=25.0 |
| 4 | 1000 | 4 | 15m | 4.901704 | False | 73 | 13.0255 | trades<=100 |
| 5 | 2000 | 4 | 15m | 4.901704 | False | 73 | 13.0255 | trades<=100 |

*Score* = same weighted formula even if invalid.

## Suggested app defaults
No config passed **all** constraints. Consider longer history (drop `max_bars`), relax thresholds, or validate on a different symbol/period.

**Best-effort (highest raw Score* on live TF, constraints ignored):** `HMM_LIVE_TRAIN_WINDOW_BARS=3000`, `HMM_LIVE_REFIT_EVERY=5`, `HMM_LIVE_K_STATES=4`. Observed MDD≈13.0255%, PF≈1.809274 — review before live use.

Full grid: `reports\hmm_optimization_results.csv`
