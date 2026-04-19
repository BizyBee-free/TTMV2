giữ **100% logic TTM V2 (breakout / basis / positioning / volume)**
→ chỉ build **hệ thống tự học để map → alpha → PnL ổn định**.

Dưới đây là **blueprint production-level**, không lý thuyết.

---

# I. Kiến trúc tổng thể (chuẩn bạn cần build)

```text
[TTM Features] 
      ↓
[Feature Store (rolling)]
      ↓
[Empirical Calibration Engine]
      ↓
[Alpha (expected return)]
      ↓
[Execution Filter]
      ↓
[Trade + Log]
      ↓
[Auto Update Loop]
```

---

# II. Step-by-step triển khai

---

# STEP 1 — Chuẩn hóa Feature Store

## Mục tiêu:

Tạo dataset chuẩn để “học từ thị trường”

## Data schema

```python
{
    "timestamp",
    "breakout",
    "basis",
    "positioning",
    "vol",
    "forward_return_4"
}
```

## Logic

```python
dataset.append(current_features)

# sau 4 bars
dataset[i]["forward_return_4"] = (close[t+4] - close[t]) / close[t]
```

## Constraint

```text
window_size = 300–500 bars
rolling = True
```

---

# STEP 2 — Build Empirical Curves (trái tim hệ thống)

## Mục tiêu:

Không đoán — **đo trực tiếp alpha**

---

## 2.1 Binning

```python
bins = quantile(feature, q=5)
```

---

## 2.2 Compute expected return

```python
for bin in bins:
    curve[bin] = mean(forward_return)
```

---

## 2.3 Ví dụ output

```python
breakout_curve = {
    "very_low": +0.0012,
    "low": +0.0004,
    "mid": 0.0,
    "high": -0.0003,
    "very_high": -0.0015
}
```

👉 đây chính là **alpha thực**

---

# STEP 3 — Compute Alpha (core)

```python
def compute_alpha(features):

    alpha = 0

    alpha += breakout_curve[get_bin(features.breakout)]
    alpha += basis_curve[get_bin(features.basis)]
    alpha += positioning_curve[get_bin(features.positioning)]
    alpha += vol_curve[get_bin(features.vol)]

    return alpha
```

---

# STEP 4 — Execution Logic (chuyển alpha → tiền)

## 4.1 Cost-aware threshold

```python
if alpha > cost + margin:
    LONG

elif alpha < -cost - margin:
    SHORT
```

---

## 4.2 No-trade zone

```python
if abs(alpha) < threshold:
    SKIP
```

---

## 4.3 Position sizing

```python
size = clip(alpha / max_alpha, 0.5, 2.0)
```

---

# STEP 5 — Auto-update loop

## Chu kỳ:

```python
update_every = 30 bars
```

---

## Logic

```python
if bar_index % 30 == 0:

    rebuild all curves from latest dataset
```

---

# STEP 6 — Stability control (cực quan trọng)

## 6.1 Min sample per bin

```python
if bin_count < 20:
    fallback_to_global_mean
```

---

## 6.2 Smooth curve

```python
curve[bin] = 0.7 * old + 0.3 * new
```

---

## 6.3 Clip alpha

```python
alpha = clip(alpha, -0.003, 0.003)
```

---

# STEP 7 — Logging (bắt buộc)

```python
log = {
    "alpha": alpha,
    "features": features,
    "bin_mapping": {...},
    "trade": executed_or_not,
    "pnl": ...
}
```

---

# III. Automation Loop hoàn chỉnh

```python
for each bar:

    features = extract()

    dataset.append(features)

    if enough_forward_data:
        update forward_return

    if bar_index % 30 == 0:
        rebuild_curves()

    alpha = compute_alpha(features)

    if abs(alpha) > threshold:
        trade(alpha)
```

---

# IV. Prompt / Setup Cursor (rất cụ thể)

## Prompt 1 — Build feature store

```text
Implement a rolling feature store for TTM V2.

Requirements:
- Store breakout, basis, positioning, vol
- Compute forward_return_4
- Maintain rolling window size 300
- Ensure no data leakage

Output:
- Python class FeatureStore
- append(), update_forward_return(), get_dataset()
```

---

## Prompt 2 — Build empirical calibration

```text
Implement empirical alpha calibration.

Requirements:
- Input: dataset (features + forward return)
- Use quantile binning (5 bins)
- Compute mean return per bin
- Handle low sample bins (fallback)
- Return mapping function feature → expected return

Output:
- build_curve(feature, returns)
- get_bin(value)
- compute_alpha(features)
```

---

## Prompt 3 — Auto update engine

```text
Build auto-update loop for alpha calibration.

Requirements:
- Recompute curves every 30 bars
- Use rolling dataset
- Smooth curve updates (EMA)
- Log all updates

Output:
- AdaptiveAlphaEngine class
```

---

## Prompt 4 — Execution layer

```text
Implement execution logic using alpha.

Requirements:
- Trade only if |alpha| > threshold
- Include transaction cost
- Position sizing proportional to alpha
- Log trades

Output:
- ExecutionEngine class
```

---

# V. Checklist (PM-level)

## Phase 1 — Data

* [ ] Feature store rolling OK
* [ ] Forward return correct
* [ ] No leakage

---

## Phase 2 — Calibration

* [ ] Curves build OK
* [ ] Bin distribution balanced
* [ ] No bin < 20 samples

---

## Phase 3 — Alpha

* [ ] Alpha distribution centered ~0
* [ ] Extreme alpha rare (<5%)

---

## Phase 4 — Execution

* [ ] Trade frequency hợp lý (20–60/ngày)
* [ ] No overtrading

---

## Phase 5 — Validation

* [ ] Rolling PnL dương
* [ ] Late ≥ Early
* [ ] Max DD trong kiểm soát

---

# VI. TODO list (ưu tiên đúng thứ tự)

## Day 1

* [ ] Build feature store
* [ ] Backfill dataset

## Day 2

* [ ] Build curve engine
* [ ] Test mapping

## Day 3

* [ ] Plug alpha vào system
* [ ] Run paper trade

## Day 4

* [ ] Add smoothing + filters

## Day 5

* [ ] Evaluate rolling PnL

---

# VII. Nguyên tắc sống còn (đừng bỏ qua)

## 1.

> Không optimize weight bằng tay nữa

## 2.

> Không cố “đúng logic” — phải “đúng PnL”

## 3.

> Alpha = empirical expectation, không phải cảm giác

---
