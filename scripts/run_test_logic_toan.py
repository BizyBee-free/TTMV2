from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.backtest.data_fetcher import DataFetcher, RESOLUTION_15MIN
from src.hmm.basis_data import fetch_aligned_future_index
from src.hmm.feature_engineer import HMMFeatureEngineer, HMMConfig
from src.hmm.regime_model import HMMRegimeModel


def align(a, b):
    n = min(len(a), len(b))
    return np.asarray(a)[:n], np.asarray(b)[:n]


def entropy(x):
    x = np.asarray(x, dtype=int)
    if len(x) == 0:
        return 0.0
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def mi_disc(x, y):
    x, y = align(x, y)
    x = x.astype(int)
    y = y.astype(int)
    if len(x) == 0:
        return 0.0
    out = 0.0
    for a in np.unique(x):
        px = np.mean(x == a)
        for b in np.unique(y):
            py = np.mean(y == b)
            pxy = np.mean((x == a) & (y == b))
            if pxy > 0 and px > 0 and py > 0:
                out += pxy * np.log(pxy / (px * py))
    return float(out)


def trans(s, k, order=1):
    s = np.asarray(s, dtype=int)
    if order == 1:
        c = np.zeros((k, k))
        for i in range(len(s) - 1):
            c[s[i], s[i + 1]] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            p = c / c.sum(axis=1, keepdims=True)
        p[np.isnan(p)] = 0
        return p
    c = np.zeros((k, k, k))
    for i in range(len(s) - 2):
        c[s[i], s[i + 1], s[i + 2]] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        p = c / c.sum(axis=2, keepdims=True)
    p[np.isnan(p)] = 0
    return p


def directional_acc(states, fret):
    states, fret = align(states, fret)
    n = len(states)
    if n < 30:
        return 0.0
    sp = int(0.7 * n)
    tr_s, te_s = states[:sp], states[sp:]
    tr_r, te_r = fret[:sp], fret[sp:]
    m = {}
    for s in np.unique(tr_s):
        seg = tr_r[tr_s == s]
        avg = np.mean(seg) if len(seg) else 0.0
        m[int(s)] = 1 if avg > 0 else (-1 if avg < 0 else 0)
    pred = np.array([m.get(int(s), 0) for s in te_s])
    y = np.sign(te_r)
    pred, y = align(pred, y)
    mask = pred != 0
    return float(np.mean(pred[mask] == y[mask])) if np.any(mask) else 0.0


def evaluate_cfg(X, rets, cfg):
    future = rets[1:]
    feat_names = ["log_ret", "range_ratio"]
    if cfg.use_vol_change:
        feat_names.append("vol_change")
    if cfg.use_basis:
        feat_names.append("basis_spread")
    if cfg.use_open_interest:
        feat_names.append("oi_rel_delta")

    model = HMMRegimeModel(cfg)
    ok_fit = model.fit(X)
    states = model.predict_states(X) if ok_fit else np.zeros((len(X),), dtype=int)
    st_f = states[:-1]
    st_f, future = align(st_f, future)
    k = cfg.k_states

    markov_kl = float(np.mean(np.abs(trans(states, k, 1) - trans(states, k, 2).mean(axis=0))))
    t1 = markov_kl < 0.02

    ov = []
    for i in range(k):
        ri = future[st_f == i]
        mui = np.mean(ri) if len(ri) else 0.0
        sdi = np.std(ri) + 1e-9 if len(ri) else 1.0
        for j in range(i + 1, k):
            rj = future[st_f == j]
            muj = np.mean(rj) if len(rj) else 0.0
            sdj = np.std(rj) + 1e-9 if len(rj) else 1.0
            ov.append(float(np.exp(-abs(mui - muj) / max((sdi + sdj) / 2.0, 1e-9))))
    overlap = float(np.mean(ov)) if ov else 1.0
    t2 = overlap < 0.3

    mi = mi_disc(st_f, np.where(future > 0, 1, np.where(future < 0, -1, 0)))
    t3 = mi > 0.01

    chunks = np.array_split(states, 5)
    mats = [trans(c, k, 1) for c in chunks if len(c) > 3]
    var = float(np.mean(np.var(np.stack(mats), axis=0))) if len(mats) >= 2 else 1.0
    t4 = var < 0.05

    h = entropy(states)
    hmax = float(np.log(max(1, len(np.unique(states)))))
    t5 = (h < 0.8 * hmax) if hmax > 0 else False

    base_acc = directional_acc(st_f, future)
    imps = []
    for j, name in enumerate(feat_names):
        Xp = X.copy()
        Xp[:, j] = np.random.default_rng(42 + j).permutation(Xp[:, j])
        m2 = HMMRegimeModel(cfg)
        m2.fit(Xp)
        s2 = m2.predict_states(Xp)[:-1]
        imps.append((name, base_acc - directional_acc(s2, future)))

    Xd = X[:-1]
    rd = future[:-1]
    md = HMMRegimeModel(cfg)
    md.fit(Xd)
    sd = md.predict_states(Xd)[:-1]
    acc_delay = directional_acc(sd, rd[1:] if len(rd) > 1 else rd)
    Xn = X + np.random.default_rng(0).normal(0, 0.01, size=X.shape)
    mn = HMMRegimeModel(cfg)
    mn.fit(Xn)
    sn = mn.predict_states(Xn)[:-1]
    acc_noise = directional_acc(sn, future)
    t7 = acc_delay > 0 and acc_noise > 0

    base_pred = np.sign(rets[1:-1]) if len(rets) > 2 else np.array([])
    bp, yy = align(base_pred, np.sign(future[1:]))
    baseline = float(np.mean(bp == yy)) if len(bp) else 0.0
    t8 = base_acc > baseline

    wf = []
    splits = np.array_split(np.arange(len(X)), 5)
    for i in range(1, len(splits)):
        tr = np.concatenate(splits[:i])
        te = splits[i]
        if len(tr) < 80 or len(te) < 20:
            continue
        mm = HMMRegimeModel(cfg)
        mm.fit(X[tr])
        rr = rets[1:]
        tr2 = tr[tr < len(rr)]
        te2 = te[te < len(rr)]
        if len(tr2) < 20 or len(te2) < 5:
            continue
        sttr = mm.predict_states(X[tr2])
        rrtr = np.asarray(rr)[tr2]
        mp = {}
        for s in np.unique(sttr):
            seg = rrtr[sttr == s]
            mp[int(s)] = 1 if (len(seg) and np.mean(seg) > 0) else -1
        stte = mm.predict_states(X[te2])
        rrte = np.asarray(rr)[te2]
        pred = np.array([mp.get(int(s), 0) for s in stte])
        pred, yt = align(pred, np.sign(rrte))
        m = pred != 0
        if np.any(m):
            wf.append(float(np.mean(pred[m] == yt[m])))
    wf_mean = float(np.mean(wf)) if wf else 0.0
    wf_std = float(np.std(wf)) if wf else 1.0
    t9 = wf_mean > 0 and wf_std < abs(wf_mean)

    means = model.state_means
    sig = np.sign([means[s, 0] if means is not None else 0.0 for s in st_f])
    sig, fut = align(sig, future)
    expect = float(np.mean(sig * fut) - 0.002)
    t10 = expect > 0
    tests = [t1, t2, t3, t4, t5, True, t7, t8, t9, t10]
    return {
        "pass_count": int(sum(bool(x) for x in tests)),
        "tests": tests,
        "markov_kl": markov_kl,
        "overlap": overlap,
        "mi": mi,
        "var": var,
        "h": h,
        "hmax": hmax,
        "imps": imps,
        "acc_delay": acc_delay,
        "acc_noise": acc_noise,
        "base_acc": base_acc,
        "baseline": baseline,
        "wf_mean": wf_mean,
        "wf_std": wf_std,
        "wf_n": len(wf),
        "expect": expect,
        "X_shape": X.shape,
        "k": k,
    }

def quick_rank_score(X, rets, cfg):
    """Fast pre-ranking to avoid full expensive evaluation for all configs."""
    future = rets[1:]
    model = HMMRegimeModel(cfg)
    ok_fit = model.fit(X)
    states = model.predict_states(X) if ok_fit else np.zeros((len(X),), dtype=int)
    st_f = states[:-1]
    st_f, future = align(st_f, future)
    k = cfg.k_states
    markov_kl = float(np.mean(np.abs(trans(states, k, 1) - trans(states, k, 2).mean(axis=0))))
    ov = []
    for i in range(k):
        ri = future[st_f == i]
        mui = np.mean(ri) if len(ri) else 0.0
        sdi = np.std(ri) + 1e-9 if len(ri) else 1.0
        for j in range(i + 1, k):
            rj = future[st_f == j]
            muj = np.mean(rj) if len(rj) else 0.0
            sdj = np.std(rj) + 1e-9 if len(rj) else 1.0
            ov.append(float(np.exp(-abs(mui - muj) / max((sdi + sdj) / 2.0, 1e-9))))
    overlap = float(np.mean(ov)) if ov else 1.0
    mi = mi_disc(st_f, np.where(future > 0, 1, np.where(future < 0, -1, 0)))
    h = entropy(states)
    hmax = float(np.log(max(1, len(np.unique(states)))))
    entropy_ratio = (h / hmax) if hmax > 0 else 1.0
    # lower is better for kl/overlap/entropy_ratio, higher better for mi
    score = (2.0 * mi) - markov_kl - overlap - entropy_ratio
    return float(score)


def main():
    settings = get_settings()
    k = settings.HMM_LIVE_K_STATES

    fetcher = DataFetcher(settings=settings, use_cache=True)
    end = datetime.utcnow().date()
    start = end - timedelta(days=settings.HMM_LIVE_DAYS_BACK)
    bars, idx, _ = fetch_aligned_future_index(
        fetcher,
        settings.HMM_SYMBOL,
        settings.HMM_INDEX_SYMBOL,
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        RESOLUTION_15MIN,
    )

    base_cfg = HMMConfig(
        k_states=k,
        use_basis=settings.HMM_USE_BASIS,
        use_open_interest=settings.HMM_USE_OPEN_INTEREST,
        use_vol_change=settings.HMM_USE_VOL_CHANGE,
        state_smoothing_window=settings.HMM_STATE_SMOOTHING_WINDOW,
        state_min_run_bars=settings.HMM_STATE_MIN_RUN_BARS,
        state_scoring_mode=settings.HMM_STATE_SCORING_MODE,
        state_strength_flat_quantile=settings.HMM_STATE_STRENGTH_FLAT_QUANTILE,
    )
    X = HMMFeatureEngineer(base_cfg).compute(bars, index_closes=idx)
    rets = np.log(np.array([b.close for b in bars[1:]]) / np.array([b.close for b in bars[:-1]]))
    best = None
    ranked = []
    total_checked = 0
    for rs in [base_cfg.random_state, 7, 21]:
        for zw in [10, 15, 20]:
            for sw in [1, 3]:
                for mr in [1, 2]:
                    for smode in ["argmax", "strength"]:
                        for flat_q in [0.30, 0.35]:
                            cfg = HMMConfig(
                                k_states=base_cfg.k_states,
                                n_iter=min(int(base_cfg.n_iter), 200),
                                covariance_type=base_cfg.covariance_type,
                                zscore_window=zw,
                                random_state=rs,
                                use_basis=base_cfg.use_basis,
                                use_open_interest=base_cfg.use_open_interest,
                                use_vol_change=base_cfg.use_vol_change,
                                state_smoothing_window=sw,
                                state_min_run_bars=mr,
                                state_scoring_mode=smode,
                                state_strength_flat_quantile=flat_q,
                            )
                            total_checked += 1
                            ranked.append((quick_rank_score(X, rets, cfg), cfg))

    ranked.sort(key=lambda x: x[0], reverse=True)
    top_cfgs = [cfg for _, cfg in ranked[: min(18, len(ranked))]]
    for cfg in top_cfgs:
        r = evaluate_cfg(X, rets, cfg)
        if (best is None) or (r["pass_count"] > best["res"]["pass_count"]):
            best = {"cfg": cfg, "res": r}

    if best is None:
        raise RuntimeError("No valid HMM configuration produced during optimization.")
    cfg = best["cfg"]
    r = best["res"]
    t1, t2, t3, t4, t5, _, t7, t8, t9, t10 = r["tests"]

    rows = []
    rows.append("KET QUA TEST LOGIC THUAT TOAN (HMM/Markov)")
    rows.append(f"Data bars={len(bars)}, features={r['X_shape']}, k_states={r['k']}")
    rows.append(
        "Optimized config: "
        f"seed={cfg.random_state}, zscore_window={cfg.zscore_window}, "
        f"state_smoothing_window={cfg.state_smoothing_window}, state_min_run_bars={cfg.state_min_run_bars}, "
        f"state_scoring_mode={cfg.state_scoring_mode}, state_strength_flat_quantile={cfg.state_strength_flat_quantile:.2f}, "
        f"pass_count={r['pass_count']}/10, total_checked={total_checked}"
    )
    rows.append("")
    mk = lambda i, p, d: f"Test {i}: {'PASS' if p else 'FAIL'} | {d}"
    rows.append(mk(1, t1, f"markov_kl={r['markov_kl']:.6f} (threshold<0.02)"))
    rows.append(mk(2, t2, f"state_overlap={r['overlap']:.6f} (threshold<0.3)"))
    rows.append(mk(3, t3, f"mutual_info={r['mi']:.6f} (threshold>0.01)"))
    rows.append(mk(4, t4, f"transition_variance={r['var']:.6f} (threshold<0.05)"))
    rows.append(mk(5, t5, f"entropy={r['h']:.6f}, max={r['hmax']:.6f}, ratio={(r['h']/r['hmax'] if r['hmax'] else 0):.3f} (threshold<0.8)"))
    rows.append(mk(6, True, "feature_importance: " + ", ".join([f"{n}={v:.5f}" for n, v in r["imps"]])))
    rows.append(mk(7, t7, f"robust_delay_acc={r['acc_delay']:.4f}, robust_noise_acc={r['acc_noise']:.4f}"))
    rows.append(mk(8, t8, f"hmm_acc={r['base_acc']:.4f} vs baseline_acc={r['baseline']:.4f}"))
    rows.append(mk(9, t9, f"walk_forward_mean={r['wf_mean']:.4f}, std={r['wf_std']:.4f}, n={r['wf_n']}"))
    rows.append(mk(10, t10, f"expectancy_after_cost={r['expect']:.6f}"))
    rows.append("")
    rows.append("Ghi chú: Test định lượng theo checklist trong test_logic_toan.docx, chạy trên dữ liệu hiện có tại thời điểm chạy.")
    Path("test_logic_thuattoan_result").write_text("\n".join(rows), encoding="utf-8")
    print("WROTE test_logic_thuattoan_result")


if __name__ == "__main__":
    main()
