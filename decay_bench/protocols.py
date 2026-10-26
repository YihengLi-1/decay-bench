"""Evaluation protocols and summary statistics.

Protocol A (later sessions): train on all held-in calibration recordings, score every held-out calibration
recording. Gap = days since the last training day.
Protocol B (cross-session matrix): for each held-in day, train on that day only and score every other
calibration recording. Gap = |days| between training and test day. More pairs, so the decay slope can be
estimated per training day and bootstrapped.

In both, a test recording is split at a trial boundary near its middle: the first part is unlabeled context
(a decoder may use its neural data, never its labels), the second part is scored.
Score: variance-weighted R2 over the bins the FALCON evaluator scores.
"""
from collections import defaultdict

import numpy as np
from sklearn.metrics import r2_score


def r2(y, p, m):
    return float(r2_score(y[m], p[m], multioutput="variance_weighted"))


def score(decoder, rec):
    h = rec.split_index()
    pred = decoder.predict(rec, h)                      # predictions for bins h: (T - h, dims)
    m = rec.mask[h:]
    return r2(rec.y[h:], pred, m)


def protocol_a(make_decoder, recs):
    train = [r for r in recs if r.split == "held_in_calib"]
    last = max(r.day for r in train)
    dec = make_decoder()
    dec.fit(train)
    return [dict(train="all_held_in", test=r.name, gap=(r.day - last).days, r2=score(dec, r))
            for r in recs if r.split == "held_out_calib"]


def protocol_b(make_decoder, recs):
    rows = []
    for d in sorted({r.day for r in recs if r.split == "held_in_calib"}):
        train = [r for r in recs if r.split == "held_in_calib" and r.day == d]
        dec = make_decoder()
        dec.fit(train)
        rows += [dict(train=str(d), test=r.name, gap=abs((r.day - d).days), r2=score(dec, r))
                 for r in recs if r.day != d]
    return rows


def slope(gaps, r2s):
    r"""Least-squares slope of R2 against log2(1 + gap). R2 is clipped at -1 so a few collapsed decoders do
    not dominate the fit."""
    return float(np.polyfit(np.log2(1 + np.asarray(gaps, float)), np.clip(r2s, -1, None), 1)[0])


def summarize(rows, n_boot=5000, seed=0, key="r2"):
    r"""Mean R2 per gap bin, the share of collapsed pairs (R2 < -1), and the decay slope with a bootstrap CI
    (None when more than half the pairs collapsed). For protocol B the slope is fitted per
    training day (off-diagonal pairs) and the bootstrap resamples training days; for protocol A there is one
    training set, so the bootstrap resamples test recordings."""
    bins = [(1, 3), (4, 7), (8, 14), (15, 30), (31, 400)]
    out = {f"{lo}-{hi}d": float(np.mean([r[key] for r in rows if lo <= r["gap"] <= hi]))
           for lo, hi in bins if any(lo <= r["gap"] <= hi for r in rows)}
    rng = np.random.default_rng(seed)
    by_train = defaultdict(list)
    for r in rows:
        if r["gap"] > 0:
            by_train[r["train"]].append(r)
    if len(by_train) > 1:
        s = np.array([slope([r["gap"] for r in v], [r[key] for r in v]) for v in by_train.values()])
        boots = [rng.choice(s, len(s)).mean() for _ in range(n_boot)]
        est = s.mean()
    else:
        v = next(iter(by_train.values()))
        g, y = np.array([r["gap"] for r in v]), np.array([r[key] for r in v])
        est = slope(g, y)
        boots = []
        for _ in range(n_boot):
            i = rng.choice(len(g), len(g))
            if len(set(g[i])) > 1:
                boots.append(slope(g[i], y[i]))
    out["collapsed"] = float(np.mean([r[key] < -1 for r in rows]))
    out["n_pairs"] = len(rows)
    if out["collapsed"] > 0.5:                          # clipped fit is meaningless when most pairs collapsed
        out["slope"], out["slope_ci"] = None, None
    else:
        out["slope"] = float(est)
        out["slope_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out
