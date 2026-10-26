"""Multi-day EEG motor imagery: Kaya et al. 2018 (Sci. Data 5:180211, figshare collection 3917698), whose file
names carry the recording date (YYMMDD), so every session has a day.

Paradigms: CLA (left hand, right hand, passive; 3 classes) and HaLT (left/right hand, left/right leg, tongue,
passive; 6 classes). Each marker run (0.85 s at 200 Hz) is one trial. 19 EEG channels (A1, A2 and the sync
channel are dropped).

Protocol B as for FALCON, per subject: train on one session, score every other session of the same subject and
paradigm; the first half of a test session's trials (in recording order) is unlabeled context, the rest is scored
with balanced accuracy.

Decoders (tangent-space logistic regression on an 8-30 Hz plus 4-40 Hz filter bank, the competition Track 2
features):
  riemann_raw       whitened by the training session's mean covariance (no adaptation)
  riemann_recenter  each session whitened by its own mean covariance, the test session's from its unlabeled
                    context (Zanini et al. 2018 re-centering): the EEG counterpart of per-session z-scoring
"""
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

KAYA = Path(os.environ.get("KAYA_DATA", "~/falcon/data/kaya2018")).expanduser()
BANDS = [(8.0, 30.0), (4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 40.0)]
FS, TRIAL = 200, 170


@dataclass
class EEGSession:
    name: str
    subject: str
    day: date
    covs: np.ndarray    # (trials, bands, ch, ch), OAS-shrunk band covariances
    y: np.ndarray       # (trials,)


def _eig_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


def oas_cov(X):
    X = X - X.mean(-1, keepdims=True)
    n_feat, n_samp = X.shape[-2], X.shape[-1]
    S = X @ np.swapaxes(X, -1, -2) / n_samp
    mu = np.trace(S, axis1=-2, axis2=-1) / n_feat
    alpha = np.mean(S ** 2, axis=(-2, -1))
    den = (n_samp + 1.0) * (alpha - mu ** 2 / n_feat)
    s = np.where(den == 0, 1.0, np.minimum((alpha + mu ** 2) / den, 1.0))
    return (1 - s)[..., None, None] * S + (s * mu)[..., None, None] * np.eye(n_feat)


def riemann_mean(Cs, iters=30):
    M = Cs.mean(0)
    for _ in range(iters):
        Ms, Mi = _eig_fn(M, np.sqrt), _eig_fn(M, lambda w: 1 / np.sqrt(w))
        T = _eig_fn(Mi @ Cs @ Mi, np.log).mean(0)
        M = Ms @ _eig_fn(T, np.exp) @ Ms
        if np.linalg.norm(T) < 1e-8:
            break
    return M


def tangent(Cs, M):
    r"""Tangent vectors of Cs at M (per band), upper triangle with off-diagonals scaled by sqrt(2)."""
    out = []
    for b in range(Cs.shape[1]):
        Mi = _eig_fn(M[b], lambda w: 1 / np.sqrt(w))
        L = _eig_fn(Mi @ Cs[:, b] @ Mi, np.log)
        iu = np.triu_indices(L.shape[-1])
        out.append(L[:, iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)))
    return np.concatenate(out, 1)


def load_kaya(paradigm="CLA"):
    import scipy.io as sio
    cache = KAYA / "cache"
    cache.mkdir(exist_ok=True)
    out = []
    for f in sorted(KAYA.glob(f"{paradigm}-Subject*.mat")):
        m = re.search(r"Subject([A-Z])-(\d{2})(\d{2})(\d{2})", f.name)
        subject, day = m.group(1), date(2000 + int(m.group(2)), int(m.group(3)), int(m.group(4)))
        c = cache / (f.stem + ".npz")
        if c.exists():
            z = np.load(c)
            covs, y = z["covs"], z["y"]
        else:
            o = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["o"]
            if int(o.sampFreq) != FS:
                continue                                  # HFREQ files (1000 Hz) are skipped
            names = list(o.chnames)
            keep = [i for i, n in enumerate(names) if n not in ("A1", "A2")]
            data = np.asarray(o.data, np.float64)[:, keep].T
            mk = np.asarray(o.marker)
            on = np.flatnonzero((mk[1:] != mk[:-1]) & (mk[1:] > 0) & (mk[1:] < 90)) + 1
            on = on[on + TRIAL <= data.shape[1]]
            y = mk[on].astype(int)
            covs = []
            for lo, hi in BANDS:
                xf = sosfiltfilt(butter(4, (lo, hi), btype="bandpass", fs=FS, output="sos"), data, axis=-1)
                covs.append(oas_cov(np.stack([xf[:, a:a + TRIAL] for a in on])))
            covs = np.stack(covs, 1)
            np.savez_compressed(c, covs=covs, y=y)
        out.append(EEGSession(f.stem, subject, day, covs, y))
    return out


class RiemannDecoder:
    def __init__(self, recenter):
        self.recenter = recenter

    def fit(self, s):
        self.M = np.stack([riemann_mean(s.covs[:, b]) for b in range(s.covs.shape[1])])
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(tangent(s.covs, self.M), s.y)
        return self

    def predict(self, covs_ctx, covs_test):
        M = (np.stack([riemann_mean(covs_ctx[:, b]) for b in range(covs_ctx.shape[1])])
             if self.recenter else self.M)
        return self.clf.predict(tangent(covs_test, M))


EEG_MODELS = {"riemann_raw": lambda: RiemannDecoder(False), "riemann_recenter": lambda: RiemannDecoder(True)}


def protocol_b_eeg(make_decoder, sessions):
    by_subj = defaultdict(list)
    for s in sessions:
        by_subj[s.subject].append(s)
    rows = []
    for subj, ss in sorted(by_subj.items()):
        if len(ss) < 2:
            continue
        for tr in ss:
            dec = make_decoder().fit(tr)
            for te in ss:
                if te is tr:
                    continue
                h = len(te.y) // 2
                pred = dec.predict(te.covs[:h], te.covs[h:])
                rows.append(dict(train=f"{subj}:{tr.day}", test=te.name, gap=abs((te.day - tr.day).days),
                                 score=float(balanced_accuracy_score(te.y[h:], pred))))
    return rows
