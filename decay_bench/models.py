"""Decoders. Each has fit(train_recordings) and predict(recording, h) -> predictions for bins h: (the test
part); bins before h are unlabeled context. Every model family should come with a normalization control, so
the leaderboard shows how much of its stability a per-session z-score alone would give.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import lfilter
from sklearn.linear_model import Ridge

TAU_BINS, HIST = 240 / 20, 4           # causal exponential filter (H1 lab, 240 ms) and 80 ms of lags


def smooth(x):
    k = np.exp(-np.arange(TAU_BINS) / TAU_BINS)
    return lfilter(k / k.sum(), [1.0], x.astype(np.float64), axis=0)


def lag(z):
    return np.concatenate([np.roll(z, i, 0) for i in range(HIST)], 1)


class RidgeDecoder:
    r"""Linear decoder on smoothed, lagged, z-scored spike counts.
    norm="train": z-score with training statistics (no adaptation).
    norm="session": z-score each test recording with its own unlabeled context (floor: a quarter of the
    training SD, for channels that are nearly silent in the context)."""

    def __init__(self, norm="train", alpha=100.0):
        self.norm, self.alpha = norm, alpha

    def fit(self, recs):
        X = np.concatenate([smooth(r.x) for r in recs])
        Y = np.concatenate([r.y for r in recs])
        M = np.concatenate([r.mask for r in recs])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.model = Ridge(alpha=self.alpha).fit(lag((X - self.mu) / self.sd)[M], Y[M])
        return self

    def predict(self, rec, h):
        x = smooth(rec.x)
        if self.norm == "session":
            mu, sd = x[:h].mean(0), np.maximum(x[:h].std(0), 0.25 * self.sd)
        else:
            mu, sd = self.mu, self.sd
        return self.model.predict(lag((x - mu) / sd))[h:]


class SpintDecoder:
    r"""A trained SPINT checkpoint (Le et al., NeurIPS 2025; github.com/shlizee/SPINT), adapted to each test
    recording without gradients from its context trials. Trained on all held-in data, so protocol A only.
    The unit-identity embedding depends only on the context, so it is computed once per recording."""

    MAXLEN = 1024

    def __init__(self, ckpt, device=None):
        import torch
        from src.models.falcon_module import FalconLitModule    # SPINT repo on PYTHONPATH
        self.torch = torch
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        lit = FalconLitModule.load_from_checkpoint(ckpt, map_location="cpu")
        self.net = lit.net.to(self.dev).eval()
        self.scale = lit.hparams.behavior_scaling_factor

    def fit(self, recs):
        return self

    def _trials(self, x, ts, m):
        xm, tm = x[m], ts[m]
        s = np.flatnonzero(tm)
        out = []
        for i, a in enumerate(s):
            tr = xm[a:(s[i + 1] if i + 1 < len(s) else len(xm))].astype(np.float32)
            f = interp1d(np.linspace(0, 1, len(tr)), tr, axis=0, kind="cubic", fill_value="extrapolate")
            out.append(f(np.linspace(0, 1, self.MAXLEN)))
        return np.array(out, np.float32)

    def predict(self, rec, h):
        torch, net, W = self.torch, self.net, self.net.window_size
        ctx = self._trials(rec.x[:h], rec.trial_start[:h], rec.mask[:h])
        idx = np.array([t for t in range(max(h, W - 1), len(rec.x)) if rec.mask[t]])
        pred = np.zeros((len(rec.x) - h, rec.y.shape[1]))
        with torch.no_grad():
            c = torch.tensor(ctx)[None].to(self.dev).permute(0, 1, 3, 2)
            uid = net.fc_id_out(net.fc_id_in(c).mean(1))
            rep = net.fc_in(net.rep.to(self.dev))
            for b in range(0, len(idx), 256):
                win = np.stack([rec.x[t - W + 1:t + 1] for t in idx[b:b + 256]]).astype(np.float32)
                src = net.fc_in(torch.tensor(win).to(self.dev).permute(0, 2, 1) + uid)
                out, _ = net.transformer(rep.expand(len(win), -1, -1), src)
                pred[idx[b:b + 256] - h] = net.fc_out(out).permute(0, 2, 1)[:, -1].cpu().numpy() / self.scale
        return pred


class GRUDecoder(RidgeDecoder):
    r"""One-layer GRU on the same smoothed, z-scored inputs as the ridge decoder (same two normalization
    options), trained on random 1 s chunks of the training recordings and run over a whole recording at test
    time. Small on purpose (CPU, minutes): it is the nonlinear counterpart of the ridge baseline, not a tuned
    model."""

    def __init__(self, norm="train", hidden=128, steps=3000, chunk=50, seed=0):
        super().__init__(norm)
        self.hidden, self.steps, self.chunk, self.seed = hidden, steps, chunk, seed

    def fit(self, recs):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = [smooth(r.x) for r in recs]
        allx = np.concatenate(X)
        self.mu, self.sd = allx.mean(0), allx.std(0) + 1e-6
        Z = [torch.tensor((x - self.mu) / self.sd, dtype=torch.float32) for x in X]
        Y = [torch.tensor(np.nan_to_num(r.y), dtype=torch.float32) for r in recs]
        M = [torch.tensor(r.mask) for r in recs]
        self.ymu = torch.tensor(np.nanmean(np.concatenate([r.y[r.mask] for r in recs]), 0), dtype=torch.float32)
        self.ysd = torch.tensor(np.nanstd(np.concatenate([r.y[r.mask] for r in recs]), 0) + 1e-6, dtype=torch.float32)
        self.rnn = torch.nn.GRU(allx.shape[1], self.hidden, batch_first=True)
        self.head = torch.nn.Linear(self.hidden, recs[0].y.shape[1])
        opt = torch.optim.Adam(list(self.rnn.parameters()) + list(self.head.parameters()), lr=1e-3)
        lens = np.array([len(z) for z in Z])
        for _ in range(self.steps):
            b = rng.choice(len(Z), 32, p=lens / lens.sum())
            st = [rng.integers(0, lens[i] - self.chunk) for i in b]
            zb = torch.stack([Z[i][s:s + self.chunk] for i, s in zip(b, st)])
            yb = torch.stack([Y[i][s:s + self.chunk] for i, s in zip(b, st)])
            mb = torch.stack([M[i][s:s + self.chunk] for i, s in zip(b, st)])
            out = self.head(self.rnn(zb)[0])
            loss = (((out - (yb - self.ymu) / self.ysd) ** 2).mean(-1) * mb).sum() / mb.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict(self, rec, h):
        import torch
        x = smooth(rec.x)
        if self.norm == "session":
            mu, sd = x[:h].mean(0), np.maximum(x[:h].std(0), 0.25 * self.sd)
        else:
            mu, sd = self.mu, self.sd
        with torch.no_grad():
            z = torch.tensor((x - mu) / sd, dtype=torch.float32)[None]
            out = self.head(self.rnn(z)[0])[0] * self.ysd + self.ymu
        return out.numpy()[h:]


MODELS = {
    "ridge_raw": lambda **kw: RidgeDecoder("train"),
    "ridge_zsess": lambda **kw: RidgeDecoder("session"),
    "spint": lambda ckpt, **kw: SpintDecoder(ckpt),
    "gru_raw": lambda **kw: GRUDecoder("train"),
    "gru_zsess": lambda **kw: GRUDecoder("session"),
}
