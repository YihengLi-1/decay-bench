import numpy as np
from datetime import date

from decay_bench.data import Recording, parse
from decay_bench.protocols import protocol_a, protocol_b, slope, summarize


def test_parse():
    assert parse("sub-MonkeyL-held-out-calib_ses-20121004_behavior+ecephys") == ("held_out_calib", date(2012, 10, 4))
    assert parse("sub-HumanPitt-held-in-calib_ses-19250108T110520") == ("held_in_calib", date(1925, 1, 8))
    assert parse("sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys")[1] == date(2020, 10, 19)


def test_slope_recovers_known_decay():
    g = np.array([1, 3, 7, 15, 31])
    assert abs(slope(g, 0.8 - 0.1 * np.log2(1 + g)) + 0.1) < 1e-9
    assert abs(slope(g, [-50.0] * 5)) < 1e-9                # clipping at -1 flattens collapsed decoders


def _rec(name, split, day, seed, gain=1.0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(400, 2))
    x = rng.poisson(np.exp(0.5 + 0.4 * np.c_[y, -y, y[:, :1]]) * gain).astype(float)
    ts = np.zeros(400, bool); ts[::50] = True
    return Recording(name, split, day, x, y, np.ones(400, bool), ts)


class Oracle:
    """Predicts the truth: every protocol should give R2 = 1."""
    def fit(self, recs): return self
    def predict(self, rec, h): return rec.y[h:]


def test_protocols_with_oracle():
    recs = [_rec("a", "held_in_calib", date(2020, 1, 1), 0), _rec("b", "held_in_calib", date(2020, 1, 3), 1),
            _rec("c", "held_out_calib", date(2020, 1, 10), 2)]
    a = protocol_a(Oracle, recs)
    assert [r["gap"] for r in a] == [7] and a[0]["r2"] == 1.0
    b = protocol_b(Oracle, recs)
    assert sorted((r["train"], r["gap"]) for r in b) == [("2020-01-01", 2), ("2020-01-01", 9), ("2020-01-03", 2), ("2020-01-03", 7)]
    s = summarize(b)
    assert abs(s["slope"]) < 1e-12 and s["n_pairs"] == 4


def test_split_index_is_trial_aligned():
    r = _rec("a", "held_in_calib", date(2020, 1, 1), 0)
    assert r.split_index() == 200 and r.trial_start[r.split_index()]


def test_collapsed_decoders_get_no_slope():
    rows = [dict(train="t", test=str(i), gap=g, r2=-3.0) for i, g in enumerate([2, 5, 9])]
    s = summarize(rows)
    assert s["slope"] is None and s["collapsed"] == 1.0


def test_gru_learns_synthetic_mapping():
    from decay_bench.models import GRUDecoder
    from decay_bench.protocols import score
    recs = [_rec("a", "held_in_calib", date(2020, 1, 1), 0), _rec("b", "held_in_calib", date(2020, 1, 1), 1)]
    dec = GRUDecoder("train", hidden=32, steps=300).fit(recs)
    assert score(dec, _rec("c", "held_in_calib", date(2020, 1, 1), 2)) > 0.2


def test_riemann_mean_and_tangent():
    from decay_bench.eeg import riemann_mean, tangent
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)); M = A @ A.T + 4 * np.eye(4)
    assert np.allclose(riemann_mean(np.stack([M, M, M])), M)
    t = tangent(np.stack([M, M])[:, None], M[None])          # a point at the reference maps to zero
    assert np.allclose(t, 0, atol=1e-8)
    B = rng.normal(size=(4, 4)); N = B @ B.T + 4 * np.eye(4)
    G = riemann_mean(np.stack([M, N]))                        # the geodesic midpoint is equidistant
    d = lambda P, Q: np.linalg.norm(tangent(P[None, None], Q[None]))
    assert abs(d(M, G) - d(N, G)) < 1e-6
