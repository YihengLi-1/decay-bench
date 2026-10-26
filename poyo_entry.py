"""POYO entry for decay_bench, protocol A (runs in a torch_brain environment, not the decay_bench one).

POYO (Azabou et al., NeurIPS 2023; torch_brain implementation) trained from scratch on FALCON held-in calibration
data, with unit tokens shared across days (each Utah-array channel keeps its identity, as for the ridge and NDT2
baselines). For a held-out day the model has no session embedding, so it uses the last held-in day's (zero-shot).
Training length is fixed in advance; held-out data is never used for selection.
Scoring matches decay_bench.protocols: each held-out recording is split at the first trial start at or after its
middle bin, and variance-weighted R2 is computed on FALCON eval bins from that point on.

  python poyo_entry.py --task m1 --epochs 150 --tag poyo_zeroshot
writes results/{task}_A_{tag}.json in the decay_bench format.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "falcon/torch_brain/examples/poyo"))
from datasets.wrapper import PoyoDatasetWrapper, PoyoReadoutConfig  # noqa: E402
from utils import create_optim, move_to_device, seed_everything, weighted_mse_loss_fn  # noqa: E402

from torch_brain.batching import collate  # noqa: E402
from torch_brain.data import Interval  # noqa: E402
from torch_brain.datasets import KarpowiczPandarinathFalcon2024  # noqa: E402
from torch_brain.models import POYO  # noqa: E402
from torch_brain.samplers import RandomFixedWindowSampler  # noqa: E402

TARGET = {"m1": ("emg", "data"), "h1": ("kinematics", "vel")}


class FalconPoyo(KarpowiczPandarinathFalcon2024):
    def __init__(self, task, root, mean, std, **kw):
        super().__init__(task=task, root=root, **kw)
        self.spiking_dataset_mixin_uniquify_unit_ids = False      # same channel = same unit across days
        self._readout = (task, mean, std)

    def get_recording_hook(self, data):
        super().get_recording_hook(data)
        task, mean, std = self._readout
        field, value = TARGET[task]
        data.readout_config = PoyoReadoutConfig(
            timestamp_key=f"{field}.timestamps", value_key=f"{field}.{value}",
            normalize_mean=mean, normalize_std=std, eval_interval="falcon_eval_intervals")


def split_time(rec, field):
    r"""Time of the first trial start at or after the middle bin (decay_bench's split_index)."""
    ts = getattr(rec, field).timestamps[:]
    idx = np.searchsorted(ts, rec.trials.start[:])
    later = idx[idx >= len(ts) // 2]
    h = int(later[0]) if len(later) else len(ts) // 2
    return ts, h


@torch.no_grad()
def predict_recording(model, ds_raw, rid, session_for_tokens, field, value, std, mean, step=0.5):
    rec = ds_raw.get_recording(rid)
    ts, h = split_time(rec, field)
    L = model.sequence_length
    t0, t1 = float(rec.domain.start[0]), float(rec.domain.end[-1])
    preds, counts = np.zeros((len(ts), getattr(rec, field).__getattribute__(value).shape[1])), np.zeros(len(ts))
    starts = np.arange(t0, max(t0, t1 - L) + 1e-9, step)
    for s in starts:
        win = rec.slice(s, s + L)
        win.session.id = session_for_tokens
        X = model.tokenize(win)
        wts = getattr(win, field).timestamps[:]
        if len(wts) == 0:
            continue
        Xb = collate([X])
        Xb = move_to_device(Xb, "cpu")
        out = model(**Xb, output_timestamps=torch.tensor(wts[None], dtype=torch.float32))[0].numpy()
        gi = np.searchsorted(ts, wts + s - 1e-9)
        ok = (gi < len(ts)) & (np.abs(ts[np.clip(gi, 0, len(ts) - 1)] - (wts + s)) < 1e-6)
        preds[gi[ok]] += out[ok]
        counts[gi[ok]] += 1
    preds = preds / np.maximum(counts, 1)[:, None] * std + mean
    y = getattr(rec, field).__getattribute__(value)[:]
    ev = rec.falcon_eval_intervals
    m = np.array([((ev.start <= t) & (t < ev.end)).any() for t in ts]) & (counts > 0)
    m[:h] = False
    return float(r2_score(y[m], preds[m], multioutput="variance_weighted"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="m1"); ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--tag", default="poyo_zeroshot"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=str(Path.home() / "falcon/tb_processed"))
    a = ap.parse_args()
    seed_everything(a.seed)
    field, value = TARGET[a.task]
    probe = FalconPoyo(a.task, a.root, 0.0, 1.0)
    train_ids = [r for r in probe.recording_ids if probe.get_recording(r).falcon_split == "held_in_calib"]
    vals = np.concatenate([getattr(probe.get_recording(r), field).__getattribute__(value)[:] for r in train_ids])
    mean, std = float(vals.mean()), float(vals.std())
    ds = FalconPoyo(a.task, a.root, mean, std)

    model = POYO(sequence_length=1.0, latent_step=0.125, num_latents_per_step=16, dim=64, depth=6, dim_head=64,
                 cross_heads=2, self_heads=8, ffn_dropout=0.2, lin_dropout=0.4, atn_dropout=0.2, dim_out=vals.shape[1])
    units = sorted({u for r in ds.recording_ids for u in ds.get_recording(r).units.id[:]})
    model.unit_emb.initialize_vocab(units)
    model.session_emb.initialize_vocab(train_ids)
    wrapped = PoyoDatasetWrapper(ds, model.tokenize)
    sampler = RandomFixedWindowSampler(
        sampling_intervals={r: iv for r, iv in ds.get_sampling_intervals("train").items() if len(iv)},
        window_length=1.0, generator=torch.Generator().manual_seed(a.seed + 1))
    loader = torch.utils.data.DataLoader(wrapped, sampler=sampler, batch_size=128, drop_last=True,
                                         collate_fn=collate, num_workers=0)

    class Cfg:                                                     # what create_optim reads from the hydra config
        epochs = a.epochs
        batch_size = 128

        class optim:
            base_lr, weight_decay, lr_decay_start = 3.125e-5, 1e-4, 0.5
    optim, sched = create_optim(model, len(loader), Cfg)
    t = time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for X, Y in loader:
            mask = Y["output_mask"]
            loss = weighted_mse_loss_fn(model(**X, output_timestamps=Y["timestamps"])[mask], Y["values"][mask],
                                        Y["weights"][mask])
            optim.zero_grad(); loss.backward(); optim.step(); sched.step(); tot += loss.item()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep} loss {tot / len(loader):.4f} ({time.time() - t:.0f}s)", flush=True)

    model.eval()
    last_day = sorted(train_ids)[-1]
    from datetime import datetime
    day = lambda rid: datetime.strptime(rid.split("_")[2], "%Y%m%d").date()
    rows = []
    for rid in ds.recording_ids:
        if ds.get_recording(rid).falcon_split != "held_out_calib":
            continue
        r2 = predict_recording(model, ds, rid, last_day, field, value, std, mean)
        rows.append(dict(train="all_held_in", test=rid, gap=(day(rid) - day(last_day)).days, r2=r2))
        print(rows[-1], flush=True)
    out = dict(task=a.task, protocol="A", model=a.tag, source="local", rows=rows,
               note=f"POYO 1.3M from scratch, {a.epochs} epochs, shared unit tokens, last held-in session embedding")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{a.task}_A_{a.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
