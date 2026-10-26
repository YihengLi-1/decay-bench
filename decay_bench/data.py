"""FALCON recordings as plain arrays, cached, with recording day and split.

Only calibration files are used: held-in calibration (training days) and held-out calibration (later days,
labeled, the only held-out data that is public). Minival files are 2 trials long and are skipped.
"""
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("FALCON_DATA", "~/falcon/data")).expanduser()
CACHE = Path(os.environ.get("DECAY_BENCH_CACHE", "~/falcon/cache")).expanduser()
TASKS = {"h1": "000954", "m1": "000941", "m2": "000953"}


@dataclass
class Recording:
    name: str
    split: str          # "held_in_calib" | "held_out_calib"
    day: date
    x: np.ndarray       # binned spike counts, 20 ms bins, (T, channels)
    y: np.ndarray       # decoding target, (T, dims)
    mask: np.ndarray    # bins the FALCON evaluator scores, minus NaN targets, (T,)
    trial_start: np.ndarray  # bool, (T,)

    def split_index(self) -> int:
        r"""First trial start at or after the middle: bins before it are unlabeled context, bins from it on
        are test. Trial-aligned so trial-based adaptation methods get whole trials."""
        starts = np.flatnonzero(self.trial_start)
        later = starts[starts >= len(self.x) // 2]
        return int(later[0]) if len(later) else len(self.x) // 2


def parse(fname: str) -> tuple[str, date]:
    m = re.search(r"(held-(?:in|out))-(calib|minival)_ses-(\d{4})-?(\d{2})-?(\d{2})", fname)
    if m is None:
        raise ValueError(f"unexpected FALCON file name {fname}")
    return f"{m.group(1).replace('-', '_')}_{m.group(2)}", date(*map(int, m.group(3, 4, 5)))


def load_task(task: str) -> list[Recording]:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    root = DATA / TASKS[task]
    out = []
    for f in sorted(root.glob("*/*.nwb")):
        split, day = parse(f.name)
        if split.endswith("minival"):
            continue
        cache = CACHE / task / (f.stem + ".npz")
        if cache.exists():
            z = np.load(cache)
            x, y, mask, ts = z["x"], z["y"], z["mask"], z["trial_start"]
        else:
            x, y, tc, m = load_nwb(str(f), dataset=getattr(FalconTask, task))
            y = np.asarray(y, np.float64)
            mask = np.asarray(m, bool) & ~np.isnan(y).any(1)
            ts = np.asarray(tc, bool)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, x=x, y=y, mask=mask, trial_start=ts)
        out.append(Recording(f.stem, split, day, np.asarray(x), y, mask, ts))
    return out
