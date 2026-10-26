"""Run an entry or rebuild the leaderboard.
  python -m decay_bench.run --task m1 --protocol A --model ridge_zsess
  python -m decay_bench.run --task m1 --protocol A --model spint --ckpt path/to/epoch_049.ckpt --tag spint_e50
  python -m decay_bench.run --leaderboard
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_task
from .models import MODELS
from .protocols import protocol_a, protocol_b, summarize

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def run(task, protocol, model, tag=None, **kw):
    recs = load_task(task)
    fn = {"A": protocol_a, "B": protocol_b}[protocol]
    rows = fn(lambda: MODELS[model](**kw), recs)
    out = dict(task=task, protocol=protocol, model=tag or model, source="local", rows=rows, summary=summarize(rows))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{task}_{protocol}_{tag or model}.json").write_text(json.dumps(out, indent=1))
    return out


def official_entries():
    D = pd.read_csv(ROOT / "official" / "falcon_authors_pairs.csv")
    D["gap"] = D.gap.abs()
    for (task, model), g in D.groupby(["task", "model"]):
        rows = [dict(train=str(r.train), test=str(r.test), gap=int(r.gap), r2=float(r.r2)) for r in g.itertuples()]
        name = {"ndt2": "NDT2 (official)", "rnn": "GRU (official)"}[model]
        yield dict(task=task, protocol="B", model=name, source="FALCON authors, private test", rows=rows,
                   summary=summarize(rows))


def leaderboard():
    entries = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))] + list(official_entries())
    for e in entries:
        e["summary"] = summarize(e["rows"])             # always from the stored rows
    lines = ["# Decoder decay leaderboard", "",
             "R2 by days since training (protocol A: since the last held-in day; protocol B: between training and "
             "test day), and the decay slope of R2 against log2(1 + days) with a 95% bootstrap CI. "
             "Less negative slope = slower decay.", ""]
    for task in sorted({e["task"] for e in entries}):
        for prot in ("A", "B"):
            es = [e for e in entries if e["task"] == task and e["protocol"] == prot]
            if not es:
                continue
            cols = sorted({k for e in es for k in e["summary"] if k[0].isdigit()}, key=lambda c: int(c.split("-")[0]))
            lines += [f"## {task.upper()} · protocol {prot}", "",
                      "| model | " + " | ".join(cols) + " | slope [95% CI] | collapsed | pairs | source |",
                      "|---|" + "---|" * (len(cols) + 4)]
            for e in sorted(es, key=lambda e: -(e["summary"]["slope"] if e["summary"]["slope"] is not None else -9)):
                s = e["summary"]
                cells = [f"{s[c]:.2f}" if c in s else "" for c in cols]
                sl = (f"{s['slope']:+.3f} [{s['slope_ci'][0]:+.3f}, {s['slope_ci'][1]:+.3f}]"
                      if s["slope"] is not None else "n/a (collapsed)")
                lines.append(f"| {e['model']} | " + " | ".join(cells) +
                             f" | {sl} | {s['collapsed']:.0%} | {s['n_pairs']} | {e['source']} |")
            lines.append("")
    (ROOT / "LEADERBOARD.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task"); ap.add_argument("--protocol", choices=["A", "B"]); ap.add_argument("--model")
    ap.add_argument("--ckpt"); ap.add_argument("--tag"); ap.add_argument("--leaderboard", action="store_true")
    a = ap.parse_args()
    if a.leaderboard:
        leaderboard()
    else:
        s = run(a.task, a.protocol, a.model, a.tag, ckpt=a.ckpt)["summary"]
        print(json.dumps(s, indent=1))
