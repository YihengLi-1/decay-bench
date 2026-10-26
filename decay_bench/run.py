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


def run_eeg(paradigm, model):
    from .eeg import EEG_MODELS, load_kaya, protocol_b_eeg
    rows = protocol_b_eeg(EEG_MODELS[model], load_kaya(paradigm))
    task = f"kaya-{paradigm.lower()}"
    out = dict(task=task, protocol="B", model=model, source="local", metric="score", metric_name="balanced accuracy",
               rows=rows, summary=summarize(rows, key="score"))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{task}_B_{model}.json").write_text(json.dumps(out, indent=1))
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
        e["summary"] = summarize(e["rows"], key=e.get("metric", "r2"))   # always from the stored rows
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
            metric = es[0].get("metric_name", "R2")
            lines += [f"## {task.upper()} · protocol {prot} · {metric}", "",
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


def figure():
    r"""One panel per task x protocol, one fixed colour per entry. Protocol A plots each held-out day; protocol B
    averages pairs within gap bins (1-3, 4-7, 8-14, 15-30, 31+ days), plotted at the bin's mean gap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    entries = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))] + list(official_entries())
    names = sorted({e["model"] for e in entries})
    colour = {n: plt.cm.tab10(i % 10) for i, n in enumerate(names)}
    panels = sorted({(e["task"], e["protocol"]) for e in entries})
    fig, axs = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.3), squeeze=False)
    bins = [(1, 3), (4, 7), (8, 14), (15, 30), (31, 400)]
    for ax, (task, prot) in zip(axs[0], panels):
        for e in [e for e in entries if e["task"] == task and e["protocol"] == prot]:
            rows = [r for r in e["rows"] if r["gap"] > 0]
            groups = ([[r for r in rows if r["gap"] == g] for g in sorted({r["gap"] for r in rows})] if prot == "A"
                      else [[r for r in rows if lo <= r["gap"] <= hi] for lo, hi in bins])
            groups = [g for g in groups if g]
            x = [np.mean([r["gap"] for r in g]) for g in groups]
            k = e.get("metric", "r2")
            y = [np.mean(np.clip([r[k] for r in g], -1, None)) for g in groups]
            ax.plot(x, y, "o", ls="--" if "official" in e["model"] else "-", ms=3.5, lw=1.4, color=colour[e["model"]])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{task.upper()} · protocol {prot}", fontsize=9)
        ax.set_xlabel("days since training", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_ylim(*((0, 1) if task.startswith("kaya") else (-1.05, 0.9)))
        if task.startswith("kaya"):
            ax.set_title(f"{task.upper()} · protocol {prot} (balanced acc.)", fontsize=9)
    axs[0][0].set_ylabel("R² (clipped at -1)", fontsize=8)
    handles = [plt.Line2D([], [], color=colour[n], ls="--" if "official" in n else "-", marker="o", ms=3.5) for n in names]
    fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "decay_curves.png", dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task"); ap.add_argument("--protocol", choices=["A", "B"]); ap.add_argument("--model")
    ap.add_argument("--ckpt"); ap.add_argument("--tag"); ap.add_argument("--leaderboard", action="store_true")
    ap.add_argument("--eeg", help="Kaya 2018 paradigm: CLA or HaLT")
    a = ap.parse_args()
    if a.leaderboard:
        leaderboard()
        figure()
    elif a.eeg:
        print(json.dumps(run_eeg(a.eeg, a.model)["summary"], indent=1))
    else:
        s = run(a.task, a.protocol, a.model, a.tag, ckpt=a.ckpt)["summary"]
        print(json.dumps(s, indent=1))
