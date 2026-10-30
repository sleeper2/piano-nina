"""Grade the train_data scores that have an explicit start tempo and compare with the Henle labels.

A score (an entry of train_data/index.json) is graded only when EVERY one of its files has a printed
numeric metronome mark at its very start (model.start_tempo); its files are concatenated in the
order of their "path" keys. Writes results/train_data_predictions.csv (one row per score, skipped ones
with the reason) and results/train_data_metrics.json, and prints the weighted (balanced) accuracy =
mean over the grades of the share of scores predicted exactly.

    python evaluate_train_data.py [--workers 4]
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np

from model import DifficultyModel, encode_score, start_tempo

HERE = os.path.dirname(os.path.abspath(__file__))


def load(job):
    """(key, paths) -> (key, start bpms, event matrix or None, skip reason). Runs in a worker."""
    key, paths = job
    bpms = [start_tempo(p) for p in paths]
    if None in bpms:
        return key, bpms, None, "no explicit start tempo"
    try:
        return key, bpms, encode_score(list(zip(paths, bpms))), ""
    except Exception as exc:  # noqa: BLE001
        return key, bpms, None, f"{type(exc).__name__}: {exc}"


def per_grade_rate(y, p, tol=0):
    """Mean over the labelled grades of P(|pred - label| <= tol | label)."""
    return float(np.mean([np.mean(np.abs(p[y == g] - g) <= tol) for g in np.unique(y)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "train_data"))
    ap.add_argument("--checkpoints", default=os.path.join(HERE, "checkpoints"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    index = json.load(open(os.path.join(args.data, "index.json")))
    jobs = [(key, [os.path.join(args.data, "scores", e["path"][k]) for k in sorted(e["path"], key=int)])
            for key, e in index.items()]
    model = DifficultyModel(args.checkpoints)

    rows = []
    with Pool(args.workers) as pool:
        for i, (key, bpms, x, reason) in enumerate(pool.imap(load, jobs), 1):
            e = index[key]
            row = {"key": key, "composer": e["composer"], "work_name": e["work_name"], "n_files": len(e["path"]),
                   "start_bpms": " ".join("-" if b is None else f"{b:g}" for b in bpms), "label": e["henle"],
                   "pred": "", "average": "", "folds": "", "skipped": reason}
            if x is not None:
                res = model.predict_matrix(x)
                row.update(pred=res["grade"], average=res["average"], folds=" ".join(map(str, res["folds"])))
            rows.append(row)
            if i % 25 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] graded {sum(r['pred'] != '' for r in rows)}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train_data_predictions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    graded = [r for r in rows if r["pred"] != ""]
    y = np.array([r["label"] for r in graded])
    p = np.array([r["pred"] for r in graded])
    grades = range(1, 10)
    metrics = {
        "n_scores": len(rows),
        "n_graded": len(graded),
        "n_no_start_tempo": sum(r["skipped"] == "no explicit start tempo" for r in rows),
        "n_failed": sum(r["skipped"] not in ("", "no explicit start tempo") for r in rows),
        "weighted_accuracy": round(per_grade_rate(y, p), 4),
        "weighted_accuracy_pm1": round(per_grade_rate(y, p, tol=1), 4),
        "accuracy": round(float(np.mean(p == y)), 4),
        "accuracy_pm1": round(float(np.mean(np.abs(p - y) <= 1)), 4),
        "per_grade": {g: {"n": int(np.sum(y == g)), "recall": round(float(np.mean(p[y == g] == g)), 3)}
                      for g in grades if np.any(y == g)},
        "confusion": [[int(np.sum((y == a) & (p == b))) for b in grades] for a in grades],
    }
    json.dump(metrics, open(os.path.join(args.out, "train_data_metrics.json"), "w"), indent=1)

    print(f"\n{metrics['n_graded']} of {metrics['n_scores']} scores graded "
          f"({metrics['n_no_start_tempo']} without an explicit start tempo, {metrics['n_failed']} failed)")
    print(f"weighted accuracy {metrics['weighted_accuracy']:.3f}   weighted accuracy ±1 {metrics['weighted_accuracy_pm1']:.3f}"
          f"   accuracy {metrics['accuracy']:.3f}   accuracy ±1 {metrics['accuracy_pm1']:.3f}")
    print("per grade: " + "  ".join(f"{g}: {m['recall']:.2f} (n={m['n']})" for g, m in metrics["per_grade"].items()))
    print("confusion (rows: label 1..9, cols: predicted 1..9):")
    for g, line in zip(grades, metrics["confusion"]):
        print(f"  {g}: " + " ".join(f"{c:4d}" for c in line))
    for r in rows:
        if r["skipped"] not in ("", "no explicit start tempo"):
            print(f"  FAILED {r['key']}: {r['skipped']}")
    print(f"-> {args.out}/train_data_predictions.csv, train_data_metrics.json")


if __name__ == "__main__":
    main()
