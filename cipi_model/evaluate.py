"""Compare CIPI predictions with the dataset's difficulty labels and write the metrics.

Dataset labels are 1..9, CIPI classes are 0..8, so the label is shifted by -1.
The models output a mean over folds (and over models for the ensemble); it is rounded
half-up to the nearest class, as a single class is needed for accuracy metrics.

Outputs (in results/): RESULTS.md, metrics.json, labels_vs_predictions.csv
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

DATASET = Path(os.environ.get("DATASET", "/data/dataset.xlsx"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
MODELS = ["ensemble", "p", "argnn", "virtuoso", "argnn_fixed"]
CLASSES = list(range(9))


def to_class(value):
    if value is None or value < 0 or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(min(8, max(0, math.floor(value + 0.5))))


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return {"n": 0, **{k: None for k in ("balanced_acc9", "acc_pm1", "balanced_acc_pm1", "acc9", "mse")}}
    within1 = np.abs(y_true - y_pred) <= 1
    present = [c for c in CLASSES if (y_true == c).any()]
    return {
        "n": int(len(y_true)),
        "balanced_acc9": float(balanced_accuracy_score(y_true, y_pred)),
        "acc_pm1": float(within1.mean()),
        "balanced_acc_pm1": float(np.mean([within1[y_true == c].mean() for c in present])),
        "acc9": float((y_true == y_pred).mean()),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
    }


def main():
    labels = pd.read_excel(DATASET, sheet_name="dataset")
    labels = labels.rename(columns={labels.columns[1]: "label"})[["file_name", "composer", "song_name", "label"]]
    labels["y_true"] = labels["label"].astype(int) - 1

    records = {}
    for line in (RESULTS_DIR / "predictions.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            records[rec["file_name"]] = rec  # last record per score wins
    preds = pd.DataFrame(records.values())
    df = labels.merge(preds, on="file_name", how="inner")
    for col in ("error", "virtuoso_error"):  # absent when nothing failed
        if col not in df:
            df[col] = None
    for m in MODELS:
        df[f"{m}_class"] = df[m].map(to_class) if m in df else None

    ok = df[df["status"] == "ok"]
    errors = df[df["status"] != "ok"][["file_name", "error"]]

    results = {}
    for m in MODELS:
        sub = ok[ok[f"{m}_class"].notna()]
        results[m] = metrics(sub["y_true"], sub[f"{m}_class"].astype(int))

    # Compact table on the dataset's 1-9 scale: label vs. predicted level per model.
    table = df[["file_name", "composer", "song_name", "label"]].copy()
    for m in MODELS:
        table[f"pred_{m}"] = (pd.to_numeric(df[f"{m}_class"]) + 1).astype("Int64")
    table["ensemble_score"] = df["ensemble"].where(df["ensemble"] >= 0).add(1).round(2)
    table.sort_values("file_name", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int)) \
        .to_csv(RESULTS_DIR / "labels_vs_predictions.csv", index=False)

    summary = {
        "scores_in_dataset": int(len(labels)),
        "scores_predicted": int(len(ok)),
        "scores_failed": int(len(errors)),
        "virtuoso_failed": int((ok["virtuoso"] < 0).sum()),
        "metrics": results,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    ens = ok[ok["ensemble_class"].notna()]
    cm = confusion_matrix(ens["y_true"], ens["ensemble_class"].astype(int), labels=CLASSES)
    summary["ensemble_acc9_by_grade"] = {
        str(c + 1): {"n": int(row.sum()), "acc9": float(row[c] / row.sum()) if row.sum() else None, "mse": mse}
        for c, row, mse in zip(CLASSES, cm, grade_mse_from_cm(cm))
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, cm, errors, ok)
    print(json.dumps(summary, indent=2))


def grade_mse_from_cm(cm):
    """Per true grade: mean of (predicted - true)^2 over that grade's row; None for an empty grade."""
    dist2 = (np.arange(len(CLASSES))[None, :] - np.arange(len(CLASSES))[:, None]) ** 2
    return [float((row * d).sum() / row.sum()) if row.sum() else None for row, d in zip(cm, dist2)]


def write_report(summary, cm, errors, ok):
    r = summary["metrics"]
    names = {
        "ensemble": "**Ensemble (p + argnn + virtuoso)**",
        "p": "Pitch (p)",
        "argnn": "Fingering ArGNN (argnn)",
        "virtuoso": "VirtuosoNet (virtuoso)",
        "argnn_fixed": "argnn, RH length fixed (not upstream)",
    }
    lines = [
        "# CIPI model results on the 806-score dataset",
        "",
        "Model: pretrained CIPI models from https://github.com/PRamoneda/difficulty-prediction-CIPI "
        "(Ramoneda et al., ESWA 2023), used without retraining.",
        "Ground truth: `806_2_score_dataset_clean.xlsx` difficulty 1-9, shifted to CIPI classes 0-8.",
        "",
        f"Scores: {summary['scores_in_dataset']} in dataset, {summary['scores_predicted']} predicted, "
        f"{summary['scores_failed']} failed. VirtuosoNet failed on {summary['virtuoso_failed']} of the "
        "predicted scores (the ensemble then averages the remaining two models, as upstream does).",
        "",
        "## Main result",
        "",
        "| Model | N | Balanced acc9 | Acc ±1 | Balanced acc ±1 | Acc9 | MSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, label in names.items():
        x = r[m]
        cells = ["–" if x[k] is None else f"{x[k]:.{2 if k == 'mse' else 3}f}"
                 for k in ("balanced_acc9", "acc_pm1", "balanced_acc_pm1", "acc9", "mse")]
        lines.append(f"| {label} | {x['n']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "- **Balanced acc9**: mean per-class recall over the 9 difficulty classes (the paper's main metric; "
        "paper reports 39.5% for its best model on CIPI).",
        "- **Acc ±1**: share of scores whose predicted class is within one level of the label (as in upstream "
        "`compute_metrics`).",
        "- **Balanced acc ±1**: the same ±1 hit rate averaged per true class, so rare classes weigh equally.",
        "- **MSE**: mean squared difference between predicted and true grade, over all scores. A miss by 1 grade "
        "counts 1 and a miss by 2 grades counts 4, so large misses dominate.",
        "- Model outputs (mean of 5 fold predictions, and of 3 models for the ensemble) are rounded half-up "
        "to the nearest class.",
        "",
        "## Ensemble confusion matrix (rows = true, cols = predicted, dataset levels 1-9)",
        "",
        "Acc9 per grade = correctly predicted / N for that true grade; their mean is the balanced acc9. "
        "MSE per grade = mean squared distance between predicted and true grade for that grade; the mean row "
        "is their average (a balanced MSE).",
        "",
        "| true \\ pred | " + " | ".join(str(c + 1) for c in CLASSES) + " | N | Acc9 | MSE |",
        "|---" * (len(CLASSES) + 4) + "|",
    ]
    grade_mse = grade_mse_from_cm(cm)
    for c, row in zip(CLASSES, cm):
        n = int(row.sum())
        acc = f"{row[c] / n:.3f}" if n else "–"
        mse = f"{grade_mse[c]:.2f}" if n else "–"
        lines.append(f"| **{c + 1}** | " + " | ".join(str(v) for v in row) + f" | {n} | {acc} | {mse} |")
    mean_acc = r["ensemble"]["balanced_acc9"]
    present = [m for m in grade_mse if m is not None]
    lines.append("| **mean** | " + " | " * len(CLASSES) + f"{int(cm.sum())} | "
                 + ("–" if mean_acc is None else f"**{mean_acc:.3f}**") + " | "
                 + (f"**{np.mean(present):.2f}**" if present else "–") + " |")
    if len(errors):
        lines += ["", "## Failed scores", ""]
        lines += [f"- `{row.file_name}`: {row.error}" for row in errors.itertuples()]
    virt_fail = ok[ok["virtuoso"] < 0]
    if len(virt_fail):
        lines += ["", "## Scores where only VirtuosoNet failed", ""]
        lines += [f"- `{row.file_name}`: {getattr(row, 'virtuoso_error', '') or ''}" for row in virt_fail.itertuples()]
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
