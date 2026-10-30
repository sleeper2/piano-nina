"""Compare RubricNet predictions with the dataset's difficulty labels and write the metrics.

Dataset labels are 1..9, RubricNet (CIPI) classes are 0..8, so the label is shifted by -1.
`rubricnet` is the mean of the 5 fold-model classes; it is rounded half-up to the nearest class,
as a single class is needed for accuracy metrics. The fold models are also scored one by one.

With TEMPO=explicit (default) only scores with an explicit tempo are evaluated (see tempo.py).

Outputs (in results/): RESULTS.md, metrics.json, labels_vs_predictions.csv, features.csv
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import tempo

SCORES_DIR = Path(os.environ.get("SCORES_DIR", "/data/score_musicxml"))
DATASET = Path(os.environ.get("DATASET", "/data/dataset.xlsx"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
FOLDS = [f"fold_{k}" for k in range(5)]
MODELS = ["rubricnet"] + FOLDS
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
    n_dataset = len(labels)
    if tempo.TEMPO == "explicit":
        labels = labels[labels["file_name"].isin({f.name for f in tempo.selected(SCORES_DIR.glob("*.musicxml"))})]

    records = {}
    for line in (RESULTS_DIR / "predictions.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            records[rec["file_name"]] = rec  # last record per score wins
    preds = pd.DataFrame(records.values())
    df = labels.merge(preds, on="file_name", how="inner")
    for col in ("error", "input_fixes", "folds", "features", "rubricnet", "score_has_tempo", "bpm"):
        if col not in df:  # absent when nothing failed / nothing succeeded
            df[col] = None
    n_no_tempo = int(((df["status"] == "ok") & (df["score_has_tempo"] == False)).sum())  # noqa: E712
    if tempo.TEMPO == "marked":  # failed scores stay in, so their errors are reported
        df = df[(df["status"] != "ok") | (df["score_has_tempo"] == True)]  # noqa: E712
    for k, fold in enumerate(FOLDS):
        df[fold] = df["folds"].map(lambda f, k=k: f[k]["class"] if isinstance(f, list) else None)
    df["rubric_score"] = df["folds"].map(
        lambda f: float(np.mean([x["score"] for x in f])) if isinstance(f, list) else None)
    for m in MODELS:
        df[f"{m}_class"] = df[m].map(to_class)

    ok = df[df["status"] == "ok"]
    errors = df[df["status"] != "ok"][["file_name", "error"]]

    results = {}
    for m in MODELS:
        sub = ok[ok[f"{m}_class"].notna()]
        results[m] = metrics(sub["y_true"], sub[f"{m}_class"].astype(int))

    order = lambda s: s.str.extract(r"(\d+)")[0].astype(int)
    # Compact table on the dataset's 1-9 scale: label vs. predicted level.
    table = df[["file_name", "composer", "song_name", "label"]].copy()
    for m in MODELS:
        table[f"pred_{m}"] = (pd.to_numeric(df[f"{m}_class"]) + 1).astype("Int64")
    table["rubricnet_mean"] = pd.to_numeric(df["rubricnet"]).add(1).round(2)
    table["rubric_score_0_12"] = pd.to_numeric(df["rubric_score"]).round(2)
    table["score_has_tempo"] = df["score_has_tempo"]
    table["bpm"] = pd.to_numeric(df["bpm"]).round(1)
    table.sort_values("file_name", key=order).to_csv(RESULTS_DIR / "labels_vs_predictions.csv", index=False)

    features = pd.DataFrame(ok["features"].tolist(), index=ok.index)
    pd.concat([ok[["file_name", "label"]], features], axis=1) \
        .sort_values("file_name", key=order).to_csv(RESULTS_DIR / "features.csv", index=False)

    cm = confusion_matrix(ok["y_true"], ok["rubricnet_class"].astype(int), labels=CLASSES)
    summary = {
        "scores_in_dataset": int(n_dataset),
        "tempo_filter": tempo.TEMPO,
        "scores_selected": int(len(df) if tempo.TEMPO == "marked" else len(labels)),
        "scores_skipped_no_tempo_mark": n_no_tempo if tempo.TEMPO == "marked" else None,
        "scores_predicted": int(len(ok)),
        "scores_failed": int(len(errors)),
        "scores_without_tempo_mark": int((ok["score_has_tempo"] == False).sum()),  # noqa: E712
        "scores_with_input_fixes": int(ok["input_fixes"].map(lambda x: isinstance(x, list)).sum()),
        "metrics": results,
        "rubricnet_by_grade": {
            str(c + 1): {"n": int(row.sum()), "acc9": float(row[c] / row.sum()) if row.sum() else None, "mse": mse}
            for c, row, mse in zip(CLASSES, cm, grade_mse_from_cm(cm))
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, cm, errors)
    print(json.dumps(summary, indent=2))


def grade_mse_from_cm(cm):
    """Per true grade: mean of (predicted - true)^2 over that grade's row; None for an empty grade."""
    dist2 = (np.arange(len(CLASSES))[None, :] - np.arange(len(CLASSES))[:, None]) ** 2
    return [float((row * d).sum() / row.sum()) if row.sum() else None for row, d in zip(cm, dist2)]


def write_report(summary, cm, errors):
    r = summary["metrics"]
    names = {"rubricnet": "**RubricNet (mean of 5 fold models)**",
             **{f: f"fold model {f[-1]} alone" for f in FOLDS}}
    selection = {
        "all": "",
        "marked": f" {summary['scores_skipped_no_tempo_mark']} scores have no metronome mark (printed or playback), "
                  "so the extractor would read them at 100 bpm; they are skipped, leaving "
                  f"{summary['scores_selected']}.",
        "explicit": f" {summary['scores_selected']} have an explicit tempo (printed numeric metronome mark in the "
                    "first measure, no other tempo later, playback tempo matching it); the others are skipped.",
    }[summary["tempo_filter"]]
    title = {"all": "", "marked": " on scores with a tempo mark",
             "explicit": " on scores with an explicit tempo"}[summary["tempo_filter"]]
    lines = [
        "# RubricNet results" + title,
        "",
        "Model: pretrained RubricNet (camera-ready checkpoints) from https://github.com/PRamoneda/rubricnet "
        "(Ramoneda et al., ISMIR 2024), used without retraining.",
        "Ground truth: `806_2_score_dataset_clean.xlsx` difficulty 1-9, shifted to CIPI classes 0-8.",
        "",
        f"Scores: {summary['scores_in_dataset']} in dataset.{selection} {summary['scores_predicted']} predicted, "
        f"{summary['scores_failed']} failed. "
        + (f"{summary['scores_without_tempo_mark']} predicted scores have no metronome mark, so the extractor "
           "assumed 100 bpm (as upstream does). " if summary["scores_without_tempo_mark"] else "")
        + f"{summary['scores_with_input_fixes']} scores needed input clean-up (see README).",
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
        "- **Balanced acc9**: mean per-class recall over the 9 difficulty classes (the paper reports 41.5% on "
        "CIPI, averaged over the 5 fold models on their test splits).",
        "- **Acc ±1**: share of scores whose predicted class is within one level of the label.",
        "- **Balanced acc ±1**: the same ±1 hit rate averaged per true class, so rare classes weigh equally.",
        "- **MSE**: mean squared difference between predicted and true grade, over all scores. A miss by 1 grade "
        "counts 1 and a miss by 2 grades counts 4, so large misses dominate. (The paper's MSE of 1.52 is the "
        "per-class average, comparable to the mean row of the confusion matrix below.)",
        "- Each fold model outputs an ordinal class; the main row averages the 5 classes and rounds half-up.",
        "",
        "## RubricNet confusion matrix (rows = true, cols = predicted, dataset levels 1-9)",
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
    mean_acc = r["rubricnet"]["balanced_acc9"]
    present = [m for m in grade_mse if m is not None]
    lines.append("| **mean** | " + " | " * len(CLASSES) + f"{int(cm.sum())} | "
                 + ("–" if mean_acc is None else f"**{mean_acc:.3f}**") + " | "
                 + (f"**{np.mean(present):.2f}**" if present else "–") + " |")
    if len(errors):
        lines += ["", "## Failed scores", ""]
        lines += [f"- `{row.file_name}`: {row.error}" for row in errors.itertuples()]
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
