"""Compare CIPI ensemble and RubricNet errors on the scores both models report (RubricNet TEMPO=marked, 560).

Reads cipi_model/results and rubricnet_model/results (no re-prediction) plus the dataset sheet, and writes
results/both_models.csv (one row per score), results/shared_misses.csv (both models off by >= 2 grades)
and results/PATTERNS.md.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
BAD = 2  # "badly wrong" = off by this many grades or more


def load():
    """All 806 scores; `score_has_tempo` marks the 560 that RubricNet reports."""
    c = pd.read_csv(ROOT / "cipi_model/results/labels_vs_predictions.csv")
    r = pd.read_csv(ROOT / "rubricnet_model/results/labels_vs_predictions.csv")
    xl = pd.read_excel(ROOT / "806_2_score_dataset_clean.xlsx", sheet_name="dataset")
    xl = xl[["file_name", "notes_count", "measures_count"]]
    rj = pd.DataFrame(json.loads(line) for line in open(ROOT / "rubricnet_model/results/predictions.jsonl"))
    assert (rj["status"] == "ok").all()
    feats = pd.DataFrame(rj["features"].tolist())
    rj = pd.concat([rj[["file_name", "rubricnet", "score_has_tempo", "bpm", "n_staves"]], feats], axis=1)
    # Same rounding as rubricnet_model/evaluate.py: mean fold class, half-up, shifted to grades 1-9.
    rj["pred_rubricnet"] = np.floor(rj["rubricnet"] + 0.5).clip(0, 8).astype(int) + 1
    rj["rubricnet_mean"] = (rj["rubricnet"] + 1).round(2)
    check = r.merge(rj, on="file_name", suffixes=("", "_jsonl"))
    assert len(check) == len(r) and (check["pred_rubricnet"] == check["pred_rubricnet_jsonl"]).all()

    m = c[["file_name", "composer", "song_name", "label", "pred_ensemble", "ensemble_score"]].rename(
        columns={"pred_ensemble": "pred_cipi", "ensemble_score": "cipi_mean"})
    m = m.merge(rj.drop(columns="rubricnet"), on="file_name").merge(xl, on="file_name")
    m.insert(0, "id", m["file_name"].str.extract(r"(\d+)", expand=False).astype(int))

    m["err_rubricnet"] = m["pred_rubricnet"] - m["label"]
    m["err_cipi"] = m["pred_cipi"] - m["label"]
    ae_r, ae_c = m["err_rubricnet"].abs(), m["err_cipi"].abs()
    m["both_exact"] = (ae_r == 0) & (ae_c == 0)
    m["both_wrong"] = (ae_r > 0) & (ae_c > 0)
    m["both_off2"] = (ae_r >= BAD) & (ae_c >= BAD)
    m["direction"] = np.select(
        [(m["err_rubricnet"] > 0) & (m["err_cipi"] > 0), (m["err_rubricnet"] < 0) & (m["err_cipi"] < 0)],
        ["both over", "both under"], np.where(m["both_wrong"], "opposite", ""))
    m["notes_per_sec"] = 1 / m["rh_average_ioi_seconds"] + 1 / m["lh_average_ioi_seconds"]
    # Onsets per minute from RubricNet's per-hand average IOI: mean of both hands, and the faster hand.
    m["ioi_tempo_avg"] = 60 / ((m["rh_average_ioi_seconds"] + m["lh_average_ioi_seconds"]) / 2)
    m["ioi_tempo_max"] = 60 / m[["rh_average_ioi_seconds", "lh_average_ioi_seconds"]].min(axis=1)
    return m.sort_values("id").reset_index(drop=True)


def md(df, index=True):
    df = df.reset_index() if index else df
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def accuracies(y, p):
    """Plain and balanced acc9, acc3 (buckets 1-3/4-6/7-9, as upstream's 3bacc) and acc ±1."""
    y, p = np.asarray(y), np.asarray(p)
    bucket = lambda v: (v - 1) // 3  # noqa: E731
    hit9, hit3, hit1 = p == y, bucket(p) == bucket(y), np.abs(p - y) <= 1
    balanced = lambda hit, cls: np.mean([hit[cls == k].mean() for k in np.unique(cls)])  # noqa: E731
    return {"Acc9": hit9.mean(), "Bal. acc9": balanced(hit9, y), "Acc3": hit3.mean(),
            "Bal. acc3": balanced(hit3, bucket(y)), "Acc ±1": hit1.mean(), "Bal. acc ±1": balanced(hit1, y)}


def binned_baseline(m, col, bins=10, folds=5, seed=0):
    """One-feature baseline: mean label of the training scores in the same `col` decile, rounded, 5-fold CV."""
    fold = np.random.default_rng(seed).integers(0, folds, len(m))
    pred = np.zeros(len(m))
    for k in range(folds):
        train, test = m[fold != k], fold == k
        edges = np.quantile(train[col], np.linspace(0, 1, bins + 1))[1:-1]
        means = train["label"].groupby(np.digitize(train[col], edges)).mean()
        pred[test] = np.round(means.reindex(np.digitize(m.loc[test, col], edges)).values)
    return pred.astype(int)


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z partialled out (linear on ranks)."""
    x, y, z = (rankdata(v) for v in (x, y, z))
    rx, ry = x - np.polyval(np.polyfit(z, x, 1), z), y - np.polyval(np.polyfit(z, y, 1), z)
    return np.corrcoef(rx, ry)[0, 1]


def baselines_section(m, m_all):
    rows = [("Note count only", m, binned_baseline(m, "notes_count")),
            ("IOI tempo only", m, binned_baseline(m, "ioi_tempo_avg")),
            ("RubricNet", m, m["pred_rubricnet"]),
            ("CIPI ensemble", m, m["pred_cipi"]),
            ("CIPI ensemble", m_all, m_all["pred_cipi"])]
    t = pd.DataFrame([{"": name, "N": len(d), **accuracies(d["label"], p)} for name, d, p in rows])
    num = t.columns[2:]
    t[num] = t[num].round(3)
    for col in num:  # bold the best value per column
        t[col] = [f"**{v:.3f}**" if v == t[col].max() else f"{v:.3f}" for v in t[col]]

    return f"""Can one feature do as well as the models? The "only" rows predict each score as the mean label of the
training scores in the same decile of that feature, rounded (5-fold cross-validation, so a score is never predicted
from its own label). IOI tempo is `ioi_tempo_avg` (see the next section).

Acc3 uses the buckets 1–3, 4–6 and 7–9; balanced metrics average the hit rate per true class (grades for acc9 and
acc ±1, buckets for acc3). Bold = best per column. All rows use the {len(m)} scores except the last, CIPI on all
{len(m_all)}.

{md(t, index=False)}

A single feature matches or beats both models on the plain metrics; the models win on the balanced ones, where rare
grades count as much as the common grades 2–3.
"""


def length_section(m):
    rows = ["notes_count", "measures_count", "ioi_tempo_avg", "ioi_tempo_max"]
    overall = pd.DataFrame({
        "dataset label": [spearmanr(m[f], m["label"])[0] for f in rows],
        "RubricNet prediction": [spearmanr(m[f], m["pred_rubricnet"])[0] for f in rows],
        "CIPI prediction": [spearmanr(m[f], m["pred_cipi"])[0] for f in rows],
    }, index=pd.Index(rows, name="feature")).round(2)
    label_r = spearmanr(m["label"], m["pred_rubricnet"])[0]
    label_c = spearmanr(m["label"], m["pred_cipi"])[0]
    partial = pd.DataFrame({
        f: {"RubricNet": partial_spearman(m["pred_rubricnet"], m["label"], m[f]),
            "CIPI": partial_spearman(m["pred_cipi"], m["label"], m[f])}
        for f in ("notes_count", "ioi_tempo_avg")}).T.round(2)
    partial.index.name = "held constant"

    def by_quintile(col, names):
        q = pd.qcut(m[col], 5, labels=names)
        return m.groupby(q, observed=True).agg(
            N=("id", "size"), median=(col, "median"), mean_label=("label", "mean"),
            bias_rubricnet=("err_rubricnet", "mean"), bias_cipi=("err_cipi", "mean")).round(2).rename(
            columns={"median": f"median {col}"})

    by_notes = by_quintile("notes_count", ["Q1 shortest", "Q2", "Q3", "Q4", "Q5 longest"]).rename_axis("note count")
    by_tempo = by_quintile("ioi_tempo_avg", ["Q1 slowest", "Q2", "Q3", "Q4", "Q5 fastest"]).rename_axis("IOI tempo")

    length_only = binned_baseline(m, "notes_count") - m["label"]
    corr = pd.DataFrame({
        g: {"N": len(d),
            "RubricNet": round(spearmanr(d["err_rubricnet"], d["notes_count"])[0], 2),
            "CIPI": round(spearmanr(d["err_cipi"], d["notes_count"])[0], 2),
            "note-count-only baseline": round(spearmanr(length_only[d.index], d["notes_count"])[0], 2)}
        for g, d in m.groupby("label")}).T
    corr["N"] = corr["N"].astype(int)
    corr.index.name = "grade"

    feats = ["notes_count", "measures_count", "notes_per_sec", "rh_pitch_range", "lh_pitch_range",
             "rh_pitch_set_lz", "rh_displacement_rate", "bpm"]

    def contrast(miss, hit):
        rows = [(f, miss[f].median(), hit[f].median(), mannwhitneyu(miss[f], hit[f]).pvalue) for f in feats]
        t = pd.DataFrame(rows, columns=["feature", "median: both miss", "median: ≥1 exact", "p (Mann-Whitney)"]).round(3)
        t["p (Mann-Whitney)"] = t["p (Mann-Whitney)"].map(lambda p: "<0.001" if p < 0.001 else p)
        return md(t, index=False)

    low, high = m[m["label"].isin([3, 4])], m[m["label"] >= 7]
    low_over = low[(low["err_rubricnet"] > 0) & (low["err_cipi"] > 0)]
    high_under = high[(high["err_rubricnet"] < 0) & (high["err_cipi"] < 0)]
    low_hit = low[low["err_rubricnet"].eq(0) | low["err_cipi"].eq(0)]
    high_hit = high[high["err_rubricnet"].eq(0) | high["err_cipi"].eq(0)]

    return f"""Spearman correlation between length or IOI tempo and grade across all {len(m)} scores: the dataset's own
labels next to each model's predicted grade. IOI tempo is note onsets per minute, 60 / average IOI in seconds, from
RubricNet's per-hand descriptors: `ioi_tempo_avg` averages the IOI of both hands, `ioi_tempo_max` takes the faster hand.

{md(overall)}

The labels are strongly tied to both length and IOI tempo. The predictions follow length at least as closely as the
labels do, but follow tempo less closely (the models under-use speed). For comparison, the predictions correlate with
the labels at {label_r:.2f} (RubricNet) and {label_c:.2f} (CIPI).

Do the models know more than length or tempo? Spearman between prediction and label with one feature held constant
(partial correlation on ranks):

{md(partial)}

### Is length or tempo over-weighted?

Mean signed error (predicted − label) by quintile of the feature, all grades pooled. A model that uses the feature
as much as the labels justify has about the same bias in every row.

{md(by_notes)}

{md(by_tempo)}

### A pitfall: error vs length within a grade

Within each grade, Spearman between signed error and note count is strongly positive for both models. This does not
show that they over-use length: any predictor that uses length shows it, because the label is fixed within a grade
while the prediction moves with length. The note-count-only baseline shows it even more strongly:

{md(corr)}

### What the shared misses look like

#### Grades 3–4 rated too high by both ({len(low_over)}) vs grades 3–4 with ≥1 exact model ({len(low_hit)})

{contrast(low_over, low_hit)}

#### Grades 7–9 rated too low by both ({len(high_under)}) vs grades 7–9 with ≥1 exact model ({len(high_hit)})

{contrast(high_under, high_hit)}
"""


def report(m, m_all):
    n = len(m)
    ae_r, ae_c = m["err_rubricnet"].abs().clip(upper=4), m["err_cipi"].abs().clip(upper=4)
    overlap = pd.crosstab(ae_r.rename("RubricNet |err|"), ae_c.rename("CIPI |err|"), margins=True)
    overlap.index = overlap.index.map(lambda v: "4+" if v == 4 else v)
    overlap.columns = overlap.columns.map(lambda v: "4+" if v == 4 else v)

    grade = m.groupby("label").agg(
        N=("id", "size"),
        both_exact=("both_exact", "sum"),
        both_wrong=("both_wrong", "sum"),
        both_off2=("both_off2", "sum"),
        mean_pred_rubricnet=("pred_rubricnet", "mean"),
        mean_pred_cipi=("pred_cipi", "mean"),
    )
    grade.insert(3, "both_wrong_share", (grade["both_wrong"] / grade["N"]))
    grade.insert(3, "median_ioi_tempo", m.groupby("label")["ioi_tempo_avg"].median().round().astype(int))
    grade.insert(3, "median_notes", m.groupby("label")["notes_count"].median().round().astype(int))
    dirs = pd.crosstab(m.loc[m["both_wrong"], "label"], m.loc[m["both_wrong"], "direction"])
    grade = grade.join(dirs).fillna(0).round(2)
    grade.index.name = "grade"

    cols = ["id", "label", "pred_rubricnet", "pred_cipi", "composer", "song_name", "measures_count", "notes_count"]
    worst = m[m["both_off2"]].sort_values(["label", "id"])[cols]

    text = f"""# Where CIPI ensemble and RubricNet both fail

Scores: the {n} that RubricNet reports (TEMPO=marked); CIPI ensemble predictions for the same scores.
"Both wrong" = neither model predicts the exact grade; "both off ≥{BAD}" = both miss by {BAD}+ grades.

- Both exact: {m['both_exact'].sum()}; both wrong: {m['both_wrong'].sum()} ({m['both_wrong'].mean():.0%});
  both off ≥{BAD}: {m['both_off2'].sum()}.
- Error correlation (signed, per score): {m['err_rubricnet'].corr(m['err_cipi']):.2f}. When both are wrong,
  they err in the same direction in {(m.loc[m['both_wrong'], 'direction'] != 'opposite').mean():.0%} of cases.
- Mean signed error: RubricNet {m['err_rubricnet'].mean():+.2f}, CIPI {m['err_cipi'].mean():+.2f}.
- Both compress the scale: grades 3–4 are mostly rated too high and grades 7–9 too low (see Per grade).
- A one-feature baseline (note count or IOI tempo) matches or beats both models on plain acc9 / acc3 / acc ±1;
  the models win on the balanced metrics (see Accuracy against one-feature baselines).
- The labels correlate with note count at {spearmanr(m['notes_count'], m['label'])[0]:.2f} and with IOI tempo at
  {spearmanr(m['ioi_tempo_avg'], m['label'])[0]:.2f}. The models follow length at least as closely as the labels do
  but under-use tempo; with length held constant they still correlate with the labels (see Length and tempo).

## Error overlap (|predicted − true|, 4+ grouped)

{md(overlap)}

## Per grade

`both over` / `both under` / `opposite` count the both-wrong scores by direction of the two errors.
`median_notes` and `median_ioi_tempo` (note onsets per minute, see Length and tempo) are over all scores of the grade.

{md(grade)}

## Accuracy against one-feature baselines

{baselines_section(m, m_all)}
## Length and tempo

{length_section(m)}
## Scores both models miss by {BAD}+ grades ({len(worst)})

{md(worst, index=False)}
"""
    (OUT / "PATTERNS.md").write_text(text)


def main():
    OUT.mkdir(exist_ok=True)
    m_all = load()
    m = m_all[m_all["score_has_tempo"]].reset_index(drop=True)
    keep = ["id", "file_name", "composer", "song_name", "label", "pred_rubricnet", "pred_cipi", "err_rubricnet",
            "err_cipi", "both_exact", "both_wrong", "both_off2", "direction", "rubricnet_mean", "cipi_mean",
            "notes_count", "measures_count", "bpm", "notes_per_sec"]
    m[keep].to_csv(OUT / "both_models.csv", index=False)
    m.loc[m["both_off2"], keep].to_csv(OUT / "shared_misses.csv", index=False)
    report(m, m_all)
    print(f"{len(m)} scores; both wrong {m['both_wrong'].sum()}, both off >= {BAD}: {m['both_off2'].sum()}")


if __name__ == "__main__":
    main()
