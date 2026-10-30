"""Feature groups B-E of FEATURES_SPEC.md: pitch span, black keys, hand displacement and hand crossing, one row
per score and one per measure, their correlation with the annotator's grade, and the held-out comparison with the
baseline (FEATURES_SPEC.md section 11). Group A (written bimanual coordination) is in coordination.py.

Hands (FEATURES_SPEC.md section 3): the top staff is the right hand and the bottom staff the left hand; extra staves
join the hand with the closer mean pitch ("heuristic"); single-staff scores get no hand features ("none"). Notes are
attacks only (no grace notes, no tied continuations); onsets are exact fractions of a quarter note.

Outputs (results/): hand_features.csv, hand_features_per_measure.csv, hand_correlations.csv, hand_validation.csv,
HAND_FEATURES.md and figures/hf_*.png.
"""
import os
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

from extract import DEFAULT_TEMPO, SCORES, parse

BLACK = {1, 3, 6, 8, 10}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def assign_hands(notes):
    """Map (part, staff) -> 'R' / 'L', plus a confidence flag."""
    counts = Counter((n[0], n[1]) for n in notes if n[6])
    if len(counts) < 2:
        return None, "none"
    total = sum(counts.values())
    main = sorted(k for k, c in counts.items() if c / total >= 0.10)
    if len(main) < 2:
        return None, "none"
    rh, lh = main[0], main[-1]
    mean = {k: np.mean([n[5] for n in notes if (n[0], n[1]) == k and n[6]]) for k in counts}
    hands = {rh: "R", lh: "L"}
    for k in counts:
        if k not in hands:
            hands[k] = "R" if abs(mean[k] - mean[rh]) <= abs(mean[k] - mean[lh]) else "L"
    confident = len(counts) == 2 and mean[rh] > mean[lh]
    return hands, "staff" if confident else "heuristic"


def stats(values, prefix, extra=()):
    v = np.asarray(values, dtype=float)
    names = ("mean", "median", "std", "iqr", "p95", "max", *[f"gt{t}" for t in extra])
    if not len(v):
        return {f"{prefix}_{s}": np.nan for s in names}
    q1, q3 = np.percentile(v, [25, 75])
    out = {f"{prefix}_mean": v.mean(), f"{prefix}_median": np.median(v), f"{prefix}_std": v.std(),
           f"{prefix}_iqr": q3 - q1, f"{prefix}_p95": np.percentile(v, 95), f"{prefix}_max": v.max()}
    out.update({f"{prefix}_gt{t}": (v > t).mean() for t in extra})
    return out


def longest_run(mask):
    best = cur = 0
    for x in mask:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def time_map(tempos):
    """Seconds elapsed at a quarter-note offset under the stored tempo events (same rule as seconds_at)."""
    ev = sorted(tempos)
    if not ev or ev[0][0] > 0:
        ev = [(0.0, DEFAULT_TEMPO)] + ev
    qs = np.array([float(q) for q, _ in ev])
    rate = np.array([60 / t if t > 0 else 0.0 for _, t in ev])
    at_event = np.concatenate([[0.0], np.cumsum(np.diff(qs) * rate[:-1])])

    def seconds(q):
        i = np.clip(np.searchsorted(qs, q, side="right") - 1, 0, len(qs) - 1)
        return at_event[i] + (np.asarray(q, float) - qs[i]) * rate[i]
    return seconds


def features(file_name):
    """One row of piece-level features and a list of per-measure rows for one score."""
    parsed = parse(os.path.join(SCORES, file_name), exact=True)
    notes, tempos, measure_len, n_measures = parsed[1], parsed[4], parsed[5], parsed[9]
    lengths = np.array([float(measure_len[i]) for i in range(len(measure_len))])
    starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    to_sec = time_map(tempos)
    secs = float(to_sec(lengths.sum()))
    quarters = float(lengths.sum())
    row = {"file_name": file_name, "measures": n_measures, "quarters": quarters, "seconds": secs}
    hands, confidence = assign_hands(notes)
    row["hand_confidence"] = confidence
    if hands is None:
        return row, []
    measure_of = lambda t: int(np.searchsorted(starts, float(t) + 1e-9, side="right") - 1)  # noqa: E731

    groups = {"R": defaultdict(list), "L": defaultdict(list)}
    for p, s, _v, onset, _d, midi, attack in notes:
        if attack:
            groups[hands[(p, s)]][onset].append(midi)  # exact Fraction onsets
    R, L = groups["R"], groups["L"]
    union = sorted(set(R) | set(L))
    both = set(R) & set(L)
    per_min = secs / 60
    # per-measure accumulators
    pm = defaultdict(lambda: defaultdict(float))

    # B1 span of everything struck together (both hands)
    span_all = []
    for t in union:
        g = R.get(t, []) + L.get(t, [])
        span_all.append(max(g) - min(g))
        k = pm[measure_of(t)]
        k["simultaneous_span_max"] = max(k["simultaneous_span_max"], span_all[-1])
    row.update(stats(span_all, "simultaneous_pitch_span"))
    for t in (12, 19, 24):
        row[f"simultaneous_pitch_span_ge{t}"] = np.mean(np.array(span_all) >= t)

    # B2/B3 span within one hand
    span = {h: [max(g) - min(g) for g in groups[h].values()] for h in "RL"}
    pooled = np.array(span["R"] + span["L"])
    row.update(stats(pooled, "hand_pitch_span", (12, 19, 24)))
    row.update(stats(span["R"], "rh_pitch_span", (12, 19, 24)))
    row.update(stats(span["L"], "lh_pitch_span", (12, 19, 24)))
    for t in (12, 19, 24):
        row[f"large_hand_span_attacks_gt{t}"] = int((pooled > t).sum())
        row[f"large_hand_span_attacks_per_measure_gt{t}"] = row[f"large_hand_span_attacks_gt{t}"] / n_measures
    for h in "RL":
        for t, g in groups[h].items():
            k = pm[measure_of(t)]
            k["hand_span_max"] = max(k["hand_span_max"], max(g) - min(g))
            k["hand_spans_gt12"] += max(g) - min(g) > 12
            k["notes"] += len(g)
            k["black_key_notes"] += sum(int(round(p)) % 12 in BLACK for p in g)

    # C black keys (sounding pitch class)
    is_black = {h: [[int(round(p)) % 12 in BLACK for p in g] for g in groups[h].values()] for h in "RL"}
    flat = {h: [b for g in is_black[h] for b in g] for h in "RL"}
    all_notes = flat["R"] + flat["L"]
    row["black_key_notes"] = int(sum(all_notes))
    row["black_key_note_ratio"] = np.mean(all_notes)
    row["black_key_note_ratio_rh"] = np.mean(flat["R"]) if flat["R"] else np.nan
    row["black_key_note_ratio_lh"] = np.mean(flat["L"]) if flat["L"] else np.nan
    att = {h: [any(g) for g in is_black[h]] for h in "RL"}
    row["black_key_attack_ratio"] = np.mean(att["R"] + att["L"])
    row["black_key_attack_ratio_rh"] = np.mean(att["R"]) if att["R"] else np.nan
    row["black_key_attack_ratio_lh"] = np.mean(att["L"]) if att["L"] else np.nan
    row["black_key_notes_per_second"] = row["black_key_notes"] / secs if secs else np.nan
    row["black_key_notes_per_measure"] = row["black_key_notes"] / n_measures
    row["black_key_notes_per_quarter"] = row["black_key_notes"] / quarters

    # D displacement between successive attacks of one hand
    disp, cent, runs = {}, [], []
    for h in "RL":
        onsets = sorted(groups[h])
        seq = [groups[h][t] for t in onsets]
        med = np.array([np.median(g) for g in seq])
        mean = np.array([np.mean(g) for g in seq])
        disp[h] = np.abs(np.diff(med))
        cent += list(np.abs(np.diff(mean)))
        runs.append(longest_run(disp[h] > 12))
        for t, d in zip(onsets[1:], disp[h]):  # a transition belongs to the measure of its second attack
            k = pm[measure_of(t)]
            k["displacement_sum"] += d
            k["displacement_max"] = max(k["displacement_max"], d)
            k["leaps_gt12"] += d > 12
    d_all = np.concatenate([disp["R"], disp["L"]])
    row.update(stats(d_all, "hand_displacement_semitones", (7, 12, 19)))
    row.update(stats(disp["R"], "rh_displacement", (7, 12, 19)))
    row.update(stats(disp["L"], "lh_displacement", (7, 12, 19)))
    row.update(stats(cent, "hand_displacement_centroid", (7, 12, 19)))
    for t in (7, 12, 19):
        row[f"large_hand_leaps_gt{t}"] = int((d_all > t).sum())
        row[f"large_hand_leaps_per_measure_gt{t}"] = row[f"large_hand_leaps_gt{t}"] / n_measures
    row["longest_large_leap_run_gt12"] = max(runs)
    row["hand_displacement_per_second"] = d_all.sum() / secs if secs else np.nan
    row["hand_displacement_per_measure"] = d_all.sum() / n_measures
    row["hand_displacement_per_quarter"] = d_all.sum() / quarters

    # E hand crossing at onsets where both hands attack
    cross = overlap = 0
    for t in both:
        c, o = np.median(L[t]) > np.median(R[t]), max(L[t]) > min(R[t])
        cross, overlap = cross + c, overlap + o
        pm[measure_of(t)]["crossings"] += c
    row["hand_crossing_count"] = int(cross)
    row["hand_overlap_count"] = int(overlap)
    row["hand_crossing_per_measure"] = cross / n_measures
    row["hand_crossing_per_minute"] = cross / per_min if per_min else np.nan
    row["hand_overlap_per_measure"] = overlap / n_measures

    measures = []
    for i, (start, length) in enumerate(zip(starts, lengths)):
        k = pm.get(i, {})
        measures.append({"file_name": file_name, "measure": i + 1, "start_quarter": round(start, 4),
                         "quarters": length, "seconds": round(float(to_sec(start + length) - to_sec(start)), 3),
                         **{c: k.get(c, 0) for c in MEASURE_COLS}})
    return row, measures


MEASURE_COLS = ["notes", "simultaneous_span_max", "hand_span_max", "hand_spans_gt12", "black_key_notes", "displacement_sum",
                "displacement_max", "leaps_gt12", "crossings"]


# ---------------------------------------------------------------- correlations

GROUPS = {  # feature -> (group, label) for the report and the figure
    "simultaneous_pitch_span_median": ("B span", "Simultaneous span, median"),
    "simultaneous_pitch_span_mean": ("B span", "Simultaneous span, mean"),
    "simultaneous_pitch_span_p95": ("B span", "Simultaneous span, p95"),
    "simultaneous_pitch_span_max": ("B span", "Simultaneous span, max"),
    "simultaneous_pitch_span_ge12": ("B span", "Simultaneous span ≥ 12"),
    "simultaneous_pitch_span_ge19": ("B span", "Simultaneous span ≥ 19"),
    "simultaneous_pitch_span_ge24": ("B span", "Simultaneous span ≥ 24"),
    "hand_pitch_span_mean": ("B span", "Hand span, mean"),
    "hand_pitch_span_median": ("B span", "Hand span, median"),
    "hand_pitch_span_p95": ("B span", "Hand span, p95"),
    "hand_pitch_span_max": ("B span", "Hand span, max"),
    "rh_pitch_span_p95": ("B span", "RH span, p95"),
    "lh_pitch_span_p95": ("B span", "LH span, p95"),
    "hand_pitch_span_gt12": ("B span", "Hand span > 12 (share)"),
    "hand_pitch_span_gt19": ("B span", "Hand span > 19 (share)"),
    "large_hand_span_attacks_per_measure_gt12": ("B span", "Hand span > 12 per measure"),
    "large_hand_span_attacks_per_measure_gt19": ("B span", "Hand span > 19 per measure"),
    "black_key_note_ratio": ("C black", "Black-key note ratio"),
    "black_key_note_ratio_rh": ("C black", "Black-key note ratio, RH"),
    "black_key_note_ratio_lh": ("C black", "Black-key note ratio, LH"),
    "black_key_attack_ratio": ("C black", "Black-key attack ratio"),
    "black_key_notes_per_second": ("C black", "Black-key notes per second"),
    "black_key_notes_per_measure": ("C black", "Black-key notes per measure"),
    "hand_displacement_semitones_mean": ("D displacement", "Displacement, mean"),
    "hand_displacement_semitones_median": ("D displacement", "Displacement, median"),
    "hand_displacement_semitones_std": ("D displacement", "Displacement, std"),
    "hand_displacement_semitones_p95": ("D displacement", "Displacement, p95"),
    "hand_displacement_semitones_max": ("D displacement", "Displacement, max"),
    "hand_displacement_semitones_gt7": ("D displacement", "Displacement > 7 (share)"),
    "hand_displacement_semitones_gt12": ("D displacement", "Displacement > 12 (share)"),
    "hand_displacement_semitones_gt19": ("D displacement", "Displacement > 19 (share)"),
    "rh_displacement_mean": ("D displacement", "RH displacement, mean"),
    "lh_displacement_mean": ("D displacement", "LH displacement, mean"),
    "hand_displacement_centroid_mean": ("D displacement", "Centroid displacement, mean"),
    "hand_displacement_centroid_p95": ("D displacement", "Centroid displacement, p95"),
    "large_hand_leaps_per_measure_gt7": ("D displacement", "Leaps > 7 per measure"),
    "large_hand_leaps_per_measure_gt12": ("D displacement", "Leaps > 12 per measure"),
    "large_hand_leaps_per_measure_gt19": ("D displacement", "Leaps > 19 per measure"),
    "longest_large_leap_run_gt12": ("D displacement", "Longest run of leaps > 12"),
    "hand_displacement_per_second": ("D displacement", "Displacement per second"),
    "hand_crossing_per_measure": ("E crossing", "Hand crossings per measure"),
    "hand_crossing_per_minute": ("E crossing", "Hand crossings per minute"),
    "hand_overlap_per_measure": ("E crossing", "Hand overlaps per measure"),
}
CONTROLS = ["notes_count", "notes_per_sec_written", "first_tempo_qpm", "seconds", "polyphony_mean"]
NOT_FEATURES = {"file_name", "measures", "quarters", "seconds", "hand_confidence"}
RAW_COUNTS = {"black_key_notes", "hand_crossing_count", "hand_overlap_count",
              *[f"large_hand_span_attacks_gt{t}" for t in (12, 19, 24)],
              *[f"large_hand_leaps_gt{t}" for t in (7, 12, 19)]}


def group_of(col):
    if col in GROUPS:
        return GROUPS[col][0]
    for key, g in (("span", "B span"), ("black", "C black"),
                   ("displacement", "D displacement"), ("leap", "D displacement"), ("crossing", "E crossing"),
                   ("overlap", "E crossing")):
        if key in col:
            return g
    raise ValueError(col)


def partial_rank(x, y, z):
    """Correlation of the ranks of x and y after regressing both on the ranks of the columns of z."""
    rx, ry = rankdata(x), rankdata(y)
    zr = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in z.T])
    res = lambda v: v - zr @ np.linalg.lstsq(zr, v, rcond=None)[0]  # noqa: E731
    return np.corrcoef(res(rx), res(ry))[0, 1]


def correlate(m, cols, meta=None):
    """Every feature column: rho with CI, tau-c, partials, and rho on the scores with a printed metronome mark.
    `meta(col)` gives (group, label); the default uses GROUPS."""
    from analyze import boot_ci
    meta = meta or (lambda col: (group_of(col), GROUPS.get(col, (None, col))[1]))
    rows = []
    for col in cols:
        d = m[[col, "label", "tempo_reliable", *CONTROLS]].dropna()
        x, y = d[col].values.astype(float), d["label"].values
        rlo, rhi = boot_ci(lambda a, b: spearmanr(a, b)[0], x, y)
        rel = d[d["tempo_reliable"]]
        group, label = meta(col)
        rows.append({
            "group": group, "feature": col, "label": label, "raw_count": col in RAW_COUNTS,
            "in_figure": col in GROUPS, "N": len(d), "missing": len(m) - len(d),
            "rho": spearmanr(x, y)[0], "rho_lo": rlo, "rho_hi": rhi, "tau_c": kendalltau(x, y, variant="c")[0],
            "partial_notes": partial_rank(x, y, d[["notes_count"]].values),
            "partial_all": partial_rank(x, y, d[CONTROLS].values),
            "rho_reliable_tempo": spearmanr(rel[col], rel["label"])[0]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- held-out evaluation (spec section 9.4-9.7)

BASELINE = "Baseline (5 controls)"
METRICS = ["QWK", "Acc9", "Bal. acc9", "Acc ±1", "MAE"]


def ordinal_metrics(y, p):
    """Quadratic weighted kappa (grades 1-9), exact / balanced / ±1 accuracy and mean absolute error."""
    conf = np.bincount((y - 1) * 9 + (p - 1), minlength=81).reshape(9, 9).astype(float)
    expected = np.outer(conf.sum(1), conf.sum(0)) / conf.sum()
    w = (np.subtract.outer(np.arange(9), np.arange(9)) ** 2) / 64
    acc = p == y
    return {"QWK": 1 - (w * conf).sum() / (w * expected).sum(), "Acc9": acc.mean(),
            "Bal. acc9": np.mean([acc[y == k].mean() for k in np.unique(y)]),
            "Acc ±1": (np.abs(p - y) <= 1).mean(), "MAE": np.abs(p - y).mean()}


def cv_predict(X, y, groups, split, folds=5):
    """Out-of-fold grades from a ridge regression on quantile-normalised features, rounded half-up to 1-9."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer
    splitter = GroupKFold(folds) if split == "composer" else KFold(folds, shuffle=True, random_state=0)
    pred = np.zeros(len(y))
    for train, test in splitter.split(X, y, groups):
        model = make_pipeline(SimpleImputer(strategy="median"),
                              QuantileTransformer(n_quantiles=200, output_distribution="normal"),
                              RidgeCV(alphas=np.logspace(-2, 3, 21)))
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return np.clip(np.floor(pred + 0.5), 1, 9).astype(int)


def evaluate(m, configs, splits=("composer", "random"), n_boot=1000, seed=0):
    """Held-out metrics per model config (name -> feature columns), with 95% bootstrap CIs and paired
    differences from the baseline config. The same bootstrap resamples are used for every config."""
    y, comp = m["label"].values, m["composer"].values
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    rows = []
    for split in splits:
        preds = {name: cv_predict(m[cols].values.astype(float), y, comp, split) for name, cols in configs.items()}
        base_point = ordinal_metrics(y, preds[BASELINE])
        base_boot = pd.DataFrame([ordinal_metrics(y[i], preds[BASELINE][i]) for i in boots])
        for name, p in preds.items():
            point = ordinal_metrics(y, p)
            bs = pd.DataFrame([ordinal_metrics(y[i], p[i]) for i in boots])
            row = {"split": split, "model": name, "features": len(configs[name])}
            for k in METRICS:
                row[k] = point[k]
                row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(bs[k], [2.5, 97.5])
                row[f"d_{k}"] = point[k] - base_point[k]
                row[f"d_{k}_lo"], row[f"d_{k}_hi"] = np.percentile(bs[k] - base_boot[k], [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)


def validate(m, c):
    """Baseline against baseline + each feature group, each group alone, and all groups (raw counts left out)."""
    feats = c[~c["raw_count"]]
    group_cols = {g: d["feature"].tolist() for g, d in feats.groupby("group")}
    configs = {BASELINE: CONTROLS}
    configs.update({f"Baseline + {g}": CONTROLS + cols for g, cols in group_cols.items()})
    configs["Baseline + all groups"] = CONTROLS + feats["feature"].tolist()
    configs.update({f"{g} only": cols for g, cols in group_cols.items()})
    configs["All groups only"] = feats["feature"].tolist()
    return evaluate(m, configs)


def validation_figure(v, name, title_note=""):
    import matplotlib.pyplot as plt
    from analyze import AXIS, BLUE, INK2, ORANGE, SURFACE, save
    d = v[(v["split"] == "composer") & (v["model"] != BASELINE)].iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.3 * len(d) + 1.6), sharey=True)
    y = np.arange(len(d))
    for ax, k, title, better in ((axes[0], "QWK", "Δ quadratic weighted κ", "higher"),
                                 (axes[1], "MAE", "Δ mean absolute error (grades)", "lower")):
        colour = [BLUE if name.startswith("Baseline +") else ORANGE for name in d["model"]]
        ax.hlines(y, d[f"d_{k}_lo"], d[f"d_{k}_hi"], color=colour, linewidth=2)
        ax.scatter(d[f"d_{k}"], y, color=colour, s=30, edgecolor=SURFACE, zorder=3)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_title(f"{title}, {better} is better", fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, d["model"])
    fig.suptitle("Difference from the baseline on held-out scores (composer-disjoint 5-fold CV, 95% paired "
                 f"bootstrap CI){title_note}\nBlue = baseline + features, orange = features without the baseline",
                 x=0.01, ha="left", fontsize=8.5, color=INK2)
    fig.tight_layout()
    return save(fig, name)


def figure(c):
    import matplotlib.pyplot as plt
    from analyze import AXIS, BLUE, GRID, INK2, MUTED, ORANGE, SURFACE, save
    c = c[c["in_figure"]]
    order = list(dict.fromkeys(c["group"]))
    c = pd.concat([c[c["group"] == g].sort_values("rho") for g in order[::-1]])
    fig, ax = plt.subplots(figsize=(9, 0.21 * len(c) + 1.5))
    y = np.arange(len(c))
    ax.hlines(y, c["rho_lo"], c["rho_hi"], color=BLUE, linewidth=2)
    ax.plot(c["rho"], y, "o", color=BLUE, markersize=5.5, markeredgecolor=SURFACE, label="Spearman ρ (95% CI)")
    ax.plot(c["partial_all"], y, "D", color=ORANGE, markersize=4.5, markeredgecolor=SURFACE,
            label="Partial ρ, controls held")
    for yi, r in zip(y, c.itertuples()):
        ax.text(1.0, yi, f"{r.rho:+.2f}  {r.partial_all:+.2f}", va="center", fontsize=7, color=INK2)
    ax.text(1.0, len(c) - 0.2, "ρ      partial", fontsize=7, color=MUTED, fontweight="bold")
    edges = np.cumsum([0] + [int((c["group"] == g).sum()) for g in order[::-1]])
    for e, g in zip(edges[1:], order[::-1]):
        ax.axhline(e - 0.5, color=GRID, linewidth=1.5)
        ax.text(-0.98, e - 1, g, fontsize=8, color=MUTED, fontweight="bold", va="center")
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y, c["label"])
    ax.set_xlim(-1, 1.22)
    ax.set_ylim(-0.7, len(c) + 0.3)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Correlation with grade (all scores with two hands)")
    ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("Pitch span, black keys, displacement and crossing against grade", pad=22)
    fig.tight_layout()
    return save(fig, "hf_correlations.png")


def corr_tables(c):
    """Markdown tables of a correlation frame, one per group."""
    from compare import md
    t = c.copy()
    t["ρ [95% CI]"] = [f"{r:+.2f} [{lo:+.2f}, {hi:+.2f}]" for r, lo, hi in zip(t.rho, t.rho_lo, t.rho_hi)]
    for col, name in (("tau_c", "τc"), ("partial_notes", "partial ρ | notes"),
                      ("partial_all", "partial ρ | all controls"), ("rho_reliable_tempo", "ρ, printed tempo")):
        t[name] = t[col].map(lambda v: f"{v:+.2f}")
    t["feature"] = t["feature"] + np.where(t["raw_count"], " (count)", "")
    out = []
    for g, d in t.groupby("group", sort=False):
        out.append(f"### {g}\n\n" + md(d[["feature", "N", "ρ [95% CI]", "τc", "partial ρ | notes",
                                         "partial ρ | all controls", "ρ, printed tempo"]], index=False))
    return "\n\n".join(out)


def validation_table(v, split="composer"):
    from compare import md
    d = v[v["split"] == split]
    rows = []
    for r in d.to_dict("records"):
        row = {"Model": r["model"], "Features": r["features"]}
        for k in ("QWK", "Bal. acc9", "Acc ±1", "MAE"):
            row[k] = f"{r[k]:.3f} [{r[f'{k}_lo']:.3f}, {r[f'{k}_hi']:.3f}]"
        for k in ("QWK", "MAE"):
            row[f"Δ{k} vs baseline"] = "–" if r["model"] == BASELINE else \
                f"{r[f'd_{k}']:+.3f} [{r[f'd_{k}_lo']:+.3f}, {r[f'd_{k}_hi']:+.3f}]"
        rows.append(row)
    return md(pd.DataFrame(rows), index=False)


def report(m, c, v, per_measure):
    conf = m["hand_confidence"].value_counts()
    top = c[~c["raw_count"]].reindex(c["partial_all"].abs().sort_values(ascending=False).index).dropna(
        subset=["rho"]).head(8)
    lines = "\n".join(f"- {r.label}: ρ {r.rho:+.2f}, partial {r.partial_all:+.2f}" for r in top.itertuples())
    comp, rand = v[v["split"] == "composer"].set_index("model"), v[v["split"] == "random"].set_index("model")
    best = comp.drop(index=BASELINE)["d_QWK"].idxmax()
    return f"""# Pitch span, black keys, displacement and crossing against grade

Generated by `nina_stats/hand_features.py`: groups B–E of `FEATURES_SPEC.md` (group A is in `COORDINATION.md`).
{len(m)} scores;
hands were assigned from staves for {conf.get("staff", 0)}, heuristically for {conf.get("heuristic", 0)} (more than
two staves, a near-empty staff, or a top staff lower than the bottom one) and not at all for {conf.get("none", 0)}
(a single staff with notes; their hand features are missing, not zero).

## Does any group add predictive value? (FEATURES_SPEC.md §11)

Ridge regression on quantile-normalised features, rounded to a grade, 5-fold CV with **composer-disjoint** folds
(no composer in both training and test). Baseline = note count, notes per second, stored tempo, written duration
and mean polyphony. Raw counts are left out of the models. 95% CIs bootstrap the held-out scores; Δ is paired
(same resamples for both models). QWK = quadratic weighted κ on grades 1–9.

Baseline: QWK {comp.loc[BASELINE, "QWK"]:.3f}, MAE {comp.loc[BASELINE, "MAE"]:.3f}. Best addition: {best}, ΔQWK
{comp.loc[best, "d_QWK"]:+.3f} [{comp.loc[best, "d_QWK_lo"]:+.3f}, {comp.loc[best, "d_QWK_hi"]:+.3f}]. All groups
together: ΔQWK {comp.loc["Baseline + all groups", "d_QWK"]:+.3f} [{comp.loc["Baseline + all groups", "d_QWK_lo"]:+.3f},
{comp.loc["Baseline + all groups", "d_QWK_hi"]:+.3f}], ΔMAE {comp.loc["Baseline + all groups", "d_MAE"]:+.3f}
[{comp.loc["Baseline + all groups", "d_MAE_lo"]:+.3f}, {comp.loc["Baseline + all groups", "d_MAE_hi"]:+.3f}].
With random (not composer-disjoint) folds the baseline gets QWK {rand.loc[BASELINE, "QWK"]:.3f} and baseline + all
groups {rand.loc["Baseline + all groups", "QWK"]:.3f}.

{validation_figure(v, "hf_validation.png")}

Composer-disjoint folds:

{validation_table(v, "composer")}

Random folds (for comparison; composers leak between training and test):

{validation_table(v, "random")}

## Correlation with grade (FEATURES_SPEC.md §11)

**Partial ρ** ranks the feature and the grade, regresses both on the ranks of the controls and correlates the
residuals: "| notes" holds the note count, "| all controls" holds the five baseline controls. **ρ, printed tempo**
repeats ρ on the {int(m["tempo_reliable"].sum())} scores with a printed metronome mark, for which per-second rates rest
on a reliable tempo; per-measure and per-quarter versions are given for the rest. CIs are 95% bootstrap
over scores.

Strongest beyond the controls (by |partial ρ, all controls|):

{lines}

{figure(c)}

{corr_tables(c)}

## Per-measure table

`results/hand_features_per_measure.csv`: {len(per_measure)} rows (one per measure of the {per_measure["file_name"].nunique()}
scores with two hands): {", ".join(f"`{c}`" for c in MEASURE_COLS)}, plus the measure's start, length in quarters and
seconds at the stored tempo. A transition between attacks belongs to the measure of the second attack. The sums over
measures equal the piece totals (checked on every run).

## Definitions

- Notes are attacks only (no grace notes, no tied continuations). Onsets are exact fractions of a quarter note
  (tuplets kept exact) and grouped by equality.
- **Simultaneous span**: highest − lowest pitch struck at one onset, both hands together (not a hand stretch).
  **Hand span**: the same within one hand. Shares are over onset groups; "per measure" counts groups.
- **Black key**: sounding pitch class in {{1, 3, 6, 8, 10}}. Note ratio counts notes; attack ratio counts a
  hand's onset groups with at least one black key.
- **Displacement**: |change| of the median pitch between a hand's successive onset groups (centroid: mean
  pitch), pooled over both hands and per hand. Per second uses the written duration at the stored tempo.
- **Crossing**: an onset where the LH median pitch is above the RH median; **overlap**: LH top note above RH
  bottom note. Both are proxies.
- Every distribution feature has mean, median, std, IQR, p95 and max; raw counts are separate columns marked
  "(count)".
"""


def load_with(feats):
    """All 806 scores with labels, controls and the given per-score features."""
    from analyze import load_all
    m = load_all().merge(feats.drop(columns=["measures", "quarters", "seconds"], errors="ignore"), on="file_name", validate="1:1")
    m["tempo_reliable"] = m["metronome_marks"] > 0
    return m


def main():
    files = sorted((f for f in os.listdir(SCORES) if f.endswith(".musicxml")),
                   key=lambda f: int(f.split("_")[1].split(".")[0]))
    with Pool(int(os.environ.get("WORKERS", os.cpu_count() or 4))) as pool:
        out = pool.map(features, files, chunksize=4)
    feats = pd.DataFrame([r for r, _ in out])
    per_measure = pd.DataFrame([row for _, rows in out for row in rows])
    sums = per_measure.groupby("file_name")[["black_key_notes", "crossings", "displacement_sum"]].sum()
    f = feats.set_index("file_name").loc[sums.index]
    assert np.allclose(sums["black_key_notes"], f["black_key_notes"])
    assert np.allclose(sums["crossings"], f["hand_crossing_count"])
    assert np.allclose(sums["displacement_sum"], f["hand_displacement_per_quarter"] * f["quarters"])
    feats.to_csv(os.path.join(OUT, "hand_features.csv"), index=False)
    per_measure.to_csv(os.path.join(OUT, "hand_features_per_measure.csv"), index=False)

    m = load_with(feats)
    c = correlate(m, [col for col in feats.columns if col not in NOT_FEATURES])
    c.to_csv(os.path.join(OUT, "hand_correlations.csv"), index=False)
    v = validate(m, c)
    v.to_csv(os.path.join(OUT, "hand_validation.csv"), index=False)
    with open(os.path.join(OUT, "HAND_FEATURES.md"), "w") as fh:
        fh.write(report(m, c, v, per_measure))
    print(f"{len(feats)} scores, {len(c)} features; hand confidence {feats['hand_confidence'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
