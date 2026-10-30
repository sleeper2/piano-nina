"""Feature group A of FEATURES_SPEC.md: written bimanual coordination, one row per score and one per measure,
its correlation with the annotator's grade and the held-out comparison with the baseline (section 11).

Everything comes from the MusicXML: onsets and durations are exact fractions of a quarter note (extract.parse with
exact=True), so tuplets keep their exact positions and subdivision grids are read from exact denominators, never
from rounded floats. Hands, as in hand_features.py, come from staves (top = right hand, bottom = left hand), with
a pitch heuristic for extra staves; the 2 single-staff scores have no features (missing, not zero).

Onset counts use attacks only (rests, grace notes and tied continuations are not onsets). Rests still count as
hand activity: a hand is "sounding" while a written note (including tied continuations) lasts and resting
otherwise, which gives the simultaneous-sounding ratio, each hand's rest share and the single-hand measures.

Outputs (results/): coordination.csv (one row per score), coordination_per_measure.csv, coordination_correlations.csv,
coordination_validation.csv, COORDINATION.md and figures/coord_*.png.
"""
import math
import os
from bisect import bisect_right
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
import pandas as pd

from extract import SCORES, parse
from hand_features import (BASELINE, CONTROLS, OUT, assign_hands, correlate, evaluate, load_with, stats,
                           validation_figure, validation_table)

# per-measure thresholds behind the "share of measures" features
THRESHOLDS = {"alternations": (">=", 4), "one_hand_share": (">=", 0.5), "similarity": ("<", 0.5),
              "density_ratio": (">=", 3), "both_sounding_share": ("<", 0.25)}


def merge(intervals):
    """Union of [start, end) intervals."""
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def intersect(x, y):
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if a < b:
            out.append([a, b])
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def per_measure_length(intervals, starts, ends):
    """Length of the intervals falling in each measure."""
    out = [Fraction(0)] * len(starts)
    for a, b in intervals:
        k = max(bisect_right(starts, a) - 1, 0)
        while k < len(starts) and starts[k] < b:
            out[k] += max(Fraction(0), min(b, ends[k]) - max(a, starts[k]))
            k += 1
    return out


def grid(positions):
    """Smallest subdivision of the quarter note (1 = quarter, 2 = eighth, 3 = triplet eighth, 4 = 16th, ...) on
    which all positions fall: the lcm of their exact denominators."""
    return math.lcm(*[p.denominator for p in positions]) if positions else None


def near(a, b, tol):
    """For each onset in a, whether b has an onset within tol quarter notes."""
    if not len(a) or not len(b):
        return np.zeros(len(a), bool)
    idx = np.searchsorted(b, a)
    left, right = b[np.clip(idx - 1, 0, len(b) - 1)], b[np.clip(idx, 0, len(b) - 1)]
    return np.minimum(np.abs(a - left), np.abs(a - right)) <= tol + 1e-9


def longest_alternation(labels):
    """Longest chain of single-hand onset groups that switch hand at every step (counted in groups)."""
    best = cur = 0
    prev = None
    for lab in labels:
        if lab == "B":
            cur, prev = 0, None
        elif prev is not None and lab != prev:
            cur += 1
        else:
            cur = 1
        prev = lab if lab != "B" else None
        best = max(best, cur)
    return best


def features(file_name):
    parsed = parse(os.path.join(SCORES, file_name), exact=True)
    notes, measure_len, n_measures = parsed[1], parsed[5], parsed[9]
    lengths = [measure_len[i] for i in range(len(measure_len))]
    starts = [sum(lengths[:i], Fraction(0)) for i in range(len(lengths))]
    ends = [s + n for s, n in zip(starts, lengths)]
    quarters = sum(lengths, Fraction(0))
    row = {"file_name": file_name, "measures": n_measures, "quarters": float(quarters)}
    hands, confidence = assign_hands(notes)
    row["hand_confidence"] = confidence
    if hands is None:
        return row, []
    measure_of = lambda t: bisect_right(starts, t) - 1  # noqa: E731

    onsets = {"R": set(), "L": set()}
    sounding = {"R": [], "L": []}
    for p, s, _v, onset, dur, _midi, attack in notes:
        h = hands[(p, s)]
        sounding[h].append((onset, onset + dur))
        if attack:
            onsets[h].add(onset)
    R, L = onsets["R"], onsets["L"]
    union = sorted(R | L)
    both = R & L
    labels = ["B" if t in both else ("R" if t in R else "L") for t in union]

    # per-measure rows
    n = len(starts)
    rm, lm = [[] for _ in range(n)], [[] for _ in range(n)]
    for t in R:
        rm[measure_of(t)].append(t)
    for t in L:
        lm[measure_of(t)].append(t)
    alternations = [0] * n
    for (a, b), t in zip(zip(labels, labels[1:]), union[1:]):
        if a != b and "B" not in (a, b):
            alternations[measure_of(t)] += 1
    s_union = {h: merge(sounding[h]) for h in "RL"}
    both_len = per_measure_length(intersect(s_union["R"], s_union["L"]), starts, ends)
    r_len = per_measure_length(s_union["R"], starts, ends)
    l_len = per_measure_length(s_union["L"], starts, ends)

    measures = []
    for i in range(n):
        r_set, l_set = set(rm[i]), set(lm[i])
        nr, nl, nb, nu = len(r_set), len(l_set), len(r_set & l_set), len(r_set | l_set)
        g_r = grid([t - starts[i] for t in r_set])
        g_l = grid([t - starts[i] for t in l_set])
        ln = lengths[i]
        measures.append({
            "file_name": file_name, "measure": i + 1, "start_quarter": float(starts[i]), "quarters": float(ln),
            "rh_onsets": nr, "lh_onsets": nl, "simultaneous_groups": nb, "nonsimultaneous_groups": nu - nb,
            "alternations": alternations[i],
            "one_hand_share": (nu - nb) / nu if nu else np.nan,
            "both_share": nb / nu if nu else np.nan,
            "similarity": nb / math.sqrt(nr * nl) if nr and nl else np.nan,
            "density_ratio": max(nr, nl) / min(nr, nl) if nr and nl else np.nan,
            "single_hand": (nr > 0) != (nl > 0),
            "rh_grid": g_r, "lh_grid": g_l,
            "grids_differ": (g_r != g_l) if g_r and g_l else np.nan,
            "cross_rhythm": (g_r % g_l != 0 and g_l % g_r != 0) if g_r and g_l else np.nan,
            "subdivision_gap": abs(math.log2(g_r / g_l)) if g_r and g_l else np.nan,
            "both_sounding_share": float(both_len[i] / ln) if ln else np.nan,
            "rh_rest_share": float(1 - r_len[i] / ln) if ln else np.nan,
            "lh_rest_share": float(1 - l_len[i] / ln) if ln else np.nan})
    pm = pd.DataFrame(measures)
    pm["dissimilarity"] = 1 - pm["similarity"]

    # 1 cross-hand onset alignment
    row["cross_hand_simultaneous_onset_ratio"] = len(both) / len(union)
    row["cross_hand_simultaneous_onset_ratio_lh"] = len(both) / len(L) if L else np.nan
    row["cross_hand_simultaneous_onset_ratio_rh"] = len(both) / len(R) if R else np.nan
    r_on, l_on = np.array(sorted(R), dtype=float), np.array(sorted(L), dtype=float)
    for tol, name in ((0, "0"), (0.05, "005"), (0.10, "010")):  # one hand's onset groups near the other's
        hits = near(r_on, l_on, tol).sum() + near(l_on, r_on, tol).sum()
        row[f"cross_hand_onset_alignment_tol{name}"] = hits / (len(r_on) + len(l_on))
    row.update(stats(pm["simultaneous_groups"], "simultaneous_onset_groups_per_measure"))
    row.update(stats(pm["nonsimultaneous_groups"], "nonsimultaneous_onset_groups_per_measure"))

    # 2 alternating hand attacks
    alt = sum(alternations)
    row["cross_hand_alternations"] = alt
    row["cross_hand_alternation_rate_per_measure"] = alt / n_measures
    row["cross_hand_alternation_rate_per_quarter"] = alt / float(quarters)
    row["cross_hand_alternation_switch_share"] = alt / (len(union) - 1) if len(union) > 1 else np.nan
    row["cross_hand_alternation_longest_run"] = longest_alternation(labels)
    row.update(stats(pm["alternations"], "alternations_per_measure"))

    # 3 rhythmic independence
    for col in ("one_hand_share", "both_share", "similarity", "dissimilarity"):
        row.update(stats(pm[col].dropna(), f"cross_hand_rhythm_independence_{col}"))
    row["onset_density_diff"] = (len(R) - len(L)) / (len(R) + len(L))
    row["onset_density_diff_abs"] = abs(row["onset_density_diff"])
    row["rh_distinct_onsets"], row["lh_distinct_onsets"] = len(R), len(L)
    for h, name in (("R", "rh"), ("L", "lh")):
        ioi = np.diff(np.array(sorted(onsets[h]), dtype=float))
        row.update({k: v for k, v in stats(ioi, f"{name}_ioi").items() if not k.endswith("_iqr")})
        row[f"{name}_ioi_cv"] = ioi.std() / ioi.mean() if len(ioi) and ioi.mean() else np.nan

    # 4 subdivision mismatch (measures where both hands have onsets)
    active = pm.dropna(subset=["grids_differ"])
    row["cross_hand_subdivision_mismatch_grids_differ_share"] = active["grids_differ"].mean() if len(active) else np.nan
    row["cross_hand_subdivision_mismatch_cross_rhythm_share"] = active["cross_rhythm"].mean() if len(active) else np.nan
    row.update(stats(active["subdivision_gap"], "cross_hand_subdivision_mismatch_gap"))
    for name in ("rh", "lh"):
        g = pm[f"{name}_grid"].dropna()
        row[f"{name}_grid_median"] = g.median() if len(g) else np.nan
        row[f"{name}_grid_max"] = g.max() if len(g) else np.nan

    # 5 onset density and overlap
    row["right_hand_onsets_per_measure"] = len(R) / n_measures
    row["left_hand_onsets_per_measure"] = len(L) / n_measures
    row["cross_hand_onset_density_ratio"] = max(len(R), len(L)) / min(len(R), len(L)) if R and L else np.nan
    row.update(stats(pm["density_ratio"].dropna(), "density_ratio_per_measure"))
    row["single_hand_measure_share"] = pm["single_hand"].mean()
    total = float(quarters)
    row["cross_hand_simultaneous_activity_ratio"] = float(sum(both_len)) / total if total else np.nan
    row["rh_rest_share"] = 1 - float(sum(r_len)) / total if total else np.nan
    row["lh_rest_share"] = 1 - float(sum(l_len)) / total if total else np.nan
    row.update(stats(pm["both_sounding_share"].dropna(), "both_sounding_share_per_measure"))

    # 6 share of measures beyond thresholds
    for col, (op, t) in THRESHOLDS.items():
        v = pm[col].dropna()
        hit = v >= t if op == ">=" else v < t
        row[f"share_measures_{col}_{'ge' if op == '>=' else 'lt'}{t}"] = hit.mean() if len(v) else np.nan
    return row, measures


# ---------------------------------------------------------------- reporting

SECTIONS = [  # (group, column prefixes)
    ("1 Onset alignment", ("cross_hand_simultaneous_onset_ratio", "cross_hand_onset_alignment",
                           "simultaneous_onset_groups",
                           "nonsimultaneous_onset_groups")),
    ("2 Alternation", ("cross_hand_alternation", "alternations_per_measure", "share_measures_alternations")),
    ("3 Rhythmic independence", ("cross_hand_rhythm_independence", "onset_density_diff", "rh_distinct",
                                 "lh_distinct", "rh_ioi", "lh_ioi", "share_measures_one_hand",
                                 "share_measures_similarity")),
    ("4 Subdivision mismatch", ("cross_hand_subdivision_mismatch", "rh_grid", "lh_grid")),
    ("5 Density and overlap", ("right_hand_onsets", "left_hand_onsets", "cross_hand_onset_density_ratio",
                               "density_ratio_per_measure", "single_hand_measure", "cross_hand_simultaneous_activity",
                               "rh_rest", "lh_rest", "both_sounding", "share_measures_density",
                               "share_measures_both_sounding")),
]
# reported in the CSV, not correlated: exact complements of another column (ρ only flips sign) and raw counts
COMPLEMENTS = ("cross_hand_rhythm_independence_both_share", "cross_hand_rhythm_independence_dissimilarity")
COUNTS = {"cross_hand_alternations", "rh_distinct_onsets", "lh_distinct_onsets"}
HEADLINE = [  # one per named metric of the spec, each added to the baseline on its own
    "cross_hand_simultaneous_onset_ratio", "cross_hand_simultaneous_onset_ratio_lh",
    "cross_hand_simultaneous_onset_ratio_rh", "cross_hand_onset_alignment_tol010",
    "simultaneous_onset_groups_per_measure_mean",
    "nonsimultaneous_onset_groups_per_measure_mean", "cross_hand_alternation_rate_per_measure",
    "cross_hand_alternation_rate_per_quarter", "cross_hand_alternation_switch_share",
    "cross_hand_alternation_longest_run", "cross_hand_rhythm_independence_one_hand_share_mean",
    "cross_hand_rhythm_independence_similarity_mean", "onset_density_diff_abs", "rh_ioi_cv", "lh_ioi_cv",
    "rh_ioi_median", "lh_ioi_median", "cross_hand_subdivision_mismatch_grids_differ_share",
    "cross_hand_subdivision_mismatch_cross_rhythm_share", "cross_hand_subdivision_mismatch_gap_mean",
    "right_hand_onsets_per_measure", "left_hand_onsets_per_measure", "cross_hand_onset_density_ratio",
    "single_hand_measure_share", "cross_hand_simultaneous_activity_ratio"]


def section_of(col):
    for name, prefixes in SECTIONS:
        if col.startswith(prefixes):
            return name
    raise ValueError(col)


def corr_figure(c):
    import matplotlib.pyplot as plt
    from analyze import AXIS, BLUE, GRID, INK2, MUTED, ORANGE, SURFACE, save
    d = c[c["feature"].isin(HEADLINE)].copy()
    order = [s for s, _ in SECTIONS]
    d = pd.concat([d[d["group"] == g].sort_values("rho") for g in order[::-1]])
    fig, ax = plt.subplots(figsize=(9, 0.27 * len(d) + 1.5))
    y = np.arange(len(d))
    ax.hlines(y, d["rho_lo"], d["rho_hi"], color=BLUE, linewidth=2)
    ax.plot(d["rho"], y, "o", color=BLUE, markersize=5.5, markeredgecolor=SURFACE, label="Spearman ρ (95% CI)")
    ax.plot(d["partial_all"], y, "D", color=ORANGE, markersize=4.5, markeredgecolor=SURFACE,
            label="Partial ρ, controls held")
    for yi, r in zip(y, d.itertuples()):
        ax.text(1.0, yi, f"{r.rho:+.2f}  {r.partial_all:+.2f}", va="center", fontsize=7, color=INK2)
    ax.text(1.0, len(d) - 0.3, "ρ      partial", fontsize=7, color=MUTED, fontweight="bold")
    edges = np.cumsum([0] + [int((d["group"] == g).sum()) for g in order[::-1]])
    for e, g in zip(edges[1:], order[::-1]):
        ax.axhline(e - 0.5, color=GRID, linewidth=1.5)
        ax.text(-0.98, e - 1, g, fontsize=8, color=MUTED, fontweight="bold", va="center")
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y, d["feature"], fontsize=7.5)
    ax.set_xlim(-1, 1.22)
    ax.set_ylim(-0.7, len(d) + 0.3)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Correlation with grade (scores with two hands)")
    ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("Written bimanual coordination against grade", pad=22)
    fig.tight_layout()
    return save(fig, "coord_correlations.png")


def report(m, c, v, pm):
    from hand_features import corr_tables
    conf = m["hand_confidence"].value_counts()
    comp = v[v["split"] == "composer"].set_index("model")
    single = comp[comp.index.str.startswith("Baseline + ") & (comp.index != "Baseline + coordination group")]
    sig = single[single["d_QWK_lo"] > 0].sort_values("d_QWK", ascending=False)
    grp = comp.loc["Baseline + coordination group"]
    sig_lines = "\n".join(f"- {name[11:]}: ΔQWK {r.d_QWK:+.3f} [{r.d_QWK_lo:+.3f}, {r.d_QWK_hi:+.3f}], ΔMAE "
                          f"{r.d_MAE:+.3f} [{r.d_MAE_lo:+.3f}, {r.d_MAE_hi:+.3f}]" for name, r in sig.iterrows()) \
        or "- none"
    cr = pm.dropna(subset=["cross_rhythm"])
    return f"""# Written bimanual coordination against grade

Generated by `nina_stats/coordination.py`: group A of `FEATURES_SPEC.md` (groups B–E are in `HAND_FEATURES.md`).
{len(m)} scores; hands from staves for {conf.get("staff", 0)}, heuristic for {conf.get("heuristic", 0)}, none for
{conf.get("none", 0)} (single staff; features missing). Per-measure table: `results/coordination_per_measure.csv`
({len(pm)} measures). Of the measures where both hands play, {cr["grids_differ"].mean():.0%} put the hands on
different subdivision grids and {cr["cross_rhythm"].mean():.1%} on incompatible ones (e.g. triplets against 16ths).

## Incremental predictive value (FEATURES_SPEC.md §11)

Same set-up as `HAND_FEATURES.md`: ridge regression on quantile-normalised features, rounded to a grade,
**composer-disjoint** 5-fold CV, baseline = note count, notes per second, stored tempo, written duration and mean
polyphony; 95% bootstrap CIs on held-out scores, Δ paired against the baseline. Each named metric is added to the
baseline on its own, then the whole group ({int(grp["features"]) - len(CONTROLS)} non-redundant features).

Baseline QWK {comp.loc[BASELINE, "QWK"]:.3f}. Whole coordination group: ΔQWK {grp["d_QWK"]:+.3f}
[{grp["d_QWK_lo"]:+.3f}, {grp["d_QWK_hi"]:+.3f}], ΔMAE {grp["d_MAE"]:+.3f} [{grp["d_MAE_lo"]:+.3f},
{grp["d_MAE_hi"]:+.3f}].

Single features whose ΔQWK interval excludes 0:

{sig_lines}

{validation_figure(v, "coord_validation.png")}

{validation_table(v, "composer")}

Random folds (composers leak between training and test), group rows only:

{validation_table(v[v["model"].isin([BASELINE, "Baseline + coordination group", "Coordination group only"])],
                  "random")}

## Correlation with grade

ρ and Kendall τc against the grade, partial ρ holding note count or all five controls (rank residuals), and ρ on
the scores with a printed metronome mark. Not correlated (but in the CSV): exact complements (`both_share` = 1 −
`one_hand_share`, `dissimilarity` = 1 − `similarity`; ρ only changes sign).

{corr_figure(c)}

{corr_tables(c)}

## Definitions

- **Onset group**: all attacks of one hand at one exact onset (a chord counts once). Rests, grace notes and tied
  continuations are not onsets.
- **Simultaneous onset ratio**: onsets where both hands attack / onsets where either does; also / LH onset groups
  and / RH onset groups. Per measure: simultaneous and non-simultaneous onset groups. **Onset alignment** (tolerance
  0, 0.05, 0.10 quarter notes): share of all onset groups of both hands that have an onset of the other hand
  within the tolerance.
- **Alternation**: successive onset groups (both hands merged, in time order) going left-only → right-only or
  back; simultaneous groups break the chain and are not counted. Switch share = alternations / (onset groups − 1);
  longest run = most onset groups in an unbroken alternating chain.
- **Rhythmic independence**, per measure: one-hand share = onset positions used by one hand only / positions used
  by either; both share = its complement. **Similarity** = cosine (Ochiai) similarity of the two hands' binary onset
  patterns on the measure's positions, |R ∩ L| / √(|R| · |L|), which is independent of measure length and time
  signature; **dissimilarity** = 1 − similarity. Defined only when both hands play in the measure. IOIs are the
  gaps between a hand's successive onset groups, in quarter notes (CV = std / mean).
- **Subdivision grid** of a hand in a measure: the smallest subdivision of the quarter note that holds all its onset
  positions, the lcm of their exact denominators (1 = quarter, 2 = eighth, 3 = triplet eighth, 4 = 16th, 6 = 16th
  triplet …). Grids differ when the two hands' grids are unequal; cross-rhythm when neither divides the other
  (2 vs 3, 3 vs 4); gap = |log2(grid R / grid L)|.
- **Density ratio** = higher / lower onset count of the two hands; undefined (missing) when one hand has no onsets,
  which is counted instead by the single-hand measure share. **Simultaneous activity ratio** = share of the written
  timeline during which both hands have a note sounding (notated durations, ties included); rest share = share
  during which that hand is silent.
- Measure-level features are summarised per piece by mean, median, std, IQR, p95 and max; "share_measures_…" is the
  share of measures beyond: {", ".join(f"{k} {op} {t}" for k, (op, t) in THRESHOLDS.items())}.
"""


def main():
    files = sorted((f for f in os.listdir(SCORES) if f.endswith(".musicxml")),
                   key=lambda f: int(f.split("_")[1].split(".")[0]))
    with Pool(int(os.environ.get("WORKERS", os.cpu_count() or 4))) as pool:
        out = pool.map(features, files, chunksize=4)
    feats = pd.DataFrame([r for r, _ in out])
    pm = pd.DataFrame([row for _, rows in out for row in rows])
    sums = pm.groupby("file_name")[["alternations", "rh_onsets", "lh_onsets"]].sum()
    f = feats.set_index("file_name").loc[sums.index]
    assert (sums["alternations"] == f["cross_hand_alternations"]).all()
    assert (sums["rh_onsets"] == f["rh_distinct_onsets"]).all() and (sums["lh_onsets"] == f["lh_distinct_onsets"]).all()
    feats.to_csv(os.path.join(OUT, "coordination.csv"), index=False)
    pm.to_csv(os.path.join(OUT, "coordination_per_measure.csv"), index=False)

    m = load_with(feats)
    cols = [col for col in feats.columns
            if col not in {"file_name", "measures", "quarters", "hand_confidence"}
            and not col.startswith(COMPLEMENTS)]
    c = correlate(m, cols, meta=lambda col: (section_of(col), col))
    c["raw_count"] = c["feature"].isin(COUNTS)
    c.to_csv(os.path.join(OUT, "coordination_correlations.csv"), index=False)

    group = c.loc[~c["raw_count"], "feature"].tolist()
    configs = {BASELINE: CONTROLS}
    configs.update({f"Baseline + {col}": CONTROLS + [col] for col in HEADLINE})
    configs["Baseline + coordination group"] = CONTROLS + group
    configs["Coordination group only"] = group
    v = pd.concat([evaluate(m, configs, splits=("composer",)),
                   evaluate(m, {k: configs[k] for k in (BASELINE, "Baseline + coordination group",
                                                         "Coordination group only")}, splits=("random",))])
    v.to_csv(os.path.join(OUT, "coordination_validation.csv"), index=False)
    with open(os.path.join(OUT, "COORDINATION.md"), "w") as fh:
        fh.write(report(m, c, v, pm))
    print(f"{len(feats)} scores, {len(c)} correlated features, {len(pm)} measures")


if __name__ == "__main__":
    main()
