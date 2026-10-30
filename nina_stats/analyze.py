"""Dataset statistics for the paper's sections 3.5 (repertoire), 3.6 (score content) and 3.7 (confounds).

Reads the dataset sheet, results/content.csv (from extract.py), labels/composers.csv and labels/scores.csv
(hand-checked classification of composers and scores), and the existing model results via
model_comparison/compare.py. Writes results/REPORT.md, results/per_grade.csv, results/correlations.csv and
the figures in results/figures/.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, kruskal, mannwhitneyu, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
FIG = OUT / "figures"
sys.path.insert(0, str(ROOT / "model_comparison"))
from compare import accuracies, binned_baseline, load, md, partial_spearman  # noqa: E402

# Reference palette from the dataviz skill (light surface).
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BUCKETS = {"Grades 1–3": "#9ec5f4", "Grades 4–6": "#3987e5", "Grades 7–9": "#184f95"}
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ERA_ORDER = ["baroque", "classical", "romantic", "early_20c", "late_20c", "21c", "unknown"]
ERA_NAMES = {"baroque": "Baroque (<1750)", "classical": "Classical (1750–1820)", "romantic": "Romantic (1820–1900)",
             "early_20c": "1900–1945", "late_20c": "1945–1999", "21c": "2000–", "unknown": "Unknown"}
BOOT = 1000

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
})


# ---------------------------------------------------------------- data

def load_all():
    m = load()  # 806 rows: label, model predictions, RubricNet descriptors, ioi_tempo_avg, score_has_tempo
    c = pd.read_csv(OUT / "content.csv")
    m = m.merge(c, on="file_name", validate="1:1")
    m["seconds_min"] = m["seconds"] / 60
    m["notes_per_sec_written"] = m["notes_count"] / m["seconds"]
    m["notes_per_measure"] = m["notes_count"] / m["measures_count"]
    m["key_accidentals"] = m["key_fifths_first"].abs()
    m["shortest_rank"] = m["shortest_type"].map({"quarter": 0, "eighth": 1, "16th": 2, "32nd": 3, "64th": 4,
                                                 "128th": 5})
    m["bucket"] = pd.cut(m["label"], [0, 3, 6, 9], labels=list(BUCKETS))
    comp, scores = HERE / "labels/composers.csv", HERE / "labels/scores.csv"
    if comp.exists() and scores.exists():
        m = m.merge(pd.read_csv(comp), on="composer", how="left", validate="m:1")
        m = m.merge(pd.read_csv(scores), on="id", how="left", validate="1:1")
        m["source"] = np.select(
            [m["simplified"].eq("yes"), m["medium"].eq("other"), m["medium"].eq("piano")],
            ["Simplified version", "Arrangement (non-piano original)", "Original piano work"], "Unknown")
    return m


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    return f"![{name}](figures/{name})"


def fmt(v, nd=2):
    return "–" if pd.isna(v) else f"{v:.{nd}f}"


def boot_ci(fn, *arrays, n=BOOT, seed=0):
    rng = np.random.default_rng(seed)
    size = len(arrays[0])
    vals = [fn(*(a[idx] for a in arrays)) for idx in (rng.integers(0, size, size) for _ in range(n))]
    return np.nanpercentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------- shared chart forms

def stacked_by_grade(ax, m, col, order, names=None, title=""):
    """Horizontal bars, one per category, length = scores, segments = grade buckets; mean grade at the end."""
    names = names or {}
    order = [o for o in order if (m[col] == o).any()]
    counts = pd.crosstab(m[col], m["bucket"]).reindex(order).fillna(0)
    means = m.groupby(col)["label"].mean().reindex(order)
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for b, colour in BUCKETS.items():
        ax.barh(y, counts[b], left=left, color=colour, height=0.62, edgecolor=SURFACE, linewidth=1.2, label=b)
        left += counts[b].values
    for yi, total, mean in zip(y, left, means):
        ax.text(total + left.max() * 0.012, yi, f"{int(total)}  ·  mean {mean:.1f}", va="center", color=INK2,
                fontsize=8)
    ax.set_yticks(y, [names.get(o, o) for o in order])
    ax.set_xlim(0, left.max() * 1.28)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Scores")
    ax.set_title(title)
    ax.tick_params(axis="y", length=0)


def bucket_legend(fig, y=1.0):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BUCKETS.values()]
    fig.legend(handles, list(BUCKETS), loc="upper right", ncol=3, bbox_to_anchor=(1, y), fontsize=8,
               handlelength=1, columnspacing=1.2)


def boxes_by_grade(ax, m, col, title, log=False, ylabel=""):
    data = [m.loc[m["label"] == g, col].dropna().values for g in range(1, 10)]
    ax.boxplot(data, positions=range(1, 10), widths=0.55, patch_artist=True, showfliers=True,
               boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE, linewidth=1),
               medianprops=dict(color="#104281", linewidth=2), whiskerprops=dict(color=BLUE, linewidth=1),
               capprops=dict(color=BLUE, linewidth=1),
               flierprops=dict(marker="o", markersize=2.5, markerfacecolor=MUTED, markeredgecolor="none",
                               alpha=0.6))
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(range(1, 10))


# ---------------------------------------------------------------- 3.5 repertoire composition

def composition_section(m):
    if "gender" not in m:
        return "_labels/composers.csv and labels/scores.csv not found: run the labelling step first._\n"
    out = []
    n = len(m)
    counts = m["composer"].value_counts()
    share = counts / n
    hhi = (share ** 2).sum()
    gini = gini_coef(counts.values)
    singletons = (counts == 1).sum()

    # concentration figure
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    top = counts.head(15)[::-1]
    a1.barh(range(len(top)), top.values, color=BLUE, height=0.62)
    for i, v in enumerate(top.values):
        a1.text(v + 1.5, i, f"{v}", va="center", fontsize=8, color=INK2)
    a1.set_yticks(range(len(top)), top.index)
    a1.grid(axis="y", visible=False)
    a1.tick_params(axis="y", length=0)
    a1.set_xlabel("Scores")
    a1.set_title("Top 15 composers by number of scores")
    cum = np.concatenate([[0], counts.cumsum().values / n * 100])
    xs = np.arange(len(cum)) / len(counts) * 100
    a2.plot(xs, cum, color=BLUE, linewidth=2)
    a2.plot([0, 100], [0, 100], color=AXIS, linewidth=1, linestyle="--")
    for k in (1, 10, 50):
        a2.plot(k / len(counts) * 100, cum[k], "o", color=BLUE, markersize=6, markeredgecolor=SURFACE)
        a2.annotate(f"top {k}: {cum[k]:.0f}% of scores", (k / len(counts) * 100, cum[k]), xytext=(8, -4),
                    textcoords="offset points", fontsize=8, color=INK2)
    a2.set_xlabel("% of composers (most-represented first)")
    a2.set_ylabel("% of scores")
    a2.set_title("Cumulative share of scores")
    a2.set_xlim(0, 100)
    a2.set_ylim(0, 102)
    a2.text(55, 18, f"{len(counts)} composers\n{singletons} with a single score\n"
                    f"Gini {gini:.2f}, effective number {1 / hhi:.0f}", fontsize=8, color=INK2)
    fig.tight_layout()
    out.append(f"""### Composers and concentration

{len(counts)} distinct composer strings (after the sheet's own name normalization) for {n} scores; {singletons}
({singletons / len(counts):.0%}) appear once. The top composer, {counts.index[0]}, has {counts.iloc[0]} scores
({share.iloc[0]:.1%}); the top 10 hold {share.head(10).sum():.1%} and the top 50 {share.head(50).sum():.1%}.
Herfindahl index {hhi:.3f} (effective number of equally represented composers {1 / hhi:.0f}); Gini {gini:.2f}.
Without {counts.index[0]} the HHI is {((counts.iloc[1:] / counts.iloc[1:].sum()) ** 2).sum():.4f}
(effective number {1 / ((counts.iloc[1:] / counts.iloc[1:].sum()) ** 2).sum():.0f}).

{save(fig, "35_composer_concentration.png")}

Composer × grade: every composer with at least 3 scores, sorted by mean grade; composers with fewer scores are
pooled in the bottom rows. Cells count scores (colour on a square-root-like scale so that small counts stay
visible next to {counts.index[0]}).

{composer_grade_heatmap(m, counts)}
""")

    # era / genre / form / source / gender: one stacked-bar figure each
    def dim(col, order, names, title, fname, height):
        fig, ax = plt.subplots(figsize=(8.5, height))
        stacked_by_grade(ax, m, col, order, names, title)
        bucket_legend(fig, 1.02)
        fig.tight_layout()
        return save(fig, fname)

    era = m["era"].fillna("unknown")
    m = m.assign(era=era)
    era_comp = m.drop_duplicates("composer")["era"].value_counts().reindex(ERA_ORDER).fillna(0).astype(int)
    era_tab = pd.DataFrame({"Scores": m["era"].value_counts().reindex(ERA_ORDER).fillna(0).astype(int),
                            "Composers": era_comp,
                            "Mean grade": m.groupby("era")["label"].mean().reindex(ERA_ORDER).round(2)})
    era_tab.index = [ERA_NAMES[e] for e in era_tab.index]
    out.append(f"""### Era

Era of the composer's main activity (for an arrangement, the original composer). Bars show the number of scores,
split by grade bucket; the label gives the mean grade.

{dim("era", ERA_ORDER, ERA_NAMES, "Scores by era", "35_era.png", 3.3)}

{md(era_tab.rename_axis("Era"))}
""")

    genre_order = m["genre"].value_counts().index.tolist()
    form_order = m["form"].value_counts().index.tolist()
    out.append(f"""### Genre and form

{dim("genre", genre_order, GENRE_NAMES, "Scores by genre", "35_genre.png", 0.32 * len(genre_order) + 1)}

{dim("form", form_order, FORM_NAMES, "Scores by form", "35_form.png", 0.32 * len(form_order) + 1)}

{md(pd.DataFrame({"Scores": m["genre"].value_counts(),
                  "Mean grade": m.groupby("genre")["label"].mean().round(2)}).rename(index=GENRE_NAMES)
    .rename_axis("Genre"))}
""")

    src_order = ["Original piano work", "Arrangement (non-piano original)", "Simplified version", "Unknown"]
    src = m["source"].value_counts().reindex(src_order).fillna(0).astype(int)
    k_y, k_n = m["source"].eq("Simplified version"), m["source"].eq("Original piano work")
    out.append(f"""### Original works, arrangements and simplified versions

A score counts as an **arrangement** when its music was originally written for other forces (song, orchestra,
opera, film or game soundtrack) and as a **simplified version** when it is an easier or reduced version of an
existing piece (marked easy/simplified/beginner, or clearly reduced; this includes easy arrangements of songs).
{src["Original piano work"]} scores ({src["Original piano work"] / n:.0%}) are original piano works,
{src["Arrangement (non-piano original)"]} ({src["Arrangement (non-piano original)"] / n:.0%}) arrangements,
{src["Simplified version"]} ({src["Simplified version"] / n:.0%}) simplified versions and {src["Unknown"]} unknown.
Simplification is the least certain label: {m["simplified"].eq("unknown").sum()} further scores (counted above by their
original medium) are short arrangements, mostly of pop, game and film music, that may or may not be reduced, so
{src["Simplified version"] / n:.0%} is a lower bound.
Simplified versions are rated lower (mean grade {m.loc[k_y, "label"].mean():.2f} vs.
{m.loc[k_n, "label"].mean():.2f} for originals).

{dim("source", src_order, None, "Scores by source type", "35_source.png", 2.2)}

{md(pd.crosstab(m["genre"], m["source"]).reindex(columns=[s for s in src_order if s in set(m["source"])])
    .reindex(genre_order).rename(index=GENRE_NAMES).rename_axis("Genre / source"))}
""")
    out.append(gender_part(m, dim))
    return "\n".join(out)


def composer_grade_heatmap(m, counts, min_scores=3):
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    named = counts[counts >= min_scores].index
    tab = pd.crosstab(m["composer"], m["label"]).reindex(columns=range(1, 10), fill_value=0)
    rows = tab.loc[named]
    rows = rows.loc[m[m["composer"].isin(named)].groupby("composer")["label"].mean()
                    .sort_values(kind="stable").index]
    pooled = {}
    for k, name in ((2, "2 scores"), (1, "1 score")):
        comp = counts[counts == k].index
        pooled[f"Composers with {name} ({len(comp)})"] = tab.loc[comp].sum()
    full = pd.concat([rows, pd.DataFrame(pooled).T])
    means = [m.loc[m["composer"] == c, "label"].mean() for c in rows.index] + \
        [m.loc[m["composer"].map(counts) == k, "label"].mean() for k in (2, 1)]

    fig, ax = plt.subplots(figsize=(8.5, 0.27 * len(full) + 1.4))
    data = full.values.astype(float)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    norm = PowerNorm(0.45, vmin=0, vmax=data.max())
    ax.imshow(np.where(data > 0, data, np.nan), cmap=cmap, norm=norm, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j]:
                ax.text(j, i, int(data[i, j]), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if norm(data[i, j]) > 0.55 else INK)
        ax.text(9.0, i, f"{int(data[i].sum())}", ha="left", va="center", fontsize=7.5, color=INK2)
        ax.text(10.0, i, f"{means[i]:.1f}", ha="left", va="center", fontsize=7.5, color=INK2)
    ax.text(9.0, -0.9, "Total", fontsize=7.5, color=MUTED, fontweight="bold")
    ax.text(10.0, -0.9, "Mean", fontsize=7.5, color=MUTED, fontweight="bold")
    ax.axhline(len(rows) - 0.5, color=INK2, linewidth=0.8)
    ax.set_xlim(-0.5, 10.9)
    ax.set_xticks(range(9), range(1, 10))
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(full)), full.index)
    ax.set_xlabel("Grade")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Composer")
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(full), 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    return save(fig, "35_composer_grade_heatmap.png")


GENRE_NAMES = {"classical": "Classical", "ragtime": "Ragtime", "jazz_blues": "Jazz / blues", "film_tv": "Film / TV",
               "video_game": "Video game", "anime_jpop": "Anime / J-, K-, C-pop", "pop_rock": "Pop / rock",
               "musical_theatre": "Musical theatre", "contemporary_piano": "Contemporary solo piano",
               "folk_traditional": "Folk / traditional", "religious": "Religious",
               "anthem_political": "Anthem / political", "children_educational": "Children's / educational",
               "unknown": "Unknown"}
FORM_NAMES = {"etude_exercise": "Étude / exercise", "sonata_sonatina": "Sonata / sonatina", "prelude": "Prelude",
              "invention_fugue_canon": "Invention / fugue / canon", "dance": "Dance", "march": "March",
              "rag": "Rag / novelty", "character_piece": "Character piece", "variations_fantasy":
                  "Variations / fantasy / medley", "song_cue": "Song / theme / cue", "other": "Other",
              "unknown": "Unknown"}


def gender_part(m, dim):
    g = m["gender"].fillna("unknown")
    m = m.assign(gender=g)
    order = ["male", "female", "mixed", "unknown"]
    names = {"male": "Male", "female": "Female", "mixed": "Mixed team", "unknown": "Unknown / undocumented"}
    comp = m.drop_duplicates("composer")["gender"].value_counts().reindex(order).fillna(0).astype(int)
    sc = m["gender"].value_counts().reindex(order).fillna(0).astype(int)
    f, ml = m[m["gender"] == "female"], m[m["gender"] == "male"]
    p_label = mannwhitneyu(f["label"], ml["label"]).pvalue if len(f) and len(ml) else np.nan

    # fairness: signed error of each model by gender, with bootstrap CI
    rows = []
    for model, col, sub in (("CIPI ensemble (806)", "pred_cipi", m),
                            ("RubricNet (560)", "pred_rubricnet", m[m["score_has_tempo"]])):
        for gen in ("male", "female"):
            d = sub[sub["gender"] == gen]
            err = (d[col] - d["label"]).values.astype(float)
            lo, hi = boot_ci(np.mean, err)
            rows.append({"Model": model, "Gender": gen, "N": len(d), "Mean signed error": err.mean(),
                         "lo": lo, "hi": hi, "MAE": np.abs(err).mean(), "Exact": (err == 0).mean()})
    fair = pd.DataFrame(rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    stacked_by_grade(a1, m, "gender", order, names, "Scores by composer gender")
    ys = {("CIPI ensemble (806)", "male"): 3, ("CIPI ensemble (806)", "female"): 2.5,
          ("RubricNet (560)", "male"): 1, ("RubricNet (560)", "female"): 0.5}
    for r in fair.to_dict("records"):
        y, colour = ys[(r["Model"], r["Gender"])], BLUE if r["Gender"] == "male" else ORANGE
        a2.plot([r["lo"], r["hi"]], [y, y], color=colour, linewidth=2)
        a2.plot(r["Mean signed error"], y, "o", color=colour, markersize=7, markeredgecolor=SURFACE)
        a2.text(r["hi"] + 0.05, y, f"{r['Gender']} (n={r['N']})", va="center", fontsize=8, color=INK2)
    a2.axvline(0, color=AXIS, linewidth=1)
    a2.set_yticks([2.75, 0.75], ["CIPI ensemble\n(806)", "RubricNet\n(560)"])
    a2.set_ylim(0, 3.5)
    a2.grid(axis="y", visible=False)
    a2.tick_params(axis="y", length=0)
    a2.set_xlabel("Mean signed error, predicted − label (95% CI)")
    a2.set_title("Model bias by composer gender")
    lo_x = min(fair["lo"].min(), -0.5)
    a2.set_xlim(lo_x - 0.1, fair["hi"].max() + 0.9)
    bucket_legend(fig, 1.08)
    fig.tight_layout()
    fname = save(fig, "35_gender.png")

    fair_t = fair.assign(**{"Mean signed error": [f"{e:+.2f} [{lo:+.2f}, {hi:+.2f}]" for e, lo, hi in
                                                  zip(fair["Mean signed error"], fair["lo"], fair["hi"])]},
                         MAE=fair["MAE"].round(2), Exact=fair["Exact"].round(3)).drop(columns=["lo", "hi"])
    top = m["composer"].value_counts().index[0]
    nt = m[m["composer"] != top]
    sc_nt = nt["gender"].value_counts().reindex(order).fillna(0).astype(int)
    # the female group sits mostly in ragtime and pop, so compare within genre too
    fm = m[m["gender"].isin(["male", "female"])].assign(err=lambda d: d["pred_cipi"] - d["label"])
    gen = fm.groupby(["genre", "gender"]).agg(n=("label", "size"), grade=("label", "mean"), err=("err", "mean"))
    gen = gen.unstack("gender")
    within = pd.DataFrame({
        "N female": gen[("n", "female")], "N male": gen[("n", "male")].fillna(0),
        "Mean grade female": gen[("grade", "female")].round(2), "Mean grade male": gen[("grade", "male")].round(2),
        "CIPI error female": gen[("err", "female")].round(2), "CIPI error male": gen[("err", "male")].round(2)})
    within = within[within["N female"] >= 3].astype({"N female": int, "N male": int})
    within.index = [GENRE_NAMES.get(i, i) for i in within.index]
    grade_g = pd.crosstab(m["gender"], m["label"]).reindex(order).fillna(0).astype(int)
    grade_g.index = [names[i] for i in grade_g.index]
    return f"""### Composer gender (fairness check)

Gender is filled only where it is documented for the (original) composer; names alone were not used, so the
{comp["unknown"]} undocumented composers (mostly MuseScore users and obscure names) stay unknown. Composers:
{comp["male"]} male, {comp["female"]} female, {comp["mixed"]} mixed teams, {comp["unknown"]} unknown.
Scores: {sc["male"]} male ({sc["male"] / len(m):.0%}), {sc["female"]} female ({sc["female"] / len(m):.1%}),
{sc["mixed"]} mixed, {sc["unknown"]} unknown.
Mean grade: male {ml["label"].mean():.2f}, female {f["label"].mean():.2f} (Mann–Whitney p = {p_label:.3f}).
{top} alone accounts for {(m["composer"] == top).sum()} of the {sc["male"]} male-composer scores (this composer's gender is
not source-checked; see `labels/composers.csv`). Without them: {sc_nt["male"]} male ({sc_nt["male"] / len(nt):.0%}),
{sc_nt["female"]} female ({sc_nt["female"] / len(nt):.1%}), {sc_nt["unknown"]} unknown, mean grade male
{nt.loc[nt["gender"] == "male", "label"].mean():.2f}.

{fname}

Grade distribution by gender:

{md(grade_g.rename_axis("Gender / grade"))}

Model error by gender (negative = the model underrates):

{md(fair_t, index=False)}

Within genre (genres with at least 3 female-composer scores; CIPI ensemble error = predicted − label):

{md(within.rename_axis("Genre"))}

The female group is small, so the intervals are wide, and it is concentrated in ragtime and pop; read
differences within genre rather than overall.
"""


def gini_coef(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


# ---------------------------------------------------------------- 3.6 score content

CONTENT = [  # column, label, log scale, unit
    ("notes_count", "Notes", True, ""),
    ("measures_count", "Measures", True, ""),
    ("seconds", "Duration at stored tempo (s)", True, ""),
    ("ioi_tempo_avg", "IOI tempo (onsets/min, RubricNet)", True, ""),
    ("onsets_per_min", "Onsets per minute (stored tempo)", True, ""),
    ("notes_per_sec_written", "Notes per second", True, ""),
    ("pitch_range", "Pitch range (semitones)", False, ""),
    ("pitch_entropy", "Pitch entropy (bits)", False, ""),
    ("polyphony_mean", "Mean polyphony (sounding notes)", False, ""),
    ("voices_max", "Max voices on one staff", False, ""),
    ("chord_span_p95", "Chord span, 95th pct (semitones)", False, ""),
    ("shortest_dur_q", "Shortest note (quarters)", True, ""),
]
MARKINGS = [("dynamics", "Dynamics"), ("hairpins", "Hairpins"), ("articulations", "Articulations"),
            ("slurs", "Slurs"), ("pedal", "Pedal"), ("fingering", "Fingering"), ("ornaments", "Ornaments"),
            ("metronome_marks", "Metronome mark"), ("lyrics", "Lyrics"), ("chord_symbols", "Chord symbols")]


def content_section(m):
    out = []
    sub = m[m["score_has_tempo"]]
    # check of the paper's existing sentence on the 560-score subset, then the full 806
    med560 = sub.groupby("label")[["notes_count", "ioi_tempo_avg"]].median()
    med806 = m.groupby("label")[["notes_count", "ioi_tempo_avg", "onsets_per_min"]].median()
    mono = lambda s: bool(np.all(np.diff(s.values) > 0))  # noqa: E731

    per = m.groupby("label").agg(
        N=("id", "size"), notes=("notes_count", "median"), measures=("measures_count", "median"),
        seconds=("seconds", "median"), ioi_tempo=("ioi_tempo_avg", "median"),
        onsets_min=("onsets_per_min", "median"), notes_sec=("notes_per_sec_written", "median"),
        pitch_range=("pitch_range", "median"), pitch_entropy=("pitch_entropy", "median"),
        pc_entropy=("pc_entropy", "median"), polyphony=("polyphony_mean", "median"),
        max_polyphony=("polyphony_max", "median"), voices_max=("voices_max", "median"),
        multivoice_measures=("multivoice_share", "mean"), chord_span_p95=("chord_span_p95", "median"),
        chords_over_octave=("chords_over_octave", "mean"),
        shortest=("shortest_type", lambda s: s.mode().iloc[0]))
    per.to_csv(OUT / "per_grade.csv")
    show = per.copy()
    for col in ["notes", "measures", "seconds", "ioi_tempo", "onsets_min", "pitch_range", "chord_span_p95"]:
        show[col] = show[col].round().astype(int)
    for col in ["notes_sec", "pitch_entropy", "pc_entropy", "polyphony", "max_polyphony", "voices_max"]:
        show[col] = show[col].round(2)
    for col in ["multivoice_measures", "chords_over_octave"]:
        show[col] = (show[col] * 100).round(1).astype(str) + "%"

    fig, axes = plt.subplots(3, 4, figsize=(13, 8.6))
    for ax, (col, title, log, _) in zip(axes.flat, CONTENT):
        boxes_by_grade(ax, m, col, title, log)
    for ax in axes[-1]:
        ax.set_xlabel("Grade")
    fig.suptitle(f"Score content by grade (all {len(m)} scores; box = IQR, line = median)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    f_content = save(fig, "36_content_by_grade.png")

    # time and key signatures
    ts = m["time_sig_first"].fillna("none").value_counts()
    ts = pd.concat([ts.head(9), pd.Series({"other": ts.iloc[9:].sum()})]) if len(ts) > 10 else ts
    keys = m["key_fifths_first"].value_counts().sort_index()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1, 1.4, 1]})
    a1.barh(range(len(ts))[::-1], ts.values, color=BLUE, height=0.62)
    for i, v in zip(range(len(ts))[::-1], ts.values):
        a1.text(v + 4, i, str(v), va="center", fontsize=8, color=INK2)
    a1.set_yticks(range(len(ts))[::-1], ts.index)
    a1.grid(axis="y", visible=False)
    a1.tick_params(axis="y", length=0)
    a1.set_title("First time signature")
    a1.set_xlabel("Scores")
    a2.bar(keys.index, keys.values, color=BLUE, width=0.7)
    for k, v in keys.items():
        a2.text(k, v + 3, str(v), ha="center", fontsize=8, color=INK2)
    a2.set_xticks(range(-7, 8), [f"{abs(k)}{'♭' if k < 0 else '♯' if k > 0 else ''}" for k in range(-7, 8)])
    a2.grid(axis="x", visible=False)
    a2.set_title("First key signature (flats ← → sharps)")
    a2.set_ylabel("Scores")
    changes = pd.DataFrame({
        "Time-signature change": m.groupby("label")["time_sig_changes"].apply(lambda s: (s > 0).mean() * 100),
        "Key-signature change": m.groupby("label")["key_sigs_distinct"].apply(lambda s: (s > 1).mean() * 100),
        "Tempo change": m.groupby("label")["tempo_changes"].apply(lambda s: (s > 0).mean() * 100)})
    for (name, s), colour, mk in zip(changes.items(), [BLUE, ORANGE, AQUA], ["o", "s", "D"]):
        a3.plot(s.index, s.values, color=colour, linewidth=2, marker=mk, markersize=5, markeredgecolor=SURFACE,
                label=name)
        a3.annotate(name.split(" ")[0], (9, s.values[-1]), xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
    a3.set_xticks(range(1, 10))
    a3.set_xlim(0.6, 10.6)
    a3.set_xlabel("Grade")
    a3.set_ylabel("% of scores")
    a3.set_title("Scores with a change of …")
    a3.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, handlelength=1.5)
    fig.tight_layout()
    f_sigs = save(fig, "36_signatures.png")

    # shortest note value by grade (100% stacked, ordinal ramp)
    types = ["quarter", "eighth", "16th", "32nd", "64th", "128th"]
    ramp = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
    st = pd.crosstab(m["label"], m["shortest_type"], normalize="index").reindex(columns=types).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(8.5, 3.3))
    bottom = np.zeros(9)
    for t, colour in zip(types, ramp):
        ax.bar(st.index, st[t], bottom=bottom, color=colour, width=0.7, edgecolor=SURFACE, linewidth=1.2, label=t)
        bottom += st[t].values
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Grade")
    ax.set_ylabel("% of scores")
    ax.set_title("Shortest written note value by grade")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=6, loc="upper left", bbox_to_anchor=(0, 1.18), fontsize=8, handlelength=1)
    fig.tight_layout()
    f_short = save(fig, "36_shortest_note.png")

    # notation markings: % of scores with at least one, by grade
    mk = pd.DataFrame({name: m.groupby("label")[col].apply(lambda s: (s > 0).mean() * 100)
                       for col, name in MARKINGS}).T
    mk["All"] = [(m[col] > 0).mean() * 100 for col, _ in MARKINGS]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(mk.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(mk.shape[0]):
        for j in range(mk.shape[1]):
            v = mk.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="#ffffff" if v > 55 else INK,
                    fontweight="bold" if j == mk.shape[1] - 1 else "normal")
    ax.set_xticks(range(mk.shape[1]), [str(g) for g in range(1, 10)] + ["All"])
    ax.set_yticks(range(mk.shape[0]), mk.index)
    ax.axvline(8.5, color=SURFACE, linewidth=4)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Grade")
    ax.set_title("% of scores with at least one marking of each kind")
    fig.tight_layout()
    f_mark = save(fig, "36_markings.png")

    avail = pd.DataFrame({"Scores with ≥1": [(m[c] > 0).sum() for c, _ in MARKINGS],
                          "%": [f"{(m[c] > 0).mean():.0%}" for c, _ in MARKINGS],
                          "Median count where present": [int(m.loc[m[c] > 0, c].median()) if (m[c] > 0).any()
                                                         else 0 for c, _ in MARKINGS],
                          "Spearman with grade": [fmt(spearmanr(m[c], m["label"])[0]) for c, _ in MARKINGS]},
                         index=[n for _, n in MARKINGS])
    tsig = m["time_sig_first"].value_counts()
    simple = m["time_sig_first"].isin(["4/4", "2/4", "3/4", "2/2"]).mean()

    out.append(f"""### Length and tempo by grade

Check of the existing sentence (560 scores with a metronome mark, RubricNet's IOI tempo): median note count
{med560["notes_count"].iloc[0]:.0f} at grade 1 → {med560["notes_count"].iloc[-1]:.0f} at grade 9, median IOI tempo
{med560["ioi_tempo_avg"].iloc[0]:.0f} → {med560["ioi_tempo_avg"].iloc[-1]:.0f} onsets per minute; monotonic:
notes {mono(med560["notes_count"])}, IOI tempo {mono(med560["ioi_tempo_avg"])}.

All {len(m)} scores: median note count {med806["notes_count"].iloc[0]:.0f} → {med806["notes_count"].iloc[-1]:.0f}
(monotonic: {mono(med806["notes_count"])}); median IOI tempo {med806["ioi_tempo_avg"].iloc[0]:.0f} →
{med806["ioi_tempo_avg"].iloc[-1]:.0f} (monotonic: {mono(med806["ioi_tempo_avg"])}); onsets per minute at the stored
playback tempo {med806["onsets_per_min"].iloc[0]:.0f} → {med806["onsets_per_min"].iloc[-1]:.0f}
(monotonic: {mono(med806["onsets_per_min"])}).

Definitions:
- **IOI tempo** = 60 / mean inter-onset interval in seconds, averaged over both hands (RubricNet's descriptors,
  as in `model_comparison`). RubricNet reads the {(~m["score_has_tempo"]).sum()} scores without a metronome mark
  at 100 bpm, so on the full set this column is partly at an assumed tempo.
- **Onsets per minute (stored tempo)** = distinct note onsets (both hands merged) per minute of written duration,
  using the playback tempo stored in the file (`<sound tempo>`; {m["has_stored_tempo"].sum()} of {len(m)} files have
  one, the rest play at MuseScore's 120). Spearman with IOI tempo: {spearmanr(m["onsets_per_min"],
  m["ioi_tempo_avg"])[0]:.2f} (all), {spearmanr(sub["onsets_per_min"], sub["ioi_tempo_avg"])[0]:.2f} (560).
- **Duration** = written length at the stored tempo; repeats and D.C./D.S. are not unfolded
  ({m["has_repeats"].mean():.0%} of scores have repeat signs or jumps), so it is a lower bound on performance time.
  Median {m["seconds"].median():.0f} s; IQR {m["seconds"].quantile(.25):.0f}–{m["seconds"].quantile(.75):.0f} s.
- **Polyphony** = notes sounding at each onset (all staves), averaged over onsets. **Voices** = most
  `<voice>`s holding notes on one staff in one measure. **Chord span** = pitch spread of notes struck together on
  one staff (95th percentile per score; % > octave = share of such chords wider than 12 semitones).
- Pitch range is over all notes; pitch entropy is the Shannon entropy of MIDI pitches (pitch-class entropy over
  the 12 classes is in the table).

{f_content}

Medians per grade (all {len(m)}; `multivoice_measures` and `chords_over_octave` are means of per-score shares):

{md(show.rename_axis("Grade"))}

### Time and key signatures

{tsig.iloc[0]} scores ({tsig.iloc[0] / len(m):.0%}) start in {tsig.index[0]}; {simple:.0%} start in 4/4, 2/4, 3/4
or 2/2. {(m["time_sig_changes"] > 0).mean():.0%} change time signature, {(m["key_sigs_distinct"] > 1).mean():.0%}
change key signature and {(m["tempo_changes"] > 0).mean():.0%} have more than one stored tempo. Key signatures stay
close to C: {(m["key_accidentals"] <= 2).mean():.0%} have at most two sharps or flats (the mode is rarely
encoded, so major and minor are not separated). Score 792 has no time signature.

{f_sigs}

{f_short}

The shortest written note is a 16th in {(m["shortest_type"] == "16th").mean():.0%} of scores; Spearman between grade
and the shortest note value (rank, quarter = 0 … 128th = 5) is {spearmanr(m["shortest_rank"], m["label"],
                                                                        nan_policy="omit")[0]:.2f}.

### Notation markings

{f_mark}

{md(avail.rename_axis("Marking"))}

Fingering is rarely annotated ({(m["fingering"] > 0).mean():.0%} of scores).
Lyrics appear in {(m["lyrics"] > 0).sum()} scores (song arrangements that kept the words).
""")
    return "\n".join(out)


# ---------------------------------------------------------------- piano key usage (part of 3.6)

PC_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
BLACK_PC = {1, 3, 6, 8, 10}
KEYS = np.arange(21, 109)  # A0 .. C8


def key_name(midi):
    return f"{PC_NAMES[midi % 12]}{midi // 12 - 1}"


def draw_keyboard(ax):
    """An 88-key keyboard aligned with bar positions 21..108; Cs labelled."""
    from matplotlib.patches import Rectangle
    for k in KEYS:
        black = k % 12 in BLACK_PC
        ax.add_patch(Rectangle((k - 0.5, 0.4 if black else 0), 1, 0.6 if black else 1,
                               facecolor=INK if black else "#ffffff", edgecolor=AXIS, linewidth=0.4,
                               zorder=2 if black else 1))
    for k in KEYS[KEYS % 12 == 0]:
        ax.text(k, -0.25, key_name(k), ha="center", va="top", fontsize=7.5, color=INK2)
    ax.set_xlim(20.5, 108.5)
    ax.set_ylim(-0.9, 1)
    ax.axis("off")


def key_usage_section(m):
    kc = pd.read_csv(OUT / "key_counts.csv", index_col="file_name")
    counts = kc[[str(k) for k in KEYS]].set_axis(KEYS, axis=1).loc[m["file_name"]]
    grade = m.set_index("file_name")["label"]
    total = counts.sum()
    share = total / total.sum() * 100
    used_by = (counts > 0).mean() * 100
    black = np.array([k % 12 in BLACK_PC for k in KEYS])

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(13, 6.6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.6, 0.55], "hspace": 0.12})
    a1.bar(KEYS[~black], share.values[~black], width=0.85, color=BLUE, label="White key")
    a1.bar(KEYS[black], share.values[black], width=0.85, color="#0d366b", label="Black key")
    for k in share.sort_values(ascending=False).index[:5]:
        a1.annotate(key_name(k), (k, share[k]), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=7.5, color=INK2)
    a1.set_ylim(0, share.max() * 1.12)
    a1.set_ylabel("% of all note attacks")
    a1.set_title(f"How often each piano key is played (all {len(m)} scores, {int(total.sum()):,} attacks)")
    a1.legend(loc="upper right", fontsize=8)
    a1.grid(axis="x", visible=False)
    a2.fill_between(KEYS, used_by.values, step="mid", color="#cde2fb")
    a2.step(KEYS, used_by.values, where="mid", color=BLUE, linewidth=1.5)
    a2.set_ylabel("% of scores\nusing the key")
    a2.set_ylim(0, 100)
    a2.grid(axis="x", visible=False)
    draw_keyboard(a3)
    f_bar = save(fig, "36_key_usage.png")

    # by grade: each row is that grade's attacks
    by_grade = counts.groupby(grade.loc[counts.index].values).sum()
    rows = by_grade.div(by_grade.sum(axis=1), axis=0) * 100
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 4.6), sharex=True,
                                 gridspec_kw={"height_ratios": [4, 0.6], "hspace": 0.08})
    from matplotlib.colors import LinearSegmentedColormap
    a1.imshow(rows.values, cmap=LinearSegmentedColormap.from_list("seq", SEQ), aspect="auto",
              extent=(20.5, 108.5, 9.5, 0.5))
    ranges = []
    for g, r in rows.iterrows():
        cum = r.cumsum() / 100
        lo, hi = KEYS[np.searchsorted(cum.values, 0.05)], KEYS[np.searchsorted(cum.values, 0.95)]
        ranges.append((g, lo, hi))
        a1.plot([lo - 0.5, lo - 0.5], [g - 0.45, g + 0.45], color=ORANGE, linewidth=1.5)
        a1.plot([hi + 0.5, hi + 0.5], [g - 0.45, g + 0.45], color=ORANGE, linewidth=1.5)
        a1.text(109.2, g, f"{key_name(lo)}–{key_name(hi)}", va="center", fontsize=7.5, color=INK2)
    a1.set_yticks(range(1, 10))
    a1.set_xticks(KEYS[KEYS % 12 == 0], [])
    a1.set_ylabel("Grade")
    a1.grid(False)
    a1.set_title("Share of each grade's attacks per key (orange ticks: central 90% of attacks)")
    draw_keyboard(a2)
    f_grade = save(fig, "36_key_usage_by_grade.png")

    # pitch classes by grade bucket
    pc = counts.T.groupby(KEYS % 12).sum().T
    pcb = pc.groupby(m.set_index("file_name").loc[pc.index, "bucket"].values, observed=True).sum()
    pcb = pcb.div(pcb.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(9, 3.4))
    width = 0.27
    for i, (b, colour) in enumerate(BUCKETS.items()):
        ax.bar(np.arange(12) + (i - 1) * width, pcb.loc[b].values, width=width - 0.03, color=colour, label=b)
    ax.set_xticks(range(12), PC_NAMES)
    ax.set_ylabel("% of attacks")
    ax.set_title("Pitch classes by grade bucket (black keys: C♯, D♯, F♯, G♯, A♯)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.set_title("Pitch classes by grade bucket (black keys: C♯, D♯, F♯, G♯, A♯)", pad=24)
    f_pc = save(fig, "36_pitch_classes.png")

    top = share.sort_values(ascending=False).head(15)
    top_t = pd.DataFrame({"Key": [key_name(k) for k in top.index], "MIDI": top.index,
                          "% of attacks": top.round(2).values, "% of scores using it": used_by[top.index].round(1).values})
    centre = share[(KEYS >= 48) & (KEYS < 84)].sum()
    rare = used_by[used_by < 5]
    black_share = share[black].sum()
    black_by_grade = (by_grade.loc[:, black].sum(axis=1) / by_grade.sum(axis=1) * 100).round(1)
    range_t = pd.DataFrame([{"Grade": g, "Central 90% of attacks": f"{key_name(lo)}–{key_name(hi)}",
                             "Keys in it": hi - lo + 1, "% black-key attacks": black_by_grade[g]}
                            for g, lo, hi in ranges])
    rank_pc = pc.sum().sort_values(ascending=False)
    return f"""### Piano key usage

Note attacks per key of the 88-key keyboard (A0–C8, MIDI 21–108; tied continuations and grace notes left out;
middle C = C4 = MIDI 60). All 88 keys are played somewhere in the dataset and no note falls outside the keyboard.
The most played key is {key_name(top.index[0])} ({top.iloc[0]:.2f}% of all attacks), followed by
{", ".join(key_name(k) for k in top.index[1:5])}. The three octaves C3–B5 hold {centre:.0f}% of all attacks. Black keys
take {black_share:.1f}% of attacks (5 of 12 keys per octave would be 41.7% if all keys were equally used). {len(rare)} keys
appear in fewer than 5% of scores: {key_name(rare.index.min())}–{key_name(rare[rare.index < 60].index.max())} at the
bottom and {key_name(rare[rare.index > 60].index.min())}–{key_name(rare.index.max())} at the top.

{f_bar}

{md(top_t, index=False)}

The range widens with grade, mostly upwards: the central 90% of attacks spans {key_name(ranges[0][1])}–{key_name(ranges[0][2])}
at grade 1 and {key_name(ranges[-1][1])}–{key_name(ranges[-1][2])} at grade 9; the bottom reaches
{key_name(min(lo for _, lo, _ in ranges))} by grade {min(g for g, lo, _ in ranges if lo == min(x for _, x, _ in ranges))} and moves little after that, while the top keeps rising.
The share of black-key attacks doubles, from {black_by_grade.iloc[0]:.0f}% at grade 1 to {black_by_grade.iloc[-1]:.0f}% at grade 9 (each
row of the heatmap sums to 100%):

{f_grade}

{md(range_t, index=False)}

Pitch classes, from most to least played: {", ".join(f"{PC_NAMES[p]} {rank_pc[p] / rank_pc.sum():.1%}" for p in rank_pc.index)}.
Harder pieces use more black keys and spread more evenly over the 12 classes.

{f_pc}

Per-score counts are in `results/key_counts.csv`.
"""


# ---------------------------------------------------------------- 3.7 confounds and correlations

CORR = [
    ("notes_count", "Note count"), ("measures_count", "Measure count"), ("seconds", "Duration (s)"),
    ("notes_per_sec_written", "Notes per second"), ("notes_per_measure", "Notes per measure"),
    ("ioi_tempo_avg", "IOI tempo (RubricNet)"), ("onsets_per_min", "Onsets per minute"),
    ("first_tempo_qpm", "Stored tempo (qpm)"), ("pitch_range", "Pitch range"), ("pitch_entropy", "Pitch entropy"),
    ("pc_entropy", "Pitch-class entropy"), ("polyphony_mean", "Mean polyphony"), ("voices_max", "Max voices"),
    ("chord_span_p95", "Chord span (p95)"), ("chords_over_octave", "Chords > octave"),
    ("shortest_rank", "Shortest note value"), ("key_accidentals", "Key-signature accidentals"),
]


def corr_table(m, cols):
    rows = []
    y = m["label"].values
    for col, name in cols:
        x = m[col].values.astype(float)
        ok = ~np.isnan(x)
        x_, y_ = x[ok], y[ok]
        rho = spearmanr(x_, y_)[0]
        tau = kendalltau(x_, y_, variant="c")[0]
        rlo, rhi = boot_ci(lambda a, b: spearmanr(a, b)[0], x_, y_)
        tlo, thi = boot_ci(lambda a, b: kendalltau(a, b, variant="c")[0], x_, y_)
        part = partial_spearman(x_, y_, m["notes_count"].values[ok]) if col != "notes_count" else np.nan
        rows.append({"Feature": name, "col": col, "N": int(ok.sum()), "rho": rho, "rho_lo": rlo, "rho_hi": rhi,
                     "tau_c": tau, "tau_lo": tlo, "tau_hi": thi, "partial_rho_given_notes": part})
    return pd.DataFrame(rows)


def composer_baseline(m, fallback, folds=5, seed=0):
    """Mean label of the same composer's training scores, rounded (5-fold CV); unseen composers use `fallback`."""
    fold = np.random.default_rng(seed).integers(0, folds, len(m))
    pred, seen = np.asarray(fallback, dtype=float).copy(), np.zeros(len(m), bool)
    for k in range(folds):
        train, test = m[fold != k], np.where(fold == k)[0]
        means = train.groupby("composer")["label"].mean()
        hit = m["composer"].iloc[test].map(means)
        idx = test[hit.notna().values]
        pred[idx] = np.floor(hit.dropna().values + 0.5)
        seen[idx] = True
    return pred.astype(int), seen


def variance_explained(d):
    """Share of label variance between composers: eta squared and the less biased omega squared."""
    groups = [g["label"].values for _, g in d.groupby("composer")]
    grand, n, k = d["label"].mean(), len(d), len(groups)
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((d["label"] - grand) ** 2).sum()
    ms_within = (ss_total - ss_between) / (n - k)
    return ss_between / ss_total, (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)


def composer_verdict(acc, n, top):
    a = acc.set_index("Predictor")
    seen, cipi_seen = a.loc["Composer only, seen composers"], a.loc["CIPI ensemble, seen composers"]
    seen_nt, cipi_nt = a.loc[f"Composer only, seen, without {top}"], a.loc[f"CIPI ensemble, seen, without {top}"]
    combo, cipi = a.loc["Composer, unseen → note count"], a.loc["CIPI ensemble"]
    return f"""On the {int(seen["N"])} scores whose composer appears in the training folds, the composer's mean grade
alone gets acc9 {seen["Acc9"]:.3f} / balanced acc9 {seen["Bal. acc9"]:.3f} / acc ±1 {seen["Acc ±1"]:.3f}, against
{cipi_seen["Acc9"]:.3f} / {cipi_seen["Bal. acc9"]:.3f} / {cipi_seen["Acc ±1"]:.3f} for the CIPI ensemble on the same
scores. Much of this is {top}: without those scores it drops to {seen_nt["Acc9"]:.3f} / {seen_nt["Bal. acc9"]:.3f} /
{seen_nt["Acc ±1"]:.3f} (ensemble {cipi_nt["Acc9"]:.3f} / {cipi_nt["Bal. acc9"]:.3f} / {cipi_nt["Acc ±1"]:.3f}).
Composer mean with the note-count fallback, on all {n} scores, gets {combo["Acc9"]:.3f} / {combo["Bal. acc9"]:.3f} /
{combo["Acc ±1"]:.3f} (ensemble {cipi["Acc9"]:.3f} / {cipi["Bal. acc9"]:.3f} / {cipi["Acc ±1"]:.3f}). This is the
floor a model can reach without reading a note, given random (not composer-disjoint) splits."""


def confounds_section(m):
    out = []
    sub = m[m["score_has_tempo"]]
    c806, c560 = corr_table(m, CORR), corr_table(sub, CORR[:7])
    pd.concat([c806.assign(subset="806"), c560.assign(subset="560")]).to_csv(OUT / "correlations.csv",
                                                                             index=False)

    # dot plot: rho and tau_c with CIs
    c = c806.sort_values("rho")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    y = np.arange(len(c))
    for off, (k, lo, hi), colour, name, mk in ((0.17, ("rho", "rho_lo", "rho_hi"), BLUE, "Spearman ρ", "o"),
                                               (-0.17, ("tau_c", "tau_lo", "tau_hi"), ORANGE, "Kendall τc", "s")):
        ax.hlines(y + off, c[lo], c[hi], color=colour, linewidth=2)
        ax.plot(c[k], y + off, mk, color=colour, markersize=6, markeredgecolor=SURFACE, label=name)
    for yi, r in zip(y, c.itertuples()):
        ax.text(1.0, yi, f"{r.rho:.2f} / {r.tau_c:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.text(1.0, len(c) - 0.4, "ρ / τc", fontsize=7.5, color=MUTED, fontweight="bold")
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.axvline(0.48, color=MUTED, linewidth=1, linestyle="--", zorder=0)
    ax.text(0.47, -0.9, "CIPI: τc = 0.48 for length", fontsize=7.5, color=MUTED, ha="right")
    ax.set_yticks(y, c["Feature"])
    ax.set_ylim(-1.3, len(c) - 0.2)
    ax.set_xlim(min(-0.05, c["rho_lo"].min() - 0.05), 1.12)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"Correlation with grade (all {len(m)} scores; 95% bootstrap CI)")
    ax.set_title("What tracks the annotator's grade")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    f_corr = save(fig, "37_correlations.png")

    # length and density vs grade
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (col, title) in zip(axes, [("notes_count", "Note count"), ("measures_count", "Measure count"),
                                       ("notes_per_sec_written", "Note density (notes per second)")]):
        ax.scatter(m["label"] + rng.uniform(-0.28, 0.28, len(m)), m[col], s=7, color=BLUE, alpha=0.35,
                   edgecolor="none")
        med = m.groupby("label")[col].median()
        ax.plot(med.index, med.values, color="#0d366b", linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE)
        r = c806.set_index("col").loc[col]
        ax.set_yscale("log")
        ax.set_xticks(range(1, 10))
        ax.set_xlabel("Grade")
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{title}: ρ = {r.rho:.2f}, τc = {r.tau_c:.2f}")
    axes[0].set_ylabel("Value (log scale); line = median")
    fig.tight_layout()
    f_len = save(fig, "37_length_density.png")

    # composer identity as a predictor (the memorisation floor)
    counts = m["composer"].value_counts()
    multi = m[m["composer"].map(counts) >= 2]
    groups = [g["label"].values for _, g in multi.groupby("composer")]
    kw = kruskal(*groups)
    eta2, omega2 = variance_explained(multi)
    eta2_nt, omega2_nt = variance_explained(multi[multi["composer"] != counts.index[0]])

    majority = np.full(len(m), int(m["label"].mode().iloc[0]))
    length = binned_baseline(m, "notes_count")
    comp_major, seen = composer_baseline(m, majority)
    comp_length, _ = composer_baseline(m, length)
    not_top = (m["composer"] != counts.index[0]).values
    rows = [("Majority class (grade 2)", m, majority),
            ("Note count only (deciles)", m, length),
            ("Composer only (unseen → majority)", m, comp_major),
            ("Composer, unseen → note count", m, comp_length),
            ("Composer only, seen composers", m[seen], comp_major[seen]),
            (f"Composer only, seen, without {counts.index[0]}", m[seen & not_top], comp_major[seen & not_top]),
            ("CIPI ensemble", m, m["pred_cipi"].values),
            ("CIPI ensemble, seen composers", m[seen], m["pred_cipi"].values[seen]),
            (f"CIPI ensemble, seen, without {counts.index[0]}", m[seen & not_top],
             m["pred_cipi"].values[seen & not_top]),
            ("RubricNet (560)", sub, sub["pred_rubricnet"].values)]
    acc = pd.DataFrame([{"Predictor": name, "N": len(d), **accuracies(d["label"].values, p),
                         "Spearman": spearmanr(p, d["label"])[0] if np.ptp(p) else np.nan,
                         "MSE": np.mean((p - d["label"].values) ** 2)}
                        for name, d, p in rows])
    keep = ["Acc9", "Bal. acc9", "Acc ±1", "Bal. acc ±1", "Spearman", "MSE"]

    fig, ax = plt.subplots(figsize=(9, 3.9))
    plot_rows = [0, 1, 2, 3, 6]
    metrics = ["Acc9", "Bal. acc9", "Acc ±1", "Bal. acc ±1"]
    colours = [AXIS, BLUE, ORANGE, AQUA, "#eda100"]
    width = 0.16
    for i, (ri, colour) in enumerate(zip(plot_rows, colours)):
        vals = acc.loc[ri, metrics].values.astype(float)
        xs = np.arange(len(metrics)) + (i - 2) * width
        ax.bar(xs, vals, width=width - 0.02, color=colour, label=acc.loc[ri, "Predictor"])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}"[1:], ha="center", fontsize=6.8, color=INK2)
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Score")
    ax.set_title(f"Composer identity as a predictor (all {len(m)} scores, 5-fold CV)")
    ax.legend(fontsize=7.5, loc="upper left", ncol=1)
    fig.tight_layout()
    f_comp = save(fig, "37_composer_floor.png")

    t = c806.copy()
    t["Spearman ρ [95% CI]"] = [f"{r:.2f} [{lo:.2f}, {hi:.2f}]" for r, lo, hi in zip(t.rho, t.rho_lo, t.rho_hi)]
    t["Kendall τc [95% CI]"] = [f"{r:.2f} [{lo:.2f}, {hi:.2f}]" for r, lo, hi in zip(t.tau_c, t.tau_lo, t.tau_hi)]
    t["Partial ρ, note count held"] = t["partial_rho_given_notes"].map(lambda v: fmt(v))
    t560 = c560.copy()
    t560["Spearman ρ"] = t560["rho"].round(2)
    t560["Kendall τc"] = t560["tau_c"].round(2)
    r = c806.set_index("col")
    r5 = c560.set_index("col")
    acc_show = acc.assign(**{k: acc[k].round(3) for k in keep})[["Predictor", "N"] + keep].fillna("–")

    out.append(f"""### Length

On all {len(m)} scores, Spearman between note count and grade is **{r.loc["notes_count", "rho"]:.2f}**
(Kendall τc **{r.loc["notes_count", "tau_c"]:.2f}**) and for measure count {r.loc["measures_count", "rho"]:.2f}
(τc {r.loc["measures_count", "tau_c"]:.2f}). On the 560-score subset: note count ρ {r5.loc["notes_count", "rho"]:.2f}
/ τc {r5.loc["notes_count", "tau_c"]:.2f}, measure count ρ {r5.loc["measures_count", "rho"]:.2f} / τc
{r5.loc["measures_count", "tau_c"]:.2f}. CIPI reports τc = 0.48 for length against grade, so length is a
stronger confound here than in CIPI. Duration at the stored tempo: ρ {r.loc["seconds", "rho"]:.2f}, τc
{r.loc["seconds", "tau_c"]:.2f}.

τc is Stuart's tau-c (`scipy.stats.kendalltau(variant="c")`), the variant CIPI uses for an ordinal label with many
ties; it is always smaller than ρ for the same data, so compare τc with τc. CIs are 95% bootstrap percentile
intervals over scores ({BOOT} resamples). "Partial ρ" is Spearman with note count partialled out (linear on
ranks), i.e. what a feature says about the grade beyond length.

{f_len}

### Note density

Notes per second at the stored tempo is the single strongest correlate of grade: ρ
{r.loc["notes_per_sec_written", "rho"]:.2f}, τc
{r.loc["notes_per_sec_written", "tau_c"]:.2f}; partial ρ with note count held fixed
{r.loc["notes_per_sec_written", "partial_rho_given_notes"]:.2f}. Notes per measure: ρ
{r.loc["notes_per_measure", "rho"]:.2f}, τc {r.loc["notes_per_measure", "tau_c"]:.2f}. IOI tempo (RubricNet): ρ
{r.loc["ioi_tempo_avg", "rho"]:.2f} (all {len(m)}) and {r5.loc["ioi_tempo_avg", "rho"]:.2f} (560; lower on all 806
because RubricNet reads unmarked scores at 100 bpm). Density stays strongly related to grade with length held fixed,
whereas duration and measure count turn negative: at a given note count, harder pieces are shorter and denser.

{f_corr}

All features, all {len(m)} scores:

{md(t[["Feature", "N", "Spearman ρ [95% CI]", "Kendall τc [95% CI]", "Partial ρ, note count held"]], index=False)}

The 560-score subset (length and tempo features):

{md(t560[["Feature", "N", "Spearman ρ", "Kendall τc"]], index=False)}

### Composer identity (the memorisation floor)

A predictor that only knows who wrote the piece: the mean grade of the same composer's scores in the training
folds, rounded (5-fold CV, same folds as the note-count baseline). A composer with no training scores falls back
to the majority grade or to the note-count baseline. {seen.sum()} of {len(m)} scores ({seen.mean():.0%}) have their
composer in the training folds; {(counts == 1).sum()} composers have a single score and can never be memorised.

Among composers with ≥ 2 scores ({len(multi)} scores, {len(groups)} composers), composer identity explains
η² = {eta2:.2f} of the grade variance, ω² = {omega2:.2f} after correcting for the many small groups
(Kruskal–Wallis H = {kw.statistic:.0f}, p = {kw.pvalue:.1e}). Without {counts.index[0]} ({counts.iloc[0]} scores, all
grades {m.loc[m["composer"] == counts.index[0], "label"].min()}–{m.loc[m["composer"] == counts.index[0], "label"].max()}):
η² = {eta2_nt:.2f}, ω² = {omega2_nt:.2f}.

{f_comp}

{md(acc_show, index=False)}

{composer_verdict(acc, len(m), counts.index[0])}
""")
    return "\n".join(out)


def main():
    OUT.mkdir(exist_ok=True)
    m = load_all()
    text = f"""# Dataset statistics (sections 3.5–3.7)

Generated by `nina_stats/analyze.py` from the dataset sheet, `results/content.csv` (`extract.py`, one parse of
every MusicXML), `labels/` (composer and score classification) and the existing model results. All {len(m)} scores
unless a section says otherwise; "560" is the subset with a metronome mark used in `model_comparison`.

## 3.5 Repertoire composition

{composition_section(m)}
## 3.6 Score content statistics

{content_section(m)}
{key_usage_section(m)}
## 3.7 Confounds and correlations

{confounds_section(m)}"""
    (OUT / "REPORT.md").write_text(text)
    print(f"wrote {OUT / 'REPORT.md'} and {len(list(FIG.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
