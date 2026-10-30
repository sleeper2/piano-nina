# nina_stats

Dataset statistics for the paper's sections 3.5 (repertoire composition), 3.6 (score content) and 3.7 (confounds
and correlations). An analysis only (no Docker): it parses the MusicXML files directly and reads the existing
model results in `cipi_model/results`, `rubricnet_model/results` and the helpers in `model_comparison/compare.py`.

## Run

Needs Python with pandas, numpy, scipy, matplotlib, lxml and openpyxl:

```
python nina_stats/extract.py    # ~3 s: parses all 806 scores -> results/content.csv, metadata.csv, key_counts.csv
python nina_stats/analyze.py    # ~10 s: results/REPORT.md, per_grade.csv, correlations.csv, figures/*.png
python nina_stats/coordination.py   # ~25 s: group A -> results/COORDINATION.md + coordination*.csv
python nina_stats/hand_features.py  # ~30 s: groups B-E -> results/HAND_FEATURES.md + hand_features*.csv
```

`hand_features.py` and `coordination.py` also need scikit-learn (held-out model comparison).

Re-run `analyze.py` after the model results or the labels change.

## Files

- `extract.py`: one pass over `../score_musicxml`. Per score: note/attack/grace counts, measures, written
  duration at the stored playback tempo (repeats not unfolded), onsets per minute, pitch range and entropy,
  polyphony, voices per staff, chord span, shortest note, time/key signatures and their changes, and counts of
  dynamics, hairpins, articulations, slurs, pedal, fingering, ornaments, metronome marks, lyrics and chord symbols.
  `metadata.csv` keeps the header text (titles, creators, rights, credits) used for labelling.
- `labels/composers.csv`: one row per composer (505): `entity`, `gender` (documented only, else unknown),
  `birth_year`, `era`, `kind`, confidence and a note.
- `labels/scores.csv`: one row per score (806): `genre`, `form`, `medium` (what the music was originally written
  for: piano/other) and `simplified` (yes/no/unknown), confidence and a note.
  Both were labelled by LLM agents from the composer names, titles and MusicXML header text (inputs in
  `labels/inputs/`, raw outputs in `labels/agent_outputs/`) and spot-checked; low-confidence rows are marked.
  Yohei Kato's gender (129 scores) is not source-checked.
- `FEATURES_SPEC.md`: the single spec for the hand and coordination features, listing only what is implemented,
  under the actual column names.
- `coordination.py`: group A, written bimanual coordination: onset alignment, alternation, rhythmic independence,
  subdivision mismatch, density and simultaneous sounding.
- `hand_features.py`: groups B–E: pitch span, black keys, hand displacement and hand crossing. It also holds the
  shared helpers (hand assignment, statistics, correlations, held-out evaluation).
- Both write one row per score, a per-measure table, correlations with grade and a held-out comparison against the
  5-control baseline (composer-disjoint CV). Both parse with `extract.parse(path, exact=True)`, which keeps onsets
  as exact Fractions.
- `tests/test_features.py`: hand-computed values for two small test scores (`python nina_stats/tests/test_features.py`).
- `analyze.py`: the report and figures. Source type = simplified version if `simplified == yes`, else arrangement
  if `medium == other`, else original piano work.
