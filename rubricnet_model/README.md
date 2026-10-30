# rubricnet_model

Evaluates the pretrained RubricNet piano-score difficulty model from
[PRamoneda/rubricnet](https://github.com/PRamoneda/rubricnet)
(Ramoneda et al., "Towards Explainable and Interpretable Musical Difficulty Estimation: A parameter-efficient
approach", ISMIR 2024) on the scores in `../score_musicxml/` against the labels in `../806_2_score_dataset_clean.xlsx`.
There's no retraining: the camera-ready checkpoints are used as-is.

## Run

From this folder, with Docker running:

```sh
docker compose build
docker compose run --rm rubricnet                      # all scores predicted, report on those with a tempo mark
TEMPO=all docker compose run --rm rubricnet            # report on all 806 scores
LIMIT=5 docker compose run --rm rubricnet              # smoke test on the first 5 scores
WORKERS=8 docker compose run --rm rubricnet            # more parallel scores
docker compose run --rm rubricnet python evaluate.py   # recompute metrics only
```

Environment variables: `TEMPO` (`marked`, the default, `explicit` or `all`), `WORKERS` (parallel scores, default 4),
`LIMIT` (only the first N scores, 0 = all), `SCORE_TIMEOUT` (seconds per score, default 1800).

**Tempo selection.** One of RubricNet's inputs, the average time between onsets, is measured in seconds. A score
without a metronome mark is read at 100 bpm, which can be far off (e.g. Flight of the Bumblebee). `TEMPO` picks the
scores:

- `marked` (default): scores where the extractor reads a metronome mark, printed or playback-only (`<sound tempo>`).
  246 of 806 scores have none and are skipped, leaving 560. Whether a score has a mark is only known after
  extraction, so `predict.py` processes every score and `evaluate.py` drops those with `score_has_tempo = false`.
- `explicit`: the stricter check copied from `cipi_model/tempo_subset.py` into `tempo.py`. It needs a printed numeric
  mark in the first measure, the same value for every later mark, and every playback tempo matching it. 439 scores
  qualify, the same ones as `cipi_model`'s explicit-tempo subsets, and `predict.py` processes only those.
- `all`: every score.

`evaluate.py` filters `predictions.jsonl`, so you can switch modes with `TEMPO=<mode> docker compose run --rm
rubricnet python evaluate.py` without re-predicting, as long as the earlier run covered the scores. All modes write
the same output files (the CSVs list only the selected scores), and the `RESULTS.md` title names the mode.

Feature extraction takes 3–30 s per score (median ~6 s). A run over all 806 scores takes roughly 20 minutes with 4
workers.
The run can be resumed. Each finished score is appended to `results/predictions.jsonl`, and a re-run skips scores
that already succeeded and retries the ones that failed. To start from scratch, delete `results/`.

## Output (`results/`)

- `RESULTS.md`: for the selected scores (see `TEMPO`), the main table (balanced acc9, acc ±1, balanced acc ±1, acc9, MSE) for RubricNet and for each
  fold model alone, the RubricNet confusion matrix with N, acc9 and MSE per grade, and the failed scores
- `metrics.json`: the same numbers in machine-readable form
- `labels_vs_predictions.csv`: one row per score with `file_name`, `composer`, `song_name`, `label` (1–9 from the
  xlsx), `pred_rubricnet` and `pred_fold_0` … `pred_fold_4` (all on the same 1–9 scale), `rubricnet_mean` (the
  unrounded mean of the fold predictions), `rubric_score_0_12` (the model's summed rubric score, mean over folds,
  scaled to 0–12 as upstream does), `score_has_tempo` and `bpm`
- `features.csv`: the 12 descriptors per score (RubricNet's input)
- `predictions.jsonl`: raw per-score records: features, per-fold class and rubric score, chosen staves, extractor
  warnings, input fixes and errors

## How it works

RubricNet is a small interpretable model: each of 12 hand-crafted descriptors goes through its own `tanh(w·x + b)`,
the 12 contributions are summed into one rubric score, and an ordinal head turns that score into one of 9 classes.

- Descriptors (per hand, right `rh_` / left `lh_`): `pitch_set_lz` (Lempel–Ziv complexity of the sequence of
  simultaneous pitch sets), `pitch_range`, `average_pitch`, `average_ioi_seconds` (mean time between onsets, so it
  depends on tempo), `displacement_rate` and `pitch_entropy`.
- `predict.py` extracts them in a child process per score with upstream's own code: `CIPIFeatureExtractor` from
  `extractor/DifficultyFeatures` and `difficulty_descriptors` from upstream `predict.py`. A score without any
  metronome mark is read at 100 bpm, as in upstream `predict.py` and the CIPI feature extraction (`cipi.sh -t 100`).
  Unlike upstream `extract_features`, a failure is reported as an error instead of being swallowed.
- The 5 camera-ready fold models (`checkpoints/rubricnet_cameraready/split_<k>.ckpt`, each with its
  `StandardScaler` `scaler_<k>.pkl`) are loaded once through upstream `RubricnetSklearn.load_model`, as in upstream
  `load_model_and_predict`. Each fold model returns an ordinal class 0–8 (upstream `_prediction2label`) and a rubric
  score (upstream `predict_regression` + `scale_to_0_12`).
- `rubricnet` = the mean of the 5 fold classes. `evaluate.py` shifts the dataset labels 1–9 to classes 0–8 and rounds
  the mean half-up to a class, the same way `cipi_model` handles its fold averages. Upstream `predict.py` returns the
  folds separately and doesn't combine them.
- Upstream's reference check passes in this setup: `inven01.musicxml` gives exactly the descriptor values hard-coded
  in upstream `check_features_extractor()`.

## Fixes to upstream input parsing

- **music21 arpeggio bug**: `patches/music21-arpeggio-spanner.patch`, applied in the `Dockerfile`. The pinned
  music21 fork (May 2023) passes the arpeggio type positionally to `ArpeggioMarkSpanner`, so a multi-chord arpeggio
  with a direction (up/down) makes music21 fail to parse the whole score ("'str' object has no attribute 'sites'").
  The patch is the same two-line fix that current music21 has (keyword argument, default type `'normal'`).
  134 scores contain such spanners; the ones that parsed before the patch give identical descriptors with it.
- **Input clean-up** (`prepare_score` in `predict.py`, same as in `cipi_model`). It works on a temp copy of a score;
  scores that need no fix are copied byte for byte. Fixes applied to a score are listed in `input_fixes` in
  `predictions.jsonl`.
  - Chord symbols (`<harmony>`) are dropped. music21 fails on some of them (`score_263`: "degree-value missing").
    They carry no playable notes; for scores that parse with them (e.g. `score_406`) the descriptors are identical.
  - Parts with no pitched notes are dropped (`score_40` has an empty duplicate "Piano" part, and music21 fails to
    convert the score). Tempo marks at the start of a measure in a dropped part move to the same measure of the first
    remaining part. `score_40`'s only metronome mark (quarter = 90) is in the empty part.
  - If a score has no time signature at all (`score_792`), an invisible 4/4 is added. Otherwise the extractor raises
    "No time signature".
  - Empty `<fingering/>` elements are dropped (`score_142`). music21 reads one as finger `None`, and the extractor
    fails sorting it next to numbered fingers. Fingerings don't enter the 12 descriptors.
  - Rest display positions that aren't a pitch are dropped (`score_748`: `<display-step>?</display-step>`), because
    music21 fails to parse them. They only set where a rest is drawn.
  - 30 scores are modified in total. A local check showed that every score parses and gets all 12 descriptors.
- **Choice of hands** (`StaffSelectingExtractor` in `predict.py`). Upstream takes the first staff as the right hand
  and the second as the left hand. That holds for 2-staff scores (798 of 806 after the clean-up above), which are
  unaffected. With more than
  two staves, staves holding less than 10% of the notes are skipped first, then the first two remaining are used.
  This changes 3 scores: `score_780` (a one-staff ossia part comes first), `score_456` and `score_789` (3 or 4 staves
  with a near-empty top staff or top two staves). For `score_269`, `score_769` and `score_792` (a nearly empty third
  staff) and `score_530` and `score_545` (a second part with an ossia or a duplicate "Reverb" playback copy), the result
  is the same as upstream. The chosen staves are recorded in `staves_used` / `notes_per_staff`.
- Scores whose repeats music21 can't expand (e.g. `score_11`, a D.C. that has no clean unfolding) are
  analysed without unfolding, as upstream does; this is recorded in `cant_expand_repeats`.

## Notes

- The image is CPU-only (`python:3.10-slim`, `torch==2.3.1`). Package versions match the checkpoints
  (scikit-learn 1.3.2 for the scalers, lightning 2.3.1 for the model checkpoints). Upstream `requirements.txt`
  is unpinned and includes jupyter, so `requirements.txt` here is a pinned subset.
- music21 is upstream's fork `adhooge/music21`, pinned to commit `21665005` (`MUSIC21_COMMIT` build arg) and
  patched as above. The upstream repo is cloned at a pinned commit (`RUBRICNET_COMMIT` build arg).
- The model was trained on CIPI (652 Henle-graded classical pieces). Our dataset overlaps little with it (no matching
  titles apart from generic ones like "Prelude"), but its grade distribution differs: CIPI is centred on grades 5–7,
  ours on 2–3.
- With `TEMPO=all`, 245 of the 806 scores have no metronome mark that music21 can see (printed or playback-only),
  so they're read at 100 bpm. `score_has_tempo` and `bpm` are in `labels_vs_predictions.csv`.
- Debugging without Docker: a Python venv with torch, torchvision, lightning, wandb, pandas, scikit-learn, scipy,
  seaborn, lxml, sortedcontainers, openpyxl and the patched music21 fork runs `predict.py` as is. Set
  `RUBRICNET_DIR` to a clone of the upstream repo, `PYTHONPATH=$RUBRICNET_DIR:$RUBRICNET_DIR/extractor/DifficultyFeatures/code`,
  and `SCORES_DIR` / `RESULTS_DIR`. For `evaluate.py`, also set `DATASET`.
- The upstream license is MIT.
