# paper: project summary

Test projects that estimate piano-score difficulty on a shared dataset. Each project lives in its own
folder with its own code, Docker config and `README.md` (how to launch it). Common data is in the root.

## Rules for Claude

- **Keep this file up to date.** After any significant change (new project folder, new results, changed
  data, changed run procedure), update the relevant section below in the same session.
- **Never start, stop or run Docker** (no `open -a Docker`, `docker build`, `docker compose run`, …).
  The user launches containers themselves. Write the Dockerfiles, compose files and README commands, then
  tell the user what to run.
- New projects go in their own folder with a `Dockerfile`, a `docker-compose.yml` that mounts root data
  read-only (`../score_musicxml`, `../806_2_score_dataset_clean.xlsx`), and a `README.md` with launch steps.

## Shared data (root)

- `score_musicxml/`: 806 piano scores, `score_<id>.musicxml` (id 1–806, ~376 MB).
- `806_2_score_dataset_clean.xlsx`
  - sheet `dataset`: one row per score: `file_name`, `Сложность (число)` = difficulty label **1–9**,
    `composer`, `song_name`, `validator_note`, `bpm`, `fact`, `tempo_note`, `notes_count`, `measures_count`.
    Label counts: 1:34, 2:149, 3:144, 4:118, 5:122, 6:126, 7:62, 8:25, 9:26.
  - sheets `stats`, `composers`, `composer_mapping`: derived summaries and composer-name normalization.
- Data-quality notes:
  - The `fact` tempo tags are missing for scores 779–806 (all empty). They also ignore playback-only tempo
    changes: 21 scores among 1–778 are untagged but have hidden `<sound tempo>` changes (e.g. a written-out rit.).
    For tempo-based selection, use the MusicXML check in `cipi_model/tempo_subset.py`, not `fact` alone.
  - Inputs the upstream CIPI parser can't handle: chord symbols (`score_263`, `score_406`), an empty first part
    (`score_40`), no time signature (`score_792`), a D.C. loop (`score_11`). See `cipi_model/` below.

## Projects

### `cipi_model/`: pretrained CIPI models (Ramoneda et al., ESWA 2023)

- Runs the pretrained models from github.com/PRamoneda/difficulty-prediction-CIPI (pinned commit
  `e68961c`) with no retraining. Models: `p` (pitch), `argnn` (fingering GNN), `virtuoso` (VirtuosoNet) and
  their `ensemble`, plus `argnn_fixed`, a non-upstream variant that fixes the right-hand sequence length
  passed to the GRU.
- Labels 1–9 map to CIPI classes 0–8. Averaged outputs are rounded half-up to a class.
- Run: `cd cipi_model && docker compose build && docker compose run --rm cipi` (resumable; see its README).
- Outputs in `cipi_model/results/`:
  - `RESULTS.md`: full-dataset metrics (balanced acc9, acc ±1, balanced acc ±1, acc9, MSE per model), plus the
    ensemble confusion matrix with N, acc9 and MSE per grade (mean row = balanced acc9 and balanced MSE).
    `evaluate.py` regenerates it.
  - Full 806, ensemble (2026-10-24): balanced acc9 0.318, acc ±1 0.778, MSE 1.63 (balanced MSE 1.85; unrounded
    ensemble score MSE 1.54). MSE is worst for grade 9 (4.27, mean bias −1.81) and grades 3–4 (2.55, 2.81): the
    model underrates hard pieces and overrates grades 1 and 3–5.
  - `metrics.json`
  - `labels_vs_predictions.csv`: per score, label vs. predicted level on the 1–9 scale
  - `predictions.jsonl`: raw records with fold predictions, errors and `input_fixes`; used for resuming
- Explicit-tempo check: `tempo_subset.py` → `results/explicit_tempo_scores.csv` (per score: qualifies or why not).
  A score qualifies when its MusicXML has a printed numeric metronome mark in the first measure, no other tempo
  value later (printed or playback) and no printed/playback mismatch. (2026-10-29: the 100-score subset reports
  and their selection/metrics code were removed at the user's request.)
- `cipi_embeddings/` (user-provided, 1.6 GB): one folder per score, `score_<id>-<hash>/` with `embed.pt`,
  `virtuoso.pt` and `summary.json`. None of the files contain the ID; it's only in the folder name.
  `cipi_embeddings_grades.csv` maps `path` (relative to `cipi_model/`) → `score_id` → `grade`. The grades come from
  the user's 806-row table (`DATASET_TABLE` in `nina_model/onset_colab_train_806_grid_2.py`, from the "prefinal"
  workbook), keyed by the same `score_id` (1–806).
  - **These IDs are not the `score_N` of `score_musicxml/` or the xlsx.** `score_1` here is grade 9, while
    `score_musicxml/score_1` is Flight of the Bumblebee (grade 8). No full mapping between the two numberings is
    known. 13 pairs are known (same piece, same grade), as `score_musicxml` → `cipi_embeddings` ID: 779→15, 781→5,
    782→27, 784→2, 786→4, 788→25, 790→24, 793→13, 794→9, 800→10, 802→12, 804→6, 805→7.
  - Grade counts match `806_2_score_dataset_clean.xlsx` except grade 5 (123 vs 122) and grade 9 (25 vs 26): one
    score is labelled differently between the two workbooks. It isn't identified, because the IDs differ and
    note counts don't match reliably.
- `cipi_colab_train_musicxml.py` (user-provided): single-file Colab script that trains the CIPI ordinal heads
  (`p` from MusicXML; `argnn`/`virtuoso` from precomputed `<stem>-<hash>/` embeddings via `--emb-dir`) on the
  806 scores. Its `DATASET_TABLE` (IDs, grades, 5-fold buckets) is identical to the ONSET script's, and it expects
  files named `score_<id>.musicxml`.
- Status (2026-10-24): all 806 scores predicted, 0 errors.
- Upstream fixes (details in `cipi_model/README.md`):
  - `patches/pyscoreparser-dacapo-once.patch` fixes an infinite D.C. unfolding loop (`score_11` was OOM-killed,
    `exit -9`).
  - `prepare_score` in `predict.py` cleans up the input: it strips `<harmony>` chord symbols, drops parts with no
    notes (otherwise voices shift by +10 and every note goes to the left hand), and adds an invisible 4/4 when a
    score has no time signature.
  - A local scan with the upstream parser (`ScoreData`) over all 806 scores: all parse, each hand gets ≥ 2 notes,
    and only 28 scores are modified.
- Gotchas:
  - Upstream's VirtuosoNet script imports its whole training stack (`train`, `logger`, `emotion`, wandb,
    tensorboard, plotly) even for inference, so those packages must stay in `requirements.txt`.
  - Upstream `get_pred` passes RH length 1 to the argnn GRU, so upstream `argnn` often predicts -1 per fold;
    `argnn_fixed` doesn't have this problem.
  - Docker Desktop has 8 GB RAM; very long scores may OOM in the fingering GNN (retry with `WORKERS=1`).
- Debugging without Docker (cipi_model): the upstream parser and the fingering embedding run locally in a Python venv
  (torch, pretty_midi, tqdm, six, pandas, setuptools<81) with `PYTHONPATH` set to a clone of the upstream repo.
  Use `torch.load(..., weights_only=False)` with newer torch. Upstream `compute_embedding_score` swallows
  exceptions (prints them and returns `{}`). `evaluate.py` and `tempo_subset.py` only need pandas, scikit-learn
  and openpyxl.

### `rubricnet_model/`: pretrained RubricNet (Ramoneda et al., ISMIR 2024)

- Runs the camera-ready RubricNet checkpoints from github.com/PRamoneda/rubricnet (pinned commit `3ff4a93`) with
  no retraining. The model sums `tanh(w·x+b)` over 12 hand-crafted descriptors (per hand: pitch-set LZ complexity,
  pitch range, average pitch, average IOI in seconds, displacement rate, pitch entropy), then an ordinal head maps the
  sum to classes 0–8. There are 5 fold models, each with its own StandardScaler. `rubricnet` = mean of the 5 fold
  classes, rounded half-up (same convention as cipi_model); `evaluate.py` also scores each fold alone.
- Features come from upstream's own extractor (`extractor/DifficultyFeatures`, `CIPIFeatureExtractor`) on upstream's
  music21 fork `adhooge/music21@21665005`. A score with no metronome mark is read at 100 bpm, as upstream does. This
  applies to 245 of 806 scores and affects the IOI-in-seconds features.
- Tempo selection (user's choice): the default `TEMPO=marked` reports only scores where the extractor reads a
  metronome mark (printed or playback), dropping the 246 that would be read at 100 bpm and leaving 560. `predict.py`
  still predicts all 806; `evaluate.py` filters on `score_has_tempo`. `TEMPO=explicit` uses the stricter explicit-tempo check
  (`tempo.py`, a copy of `check_tempo` from `cipi_model/tempo_subset.py`), and `TEMPO=all` uses all 806.
  Re-evaluating needs no re-prediction. All modes write the same output files, and the CSVs list only the selected
  scores.
- Run: `cd rubricnet_model && docker compose build && docker compose run --rm rubricnet` (resumable, ~20 min with 4 workers). Outputs in `rubricnet_model/results/`: `RESULTS.md`, `metrics.json`, `labels_vs_predictions.csv`,
  `features.csv`, `predictions.jsonl`.
- Pinned versions matter: the scalers were pickled with scikit-learn 1.3.2 and the checkpoints with lightning 2.3.1.
  `rubricnet/sampler.py` imports torchvision, and `rubricnet.py` imports wandb.
- Fixes (details in `rubricnet_model/README.md`):
  - `patches/music21-arpeggio-spanner.patch`: the fork crashes on multi-chord arpeggios with a direction (25 scores
    fail without it). The patch applies the same 2-line fix as current music21; descriptors are unchanged on scores
    that parsed anyway.
  - `prepare_score`: strips chord symbols, empty parts (moving their tempo marks to the kept part, as `score_40`'s
    only metronome mark is in its empty part), empty `<fingering/>` (`score_142`) and invalid rest
    `display-step` (`score_748`), and adds 4/4 when there's no time signature (30 scores modified).
  - Hand choice: upstream uses staff 0 = RH and staff 1 = LH. With more than 2 staves, staves with <10% of the notes
    are skipped first, which changes only `score_456`, `score_780` and `score_789`.
- Docker run (2026-10-25, all 806 predicted, 0 failed) matches the local venv run exactly. It was made before the
  `score_40` tempo fix, so `score_40` is still recorded without a tempo. `results/RESULTS.md` (TEMPO=marked, 560
  scores): balanced acc9 0.338, acc ±1 0.780, balanced acc ±1 0.773, MSE 1.89 (balanced MSE 2.12). The CIPI ensemble on
  the same 560 gets 0.326 / 0.777 / 0.735 / MSE 1.64 (balanced 1.95). Grade 2 is predicted as 1 in 75 of 139 scores.
  On all 806 (TEMPO=all): 0.351 / 0.792 / 0.794 / MSE 1.77.
- Debugging without Docker: the extractor and models run in a venv (see the README Notes). The reference check is
  upstream's `inven01.musicxml`, whose descriptors match upstream's hard-coded values exactly.

### `model_comparison/`: shared errors of CIPI ensemble and RubricNet

- An analysis only (no Docker): `python model_comparison/compare.py` (pandas, scipy, openpyxl) reads both models'
  `results/` on RubricNet's 560 scores and writes `results/PATTERNS.md`, `both_models.csv` and `shared_misses.csv`.
  Re-run it after either model's results change. `PATTERNS.md` also has the baseline accuracy table (plain and
  balanced acc9 / acc3 / acc ±1, on 560 + CIPI 806; baselines: note count only and IOI tempo only, 5-fold CV), length/IOI-tempo correlations with labels
  and predictions, partial correlations, and bias by note-count / IOI-tempo quintile.
  IOI tempo = 60 / average IOI in seconds from RubricNet's descriptors (`ioi_tempo_avg`: mean of both hands;
  `ioi_tempo_max`: faster hand). It correlates with labels at 0.83 but with predictions only at 0.69–0.71, and the
  fastest quintile is underrated (RubricNet −1.01, CIPI −0.68): the models under-use speed.
- Findings (2026-10-25): both models are exact on 73 scores, both wrong on 256 (46%), and both off by ≥2 grades on 48.
  The signed errors correlate at 0.55, and 77% of shared misses go in the same direction. Mean bias: RubricNet −0.48,
  CIPI +0.16.
  - Grades 3–4 are mostly rated too high (predicted 5–7), and grades 7–9 too low. Grades 2 and 6 are the most reliable.
  - Within every grade, error rises with note count (Spearman 0.2–0.8): long easy scores are overrated, short hard
    ones underrated. **This is mostly an artifact of conditioning on the label**: a length-only predictor shows it even
    more (0.3–0.96). The fair test (mean error by note-count quintile, all grades pooled) shows only mild miscalibration:
    overall Spearman(error, notes) is 0.15 (RubricNet) and 0.20 (CIPI); CIPI overrates scores of ~600–1500 notes by +0.8.
  - The real finding is a length baseline: mean label per note-count decile (5-fold CV) gets acc9 0.373, balanced acc9
    0.261 and acc ±1 0.793, about as good as both models except on balanced acc9 (0.338 / 0.326). With length held
    constant, the predictions still correlate with the labels (partial Spearman 0.44 RubricNet, 0.39 CIPI), so the
    models do add information beyond length.
  - Length is strongly tied to the labels themselves: across the 560, Spearman(note count, label) = 0.82, about the
    same as the models' own correlation with the labels (RubricNet 0.82, CIPI 0.83). Note count vs. prediction is
    0.83 (RubricNet) and 0.89 (CIPI); measures vs. label is 0.67, vs. predictions 0.72 / 0.77.
  - Yohei Kato has 129 of the 560 scores, all grades 1–4.

### `nina_model/`: ONSET difficulty model (user's own), inference on the CIPI/Henle scores

- `onset_colab_train_806_grid_2.py` is the user's Colab training script. Its 5 fold checkpoints in `checkpoints/`
  were trained with `--transpose 3 --features core+timing --hidden 64 --batch-size 32` (uni-GRU, 212 inputs).
- `model.py` is inference only: the training encoder (v3, rate from IOI) copied with the dataset-specific
  `STAFF_OVERRIDE`/`SINGLE_TEMPO_SCORES` dropped. `predict([(path, start_bpm), ...])` concatenates the files'
  event matrices, and the grade is the mean of the fold grades, rounded half up. A fold's "below grade 1" counts as 1.
- `evaluate_train_data.py` grades `train_data/index.json` (652 scores, `henle` labels 1–9, paths under
  `train_data/scores/`; 44 are multi-file). A score is used only if every file has a printed numeric
  metronome mark at time 0 of its first measure (`model.start_tempo`; a stdlib scan finds 381 such scores: 112
  have no tempo at all, 101 only a playback `<sound tempo>`, ~50 a first printed mark later on, and 7 a mark
  inside bar 1 but after the downbeat). No mark in the data is hidden (`print-object="no"`). Writes `results/train_data_predictions.csv` and `train_data_metrics.json`
  (weighted = balanced accuracy).
- Run: `cd nina_model && docker compose build && docker compose run --rm nina`. Not run yet (2026-10-28).

### `nina_stats/`: dataset statistics for paper sections 3.5–3.7

- An analysis only (no Docker). `python nina_stats/extract.py` parses all 806 MusicXML files (~3 s) into
  `results/content.csv` + `metadata.csv`. `python nina_stats/analyze.py` writes `results/REPORT.md`,
  `per_grade.csv`, `correlations.csv` and 14 PNGs in `results/figures/`. It imports `load`, `accuracies`,
  `binned_baseline` and `partial_spearman` from `model_comparison/compare.py`. Needs pandas, scipy, matplotlib,
  lxml and openpyxl.
- `labels/composers.csv` (gender, era, kind) and `labels/scores.csv` (genre, form, original medium, simplified)
  were labelled by LLM agents from names, titles and MusicXML headers, then spot-checked. Gender is filled only when
  documented. Yohei Kato's gender is not source-checked.
- Checks: extracted notes + grace notes match the sheet's `notes_count` on 98.6% of scores, and measures match
  100%. Onsets per minute at the stored tempo correlates 0.93 with RubricNet's IOI tempo on the 560.
- Findings (2026-10-25, all 806):
  - Length vs. grade: note count ρ 0.80 / τc 0.67 (560: 0.82 / 0.68), measures ρ 0.65 / τc 0.50, against CIPI's
    τc 0.48. Note density (notes per second at the stored tempo) is the strongest correlate, ρ 0.87 / τc 0.74, with
    partial ρ 0.63 when note count is held.
  - Composer-only predictor (5-fold CV): acc9 0.419 on the 356 scores whose composer was seen (CIPI 0.399), but only
    0.300 without Yohei Kato. ω² is 0.79 (0.61 without Kato).
  - 505 composers, of which 429 are singletons. Genres: contemporary piano 264, ragtime 184, classical 134. 68% are
    original piano works, 28% arrangements and 2% simplified (a lower bound; 62 scores are unclear). Documented
    gender: 30 female-composer scores (3.7%).
  - Piano key usage (REPORT.md 3.6, from `results/key_counts.csv`, attacks per MIDI key 21–108, written by
    `extract.py`): all 88 keys are used. The top keys are C4 and G3 (3.5% each), then C5, G4, D5. C3–B5 holds 80% of
    attacks and black keys 32%. The central 90% range grows from B2–E5 (grade 1) to E2–F♯6 (grade 9), mostly upwards.
    The black-key share goes from 21% to 42%.
  - Hand and coordination features: a single spec, `nina_stats/FEATURES_SPEC.md` (implemented features only),
    merges the user's two specs. Group A (written bimanual coordination) is in `coordination.py` →
    `COORDINATION.md`; groups B–E (span, black keys, displacement, crossing) are in `hand_features.py` →
    `HAND_FEATURES.md`. Tests: `python nina_stats/tests/test_features.py`.
  - Held out, with composer-disjoint CV and QWK, the baseline (5 controls) gets 0.831. Adding displacement (D) gives
    +0.042 [+0.028, +0.057] and MAE −0.06. Adding B–E together gives +0.040; span, black keys and crossing each give
    about +0.02.
  - Group A as a whole gives +0.037 [+0.022, +0.055]. No single metric adds more than +0.018 (LH median IOI, RH IOI
    CV, non-simultaneous onset groups per measure). Alternation and simultaneity have partial ρ about 0. Hands sit
    on different grids in 57% of measures where both play; true cross-rhythms are 1.4%.
  - The baseline gets balanced acc9 0.393 with composer-disjoint CV, higher than the CIPI ensemble's 0.318.
  - `extract.parse(exact=True)` returns Fraction onsets; the float path (default) is unchanged.
