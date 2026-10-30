# cipi_model

Evaluates the pretrained piano-score difficulty models from
[PRamoneda/difficulty-prediction-CIPI](https://github.com/PRamoneda/difficulty-prediction-CIPI)
(Ramoneda et al., "Combining piano performance dimensions for score difficulty classification", ESWA 2023)
on the scores in `../score_musicxml/` against the labels in `../806_2_score_dataset_clean.xlsx`.
There's no retraining script here, only compiled embeddings for ArGNN and virtuoso

## Run

From this folder, with Docker running:

```sh
docker compose build
docker compose run --rm cipi                      # all 806 scores: predict.py, then evaluate.py
LIMIT=5 docker compose run --rm cipi              # smoke test on the first 5 scores
WORKERS=2 docker compose run --rm cipi            # fewer parallel scores (less RAM)
docker compose run --rm cipi python evaluate.py   # recompute metrics only
```

Environment variables: `WORKERS` (parallel scores, default 4), `LIMIT` (only the first N scores, 0 = all),
`SCORE_TIMEOUT` (seconds per score, default 3600), `TORCH_THREADS` (per score, default 1).

The run can be resumed. Each finished score is appended to `results/predictions.jsonl`, and a re-run
skips scores that already succeeded and retries the ones that failed (including scores where only
VirtuosoNet failed). A score that fails because it ran out of memory
(the fingering GNN builds a dense notes × notes graph per hand, so very long scores need GBs) can be
retried with `WORKERS=1`. To start from scratch, delete `results/`.

## Output (`results/`)

- `RESULTS.md`: the main table (balanced acc9, acc ±1, balanced acc ±1, acc9, MSE per model), the ensemble
  confusion matrix and the failed scores
- `metrics.json`: the same numbers in machine-readable form
- `labels_vs_predictions.csv`: one row per score with `file_name`, `composer`, `song_name`, `label`
  (the 1–9 difficulty from the xlsx), the predicted level per model (`pred_ensemble`, `pred_p`, `pred_argnn`,
  `pred_virtuoso`, `pred_argnn_fixed`, all on the same 1–9 scale) and `ensemble_score` (the unrounded
  ensemble value)
- `predictions.jsonl`: raw per-score records, including fold-level predictions and errors

## Scores with one explicit, consistent tempo

```sh
docker compose run --rm cipi python tempo_subset.py
```

`tempo_subset.py` checks each MusicXML file. A score qualifies when it has a printed numeric metronome mark in the first
measure, every later metronome mark has the same value, and every playback tempo (`<sound tempo>`) equals the
printed one in quarter notes per minute. It writes `results/explicit_tempo_scores.csv`: per score, whether it
qualifies and why not, next to the xlsx `bpm`/`fact`.

The check is stricter than the xlsx `fact` tags in two ways. 21 scores have no tag but contain hidden playback tempo
changes (e.g. a written-out ritardando in `score_43`). And scores 779–806 have no `fact` tag at all.

## How it works

- `predict.py` runs each score in a child process, so a crash or OOM on one score doesn't stop the run.
  It uses the upstream functions from `eswa_difficulty.compute_difficulty`:
  - `p`: piano-roll pitch model; `argnn`: piano-fingering ArGNN embedding model; `virtuoso`: VirtuosoNet
    embedding model. Each has 5 CV folds, and the fold predictions are averaged.
  - `ensemble` is the mean of the three, exactly like upstream `compute_difficulty()`. Two differences
    from upstream in how the code is run, neither of which changes the results: the fingering embedding is
    computed once and shared by `p` and `argnn`, where upstream computes it twice with identical output; and the
    VirtuosoNet script is launched with `subprocess.run` instead of `os.system`, so its error is recorded
    when it fails.
  - `argnn_fixed` is an extra, non-upstream variant, reported separately and not part of the ensemble.
    Upstream `get_pred` passes `feats.rh.shape[0]` after `unsqueeze(0)` as the right-hand length,
    which evaluates to 1, so the right-hand GRU only sees the first note. `argnn_fixed` passes the true lengths.
- `evaluate.py` shifts the dataset labels 1–9 to CIPI classes 0–8, rounds the averaged outputs
  half-up to the nearest class and computes the metrics.

## Fixes to upstream input parsing

Both the fingering models and VirtuosoNet read scores with upstream's `pyScoreParser`. Two problems
in it are worked around:

- **Endless repeat unfolding**: `patches/pyscoreparser-dacapo-once.patch`, applied in the `Dockerfile`.
  The parser unfolds repeats and D.C. jumps but never marks a D.C. as done. If "Fine" sits inside
  a 1st ending that gets skipped on the second pass (e.g. `score_11`), it jumps back forever and the process
  is OOM-killed (`exit -9`). The patch takes each D.C. jump only once, the same way repeats are already handled.
  Checked on all 806 scores: only `score_11` is affected (39 → 109 unfolded measures), and every other score
  unfolds to exactly the same length as without the patch.
- **Input clean-up** (`prepare_score` in `predict.py`). It works on the temp copy of a score; scores that need
  no fix are copied byte for byte. Fixes applied to a score are listed in `input_fixes` in `predictions.jsonl`.
  - Chord symbols (`<harmony>`) are dropped. The parser raises `ChordSymbolParseException` on some of them
    (`score_263`, `score_406`). 26 scores have chord symbols. For the 24 that parsed anyway, the parsed notes are
    identical with and without stripping.
  - Parts with no pitched notes are dropped (`score_40` has an empty duplicate "Piano" part first). The parser
    adds 10 to the voice numbers of every later part, and the models assign voice ≥ 5 to the left hand, so
    every note went to the left hand, the right hand was empty, and the fingering graph crashed.
  - If a score has no time signature at all (`score_792`, Messiaen), an invisible 4/4 is added, the MusicXML
    default, because the parser needs one. It only affects VirtuosoNet's beat and measure features.
  - Checked locally with the upstream parser on all 806 scores: all parse, and each hand gets ≥ 2 notes.

## Notes

- The image is CPU-only (`python:3.10-slim`, `torch==2.0.1`). Upstream `requirements.txt` pins CUDA wheels
  and `daal`, which don't install on arm64, so `requirements.txt` here is a trimmed subset.
- The upstream repo is cloned at a pinned commit (`CIPI_COMMIT` build arg in the `Dockerfile`).
- The upstream license is AGPL-3.0.
