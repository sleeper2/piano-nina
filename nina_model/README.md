# nina_model: inference with the ONSET difficulty model

`onset_colab_train_806_grid_2.py` is the training script. The 5 fold checkpoints in `checkpoints/` were
trained on Colab with `--transpose 3 --features core+timing --hidden 64 --batch-size 32`.

- `model.py`: inference only. `DifficultyModel("checkpoints").predict([(musicxml_path, start_bpm), ...])`
  takes one `(file, start tempo in quarter-note bpm)` pair per file of a score. It encodes each file as
  in training, concatenates the files' event matrices in order and returns the mean of the fold grades
  (1–9, rounded half up) plus each fold's grade.
- `evaluate_train_data.py`: grades the scores in `train_data/index.json` and compares them with the `henle`
  labels. A score is used only if every one of its files has a printed numeric metronome mark at its
  very start (time 0 of the first measure); that mark is the file's start tempo. Playback-only tempos
  (`<sound tempo>`), tempo words without a number and marks later in the piece don't count. Later tempo
  marks in a file are honoured, as in training.

## Run

```bash
cd nina_model
docker compose build
docker compose run --rm nina            # = python evaluate_train_data.py
```

Without Docker: `pip install torch numpy music21`, then `python evaluate_train_data.py [--workers 4]`.

Outputs in `results/`:
- `train_data_predictions.csv`: one row per score, with the start bpm of each file, label, predicted
  grade, fold grades, and the skip reason when the score wasn't graded
- `train_data_metrics.json`: weighted accuracy (balanced: mean per-grade exact rate), weighted
  accuracy ±1, plain accuracy and accuracy ±1, per-grade recall, and the confusion matrix
