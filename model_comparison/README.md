# model_comparison

Where the CIPI ensemble (`cipi_model`) and RubricNet (`rubricnet_model`) fail on the same scores. It uses the
560 scores in RubricNet's `results/labels_vs_predictions.csv` (TEMPO=marked) and reads only existing results,
so it needs no Docker and no re-prediction.

## Run

Needs Python with pandas, scipy and openpyxl:

```
python model_comparison/compare.py
```

## Outputs (`results/`)

- `PATTERNS.md`: key findings, error overlap, per-grade breakdown, accuracy (plain and balanced acc9 / acc3 / acc ±1)
  against one-feature baselines (note count, IOI tempo), length and
  tempo correlations, and the list of scores both models miss by 2+ grades.
- `both_models.csv`: one row per score, with the label, both predictions, signed errors, the flags `both_exact`,
  `both_wrong` and `both_off2`, the error `direction`, and length/tempo columns.
- `shared_misses.csv`: the `both_off2` rows only.

Re-run after either model's results change.
