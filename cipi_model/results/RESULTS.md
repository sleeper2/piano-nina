# CIPI model results on the 806-score dataset

Model: pretrained CIPI models from https://github.com/PRamoneda/difficulty-prediction-CIPI (Ramoneda et al., ESWA 2023), used without retraining.
Ground truth: `806_2_score_dataset_clean.xlsx` difficulty 1-9, shifted to CIPI classes 0-8.

Scores: 806 in dataset, 806 predicted, 0 failed. VirtuosoNet failed on 0 of the predicted scores (the ensemble then averages the remaining two models, as upstream does).

## Main result

| Model | N | Balanced acc9 | Acc ±1 | Balanced acc ±1 | Acc9 | MSE |
|---|---|---|---|---|---|---|
| **Ensemble (p + argnn + virtuoso)** | 806 | 0.318 | 0.778 | 0.743 | 0.361 | 1.63 |
| Pitch (p) | 806 | 0.296 | 0.748 | 0.745 | 0.305 | 1.97 |
| Fingering ArGNN (argnn) | 734 | 0.212 | 0.604 | 0.545 | 0.200 | 3.30 |
| VirtuosoNet (virtuoso) | 806 | 0.314 | 0.734 | 0.765 | 0.299 | 2.11 |
| argnn, RH length fixed (not upstream) | 806 | 0.214 | 0.577 | 0.610 | 0.234 | 3.36 |

- **Balanced acc9**: mean per-class recall over the 9 difficulty classes (the paper's main metric; paper reports 39.5% for its best model on CIPI).
- **Acc ±1**: share of scores whose predicted class is within one level of the label (as in upstream `compute_metrics`).
- **Balanced acc ±1**: the same ±1 hit rate averaged per true class, so rare classes weigh equally.
- **MSE**: mean squared difference between predicted and true grade, over all scores. A miss by 1 grade counts 1 and a miss by 2 grades counts 4, so large misses dominate.
- Model outputs (mean of 5 fold predictions, and of 3 models for the ensemble) are rounded half-up to the nearest class.

## Ensemble confusion matrix (rows = true, cols = predicted, dataset levels 1-9)

Acc9 per grade = correctly predicted / N for that true grade; their mean is the balanced acc9. MSE per grade = mean squared distance between predicted and true grade for that grade; the mean row is their average (a balanced MSE).

| true \ pred | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | N | Acc9 | MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 10 | 20 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 34 | 0.294 | 1.06 |
| **2** | 28 | 95 | 16 | 9 | 1 | 0 | 0 | 0 | 0 | 149 | 0.638 | 0.60 |
| **3** | 8 | 47 | 27 | 27 | 15 | 17 | 3 | 0 | 0 | 144 | 0.188 | 2.55 |
| **4** | 0 | 8 | 10 | 21 | 26 | 44 | 8 | 1 | 0 | 118 | 0.178 | 2.81 |
| **5** | 0 | 1 | 2 | 6 | 16 | 75 | 22 | 0 | 0 | 122 | 0.131 | 1.52 |
| **6** | 0 | 0 | 0 | 1 | 7 | 86 | 28 | 4 | 0 | 126 | 0.683 | 0.44 |
| **7** | 0 | 1 | 1 | 1 | 3 | 22 | 29 | 5 | 0 | 62 | 0.468 | 1.44 |
| **8** | 0 | 0 | 0 | 0 | 1 | 8 | 9 | 7 | 0 | 25 | 0.280 | 2.00 |
| **9** | 0 | 0 | 1 | 0 | 0 | 1 | 14 | 10 | 0 | 26 | 0.000 | 4.27 |
| **mean** |  |  |  |  |  |  |  |  |  | 806 | **0.318** | **1.85** |
