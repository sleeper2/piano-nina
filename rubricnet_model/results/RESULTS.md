# RubricNet results on scores with a tempo mark

Model: pretrained RubricNet (camera-ready checkpoints) from https://github.com/PRamoneda/rubricnet (Ramoneda et al., ISMIR 2024), used without retraining.
Ground truth: `806_2_score_dataset_clean.xlsx` difficulty 1-9, shifted to CIPI classes 0-8.

Scores: 806 in dataset. 246 scores have no metronome mark (printed or playback), so the extractor would read them at 100 bpm; they are skipped, leaving 560. 560 predicted, 0 failed. 27 scores needed input clean-up (see README).

## Main result

| Model | N | Balanced acc9 | Acc ±1 | Balanced acc ±1 | Acc9 | MSE |
|---|---|---|---|---|---|---|
| **RubricNet (mean of 5 fold models)** | 560 | 0.338 | 0.780 | 0.773 | 0.300 | 1.89 |
| fold model 0 alone | 560 | 0.343 | 0.741 | 0.747 | 0.284 | 2.16 |
| fold model 1 alone | 560 | 0.262 | 0.716 | 0.677 | 0.239 | 2.26 |
| fold model 2 alone | 560 | 0.314 | 0.700 | 0.707 | 0.255 | 2.37 |
| fold model 3 alone | 560 | 0.327 | 0.802 | 0.794 | 0.375 | 1.62 |
| fold model 4 alone | 560 | 0.382 | 0.805 | 0.803 | 0.357 | 1.53 |

- **Balanced acc9**: mean per-class recall over the 9 difficulty classes (the paper reports 41.5% on CIPI, averaged over the 5 fold models on their test splits).
- **Acc ±1**: share of scores whose predicted class is within one level of the label.
- **Balanced acc ±1**: the same ±1 hit rate averaged per true class, so rare classes weigh equally.
- **MSE**: mean squared difference between predicted and true grade, over all scores. A miss by 1 grade counts 1 and a miss by 2 grades counts 4, so large misses dominate. (The paper's MSE of 1.52 is the per-class average, comparable to the mean row of the confusion matrix below.)
- Each fold model outputs an ordinal class; the main row averages the 5 classes and rounds half-up.

## RubricNet confusion matrix (rows = true, cols = predicted, dataset levels 1-9)

Acc9 per grade = correctly predicted / N for that true grade; their mean is the balanced acc9. MSE per grade = mean squared distance between predicted and true grade for that grade; the mean row is their average (a balanced MSE).

| true \ pred | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | N | Acc9 | MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 29 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32 | 0.906 | 0.09 |
| **2** | 75 | 54 | 9 | 1 | 0 | 0 | 0 | 0 | 0 | 139 | 0.388 | 0.63 |
| **3** | 20 | 53 | 20 | 16 | 8 | 5 | 2 | 0 | 0 | 124 | 0.161 | 2.08 |
| **4** | 7 | 16 | 19 | 8 | 19 | 6 | 4 | 0 | 0 | 79 | 0.101 | 2.85 |
| **5** | 0 | 4 | 9 | 9 | 16 | 10 | 5 | 0 | 0 | 53 | 0.302 | 2.09 |
| **6** | 0 | 0 | 4 | 7 | 16 | 21 | 4 | 2 | 0 | 54 | 0.389 | 1.70 |
| **7** | 0 | 2 | 4 | 3 | 6 | 11 | 9 | 5 | 1 | 41 | 0.220 | 4.51 |
| **8** | 0 | 0 | 0 | 2 | 1 | 1 | 8 | 6 | 2 | 20 | 0.300 | 2.75 |
| **9** | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 10 | 5 | 18 | 0.278 | 2.39 |
| **mean** |  |  |  |  |  |  |  |  |  | 560 | **0.338** | **2.12** |
