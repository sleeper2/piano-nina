# Where CIPI ensemble and RubricNet both fail

Scores: the 560 that RubricNet reports (TEMPO=marked); CIPI ensemble predictions for the same scores.
"Both wrong" = neither model predicts the exact grade; "both off ≥2" = both miss by 2+ grades.

- Both exact: 73; both wrong: 256 (46%);
  both off ≥2: 48.
- Error correlation (signed, per score): 0.55. When both are wrong,
  they err in the same direction in 77% of cases.
- Mean signed error: RubricNet -0.48, CIPI +0.16.
- Both compress the scale: grades 3–4 are mostly rated too high and grades 7–9 too low (see Per grade).
- A one-feature baseline (note count or IOI tempo) matches or beats both models on plain acc9 / acc3 / acc ±1;
  the models win on the balanced metrics (see Accuracy against one-feature baselines).
- The labels correlate with note count at 0.82 and with IOI tempo at
  0.83. The models follow length at least as closely as the labels do
  but under-use tempo; with length held constant they still correlate with the labels (see Length and tempo).

## Error overlap (|predicted − true|, 4+ grouped)

| RubricNet |err| | 0 | 1 | 2 | 3 | 4+ | All |
|---|---|---|---|---|---|---|
| 0 | 73 | 75 | 16 | 4 | 0 | 168 |
| 1 | 110 | 102 | 51 | 6 | 0 | 269 |
| 2 | 18 | 39 | 22 | 5 | 0 | 84 |
| 3 | 8 | 8 | 5 | 6 | 1 | 28 |
| 4+ | 0 | 2 | 4 | 2 | 3 | 11 |
| All | 209 | 226 | 98 | 23 | 4 | 560 |

## Per grade

`both over` / `both under` / `opposite` count the both-wrong scores by direction of the two errors.
`median_notes` and `median_ioi_tempo` (note onsets per minute, see Length and tempo) are over all scores of the grade.

| grade | N | both_exact | both_wrong | median_notes | median_ioi_tempo | both_wrong_share | both_off2 | mean_pred_rubricnet | mean_pred_cipi | both over | both under | opposite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 9 | 3 | 154 | 35 | 0.09 | 0 | 1.09 | 1.84 | 3 | 0 | 0 |
| 2 | 139 | 33 | 30 | 198 | 59 | 0.22 | 0 | 1.54 | 2.07 | 7 | 16 | 7 |
| 3 | 124 | 6 | 86 | 384 | 109 | 0.69 | 14 | 2.69 | 3.3 | 21 | 42 | 23 |
| 4 | 79 | 4 | 56 | 826 | 140 | 0.71 | 17 | 3.63 | 4.78 | 28 | 14 | 14 |
| 5 | 53 | 1 | 27 | 1402 | 179 | 0.51 | 6 | 4.64 | 5.55 | 13 | 6 | 8 |
| 6 | 54 | 14 | 14 | 1669 | 198 | 0.26 | 0 | 5.37 | 6.22 | 4 | 5 | 5 |
| 7 | 41 | 5 | 18 | 1675 | 231 | 0.44 | 6 | 5.76 | 6.27 | 1 | 16 | 1 |
| 8 | 20 | 1 | 9 | 1888 | 276 | 0.45 | 3 | 7.05 | 6.9 | 0 | 9 | 0 |
| 9 | 18 | 0 | 13 | 3448 | 338 | 0.72 | 2 | 7.94 | 7.06 | 0 | 13 | 0 |

## Accuracy against one-feature baselines

Can one feature do as well as the models? The "only" rows predict each score as the mean label of the
training scores in the same decile of that feature, rounded (5-fold cross-validation, so a score is never predicted
from its own label). IOI tempo is `ioi_tempo_avg` (see the next section).

Acc3 uses the buckets 1–3, 4–6 and 7–9; balanced metrics average the hit rate per true class (grades for acc9 and
acc ±1, buckets for acc3). Bold = best per column. All rows use the 560 scores except the last, CIPI on all
806.

|  | N | Acc9 | Bal. acc9 | Acc3 | Bal. acc3 | Acc ±1 | Bal. acc ±1 |
|---|---|---|---|---|---|---|---|
| Note count only | 560 | **0.373** | 0.261 | 0.714 | 0.653 | 0.793 | 0.683 |
| IOI tempo only | 560 | 0.350 | 0.247 | **0.770** | 0.723 | **0.834** | 0.750 |
| RubricNet | 560 | 0.300 | **0.338** | 0.755 | 0.700 | 0.780 | **0.773** |
| CIPI ensemble | 560 | **0.373** | 0.326 | 0.757 | 0.726 | 0.777 | 0.735 |
| CIPI ensemble | 806 | 0.361 | 0.318 | 0.758 | **0.735** | 0.778 | 0.743 |

A single feature matches or beats both models on the plain metrics; the models win on the balanced ones, where rare
grades count as much as the common grades 2–3.

## Length and tempo

Spearman correlation between length or IOI tempo and grade across all 560 scores: the dataset's own
labels next to each model's predicted grade. IOI tempo is note onsets per minute, 60 / average IOI in seconds, from
RubricNet's per-hand descriptors: `ioi_tempo_avg` averages the IOI of both hands, `ioi_tempo_max` takes the faster hand.

| feature | dataset label | RubricNet prediction | CIPI prediction |
|---|---|---|---|
| notes_count | 0.82 | 0.83 | 0.89 |
| measures_count | 0.67 | 0.72 | 0.77 |
| ioi_tempo_avg | 0.83 | 0.71 | 0.69 |
| ioi_tempo_max | 0.8 | 0.66 | 0.67 |

The labels are strongly tied to both length and IOI tempo. The predictions follow length at least as closely as the
labels do, but follow tempo less closely (the models under-use speed). For comparison, the predictions correlate with
the labels at 0.82 (RubricNet) and 0.83 (CIPI).

Do the models know more than length or tempo? Spearman between prediction and label with one feature held constant
(partial correlation on ranks):

| held constant | RubricNet | CIPI |
|---|---|---|
| notes_count | 0.44 | 0.39 |
| ioi_tempo_avg | 0.58 | 0.64 |

### Is length or tempo over-weighted?

Mean signed error (predicted − label) by quintile of the feature, all grades pooled. A model that uses the feature
as much as the labels justify has about the same bias in every row.

| note count | N | median notes_count | mean_label | bias_rubricnet | bias_cipi |
|---|---|---|---|---|---|
| Q1 shortest | 113 | 164.0 | 2.03 | -0.65 | -0.25 |
| Q2 | 112 | 248.5 | 2.46 | -0.69 | -0.15 |
| Q3 | 111 | 565.0 | 3.99 | -0.76 | 0.18 |
| Q4 | 112 | 1150.0 | 4.72 | -0.06 | 0.83 |
| Q5 longest | 112 | 2308.0 | 6.43 | -0.25 | 0.2 |

| IOI tempo | N | median ioi_tempo_avg | mean_label | bias_rubricnet | bias_cipi |
|---|---|---|---|---|---|
| Q1 slowest | 112 | 40.13 | 1.96 | -0.47 | 0.18 |
| Q2 | 112 | 80.63 | 2.74 | -0.43 | 0.25 |
| Q3 | 112 | 124.41 | 3.61 | -0.32 | 0.47 |
| Q4 | 112 | 182.3 | 4.57 | -0.18 | 0.58 |
| Q5 fastest | 112 | 254.17 | 6.73 | -1.01 | -0.68 |

### A pitfall: error vs length within a grade

Within each grade, Spearman between signed error and note count is strongly positive for both models. This does not
show that they over-use length: any predictor that uses length shows it, because the label is fixed within a grade
while the prediction moves with length. The note-count-only baseline shows it even more strongly:

| grade | N | RubricNet | CIPI | note-count-only baseline |
|---|---|---|---|---|
| 1 | 32 | 0.24 | 0.35 | 0.28 |
| 2 | 139 | 0.4 | 0.57 | 0.78 |
| 3 | 124 | 0.54 | 0.79 | 0.95 |
| 4 | 79 | 0.58 | 0.69 | 0.92 |
| 5 | 53 | 0.31 | 0.52 | 0.89 |
| 6 | 54 | 0.2 | 0.41 | 0.94 |
| 7 | 41 | 0.66 | 0.55 | 0.96 |
| 8 | 20 | 0.53 | 0.78 | 0.95 |
| 9 | 18 | 0.75 | 0.49 | 0.65 |

### What the shared misses look like

#### Grades 3–4 rated too high by both (49) vs grades 3–4 with ≥1 exact model (61)

| feature | median: both miss | median: ≥1 exact | p (Mann-Whitney) |
|---|---|---|---|
| notes_count | 1184.0 | 527.0 | <0.001 |
| measures_count | 80.0 | 43.0 | <0.001 |
| notes_per_sec | 4.575 | 4.532 | 0.482 |
| rh_pitch_range | 39.0 | 29.0 | <0.001 |
| lh_pitch_range | 44.0 | 29.0 | <0.001 |
| rh_pitch_set_lz | 127.0 | 55.0 | <0.001 |
| rh_displacement_rate | 0.633 | 0.475 | <0.001 |
| bpm | 98.0 | 100.0 | 0.743 |

#### Grades 7–9 rated too low by both (38) vs grades 7–9 with ≥1 exact model (39)

| feature | median: both miss | median: ≥1 exact | p (Mann-Whitney) |
|---|---|---|---|
| notes_count | 1473.0 | 2267.0 | 0.002 |
| measures_count | 59.5 | 102.0 | <0.001 |
| notes_per_sec | 10.061 | 8.749 | 0.008 |
| rh_pitch_range | 43.5 | 52.0 | 0.001 |
| lh_pitch_range | 43.5 | 53.0 | 0.004 |
| rh_pitch_set_lz | 172.0 | 254.0 | 0.005 |
| rh_displacement_rate | 0.664 | 0.834 | <0.001 |
| bpm | 120.0 | 110.0 | 0.74 |

## Scores both models miss by 2+ grades (48)

| id | label | pred_rubricnet | pred_cipi | composer | song_name | measures_count | notes_count |
|---|---|---|---|---|---|---|---|
| 6 | 3 | 5 | 5 | Pyotr Ilyich Tchaikovsky | The Seasons: Op. 37b XII. December | 278 | 2735 |
| 78 | 3 | 6 | 7 | Rich Brian | History | 105 | 1467 |
| 85 | 3 | 5 | 5 | Heath | PIANO SOLO Detective Conan - Mune Ga Doki Doki | 143 | 2236 |
| 108 | 3 | 6 | 6 | Wim Mertens | Lir | 485 | 5924 |
| 120 | 3 | 6 | 5 | Ramin Djawadi | Dr Ford (Advanced Piano Solo) | 83 | 1512 |
| 131 | 3 | 6 | 6 | Joe Hisaishi | Princess Mononoke | 71 | 929 |
| 140 | 3 | 7 | 5 | Katherina Ward | Midnight Snowfall | 58 | 913 |
| 184 | 3 | 1 | 5 | Pat Schweenis | Whiskey is my Lullaby | 96 | 681 |
| 219 | 3 | 6 | 6 | Ross Halkett | Snowflakes | 80 | 1120 |
| 436 | 3 | 5 | 6 | D-Block & S-te-Fan | Ghost Stories | 133 | 1302 |
| 473 | 3 | 7 | 6 | Richard Wagner | Lohengrin | 75 | 1375 |
| 575 | 3 | 5 | 6 | Mahito Yokota | Stargazing - Astral Observatory | 171 | 2295 |
| 722 | 3 | 1 | 5 | Adele | When We Were Young | 118 | 831 |
| 738 | 3 | 5 | 1 | Nathan Chapman | An Embrace | 25 | 233 |
| 116 | 4 | 6 | 6 | Shuka Saito | Papapa Ore wo Suki nano wa Omae dake ka yo OP | 47 | 1262 |
| 126 | 4 | 6 | 7 | ZUN | Last Remote | 146 | 2942 |
| 241 | 4 | 2 | 6 | Alan Parker | La complainte de Josie | 110 | 1019 |
| 312 | 4 | 2 | 2 | Bianca Washington | The Song Of Sound And Silence | 16 | 170 |
| 350 | 4 | 2 | 2 | Molly Jaynes | Transcription Project | 24 | 265 |
| 406 | 4 | 1 | 2 | Bianca Giovanella | Garota de Ipanema | 34 | 414 |
| 414 | 4 | 7 | 6 | JaydenZap | Valez Addormentato | 108 | 1000 |
| 421 | 4 | 7 | 7 | Cécile Chaminade | Barcarolle | 71 | 1317 |
| 426 | 4 | 2 | 6 | MD Steele | High Ground | 47 | 775 |
| 516 | 4 | 1 | 2 | Braden Jones | The Void | 44 | 777 |
| 564 | 4 | 7 | 7 | Richard Wagner | Elsa's Procession to the Cathedral | 113 | 1914 |
| 601 | 4 | 6 | 6 | Carl Czerny | School Of Velocity | 52 | 1195 |
| 602 | 4 | 6 | 6 | Ramin Djawadi | The King's Arrival (Piano Solo) Game Of Thrones OST Season 1 Episode 1 | 149 | 1530 |
| 610 | 4 | 2 | 6 | Ed Sheeran | Shape of You | 185 | 1888 |
| 623 | 4 | 6 | 6 | Dianna Smith | Compo | 129 | 1872 |
| 635 | 4 | 7 | 7 | Connor Roper | Null | 91 | 1282 |
| 638 | 4 | 6 | 6 | Yuki Hayashi | United States of Smash | 116 | 1374 |
| 57 | 5 | 3 | 2 | Tony Banks | Fountain of Samba-Maquis | 42 | 458 |
| 60 | 5 | 7 | 7 | Nicolas Mason | Ferric | 57 | 962 |
| 225 | 5 | 3 | 3 | Joia Araguàs | Ah! Vous dirai-je, Maman | 52 | 513 |
| 248 | 5 | 2 | 3 | Lastigs13 | Psuedo-Japan | 31 | 423 |
| 502 | 5 | 3 | 7 | Lorde | Green Light | 124 | 2470 |
| 582 | 5 | 7 | 7 | John Powell | How To Train Your Dragon Main Theme | 79 | 1497 |
| 37 | 7 | 3 | 3 | Danny Daglas | Prelude | 26 | 647 |
| 95 | 7 | 5 | 5 | Toby Fox | Spider Dance | 34 | 993 |
| 250 | 7 | 3 | 4 | Yasuaki Fujita | Title Theme | 37 | 568 |
| 485 | 7 | 2 | 2 | Carl Czerny | 10 | 8 | 167 |
| 692 | 7 | 2 | 5 | Wim Mertens | Struggle for pleasure | 88 | 1633 |
| 706 | 7 | 5 | 5 | DJ Totto | Curtain Call Memories | 50 | 656 |
| 50 | 8 | 4 | 6 | DeDuck | MiniBoss Theme 1 | 30 | 894 |
| 187 | 8 | 6 | 5 | Stanisław Warduliński | II Preludium | 44 | 766 |
| 616 | 8 | 4 | 6 | Maksym Giza | Run | 46 | 933 |
| 803 | 9 | 7 | 7 | Alexander Scriabin | Etude in D-sharp minor, Op. 8 No. 12 | 56 | 2403 |
| 805 | 9 | 4 | 3 | Alexander Scriabin | Etude in F-sharp minor, Op. 42 No. 2 | 32 | 384 |
