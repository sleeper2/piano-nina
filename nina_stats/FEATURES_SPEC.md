# Piano score difficulty: hand and coordination features

This spec merges the two feature requests ("Additional Feature Extraction Specification" and "Written bimanual
coordination complexity") into one. It lists only what is implemented. Column names are the ones in the output
files.

- Group A is implemented in `nina_stats/coordination.py`.
- Groups B–E are implemented in `nina_stats/hand_features.py`.

Results are in `results/COORDINATION.md` (A) and `results/HAND_FEATURES.md` (B–E).

## 1. Objective

The goal is interpretable features of the physical and coordination demands of a piano score, taken from the
MusicXML alone, and a test of whether they predict the annotator's grade (1–9) beyond length, density and tempo. All
806 scores of the dataset are used. Every feature exists at piece level, and the countable ones also exist per
measure.

## 2. Input and note events

Each MusicXML file is parsed once (`extract.parse(path, exact=True)`) into note events with these fields:

| Field | Meaning |
|---|---|
| part, staff, voice | from the MusicXML |
| onset, duration | **exact fractions of a quarter note** (MusicXML `divisions` applied with exact arithmetic), so tuplets keep exact positions |
| pitch | sounding MIDI pitch from `<step>`, `<alter>`, `<octave>` (accidentals are already resolved in `<alter>`) |
| attack | false for a tied continuation (`<tie type="stop">`) |
| measure | index of the measure that contains the onset |

- **Excluded:** grace notes, rests and unpitched notes produce no events.
- **Chords and voices:** chords (`<chord/>`), `<backup>`, `<forward>`, multiple voices and simultaneous notes across
  staves are handled through the time cursor.
- **Onset groups:** onsets are compared by exact equality, so an onset group is all attacks of one hand at one onset,
  and a chord counts once.
- **Measures:** each measure's length is the longest voice in any part. Measure starts are the cumulative sum of those
  lengths.
- **Tempo:**
  - Seconds come from the playback tempo stored in `<sound tempo>`. Before the first tempo event the tempo is
    MuseScore's default of 120 qpm.
  - A tempo is treated as reliable when the score has a printed metronome mark (561 scores).
  - Every per-second rate also exists per measure and per quarter note, for scores without a reliable tempo.

## 3. Hand assignment

- **Right and left hand:** the staves holding notes are taken in score order. The top staff is the right hand (RH)
  and the bottom staff the left hand (LH).
- **Extra staves:** any further staff, including a staff with less than 10% of the attacks, joins the hand whose
  mean pitch is closer.
- **Confidence:** each score gets a `hand_confidence` value:

  | Value | Meaning | Scores |
  |---|---|---|
  | `staff` | exactly two staves, and the top one is higher | 796 |
  | `heuristic` | the extra-staff or inverted-staff rule was used | 8 |
  | `none` | fewer than two staves with notes | 2 |

  For the 2 `none` scores, every hand feature is missing (empty), not zero.

## 4. Group A: written bimanual coordination (`coordination.py`)

The two hands' **onset groups** are merged in time order. Each onset is labelled **L** (left only), **R** (right
only) or **B** (both hands).

### A1. Onset alignment

- **Simultaneous onset ratio:**
  - `cross_hand_simultaneous_onset_ratio` = #B / (#L + #R + #B);
  - `_lh` = #B / LH onset groups;
  - `_rh` = #B / RH onset groups.
- **Onset alignment within a tolerance** (`cross_hand_onset_alignment_tol0`, `_tol005`, `_tol010`): the share of all
  onset groups of both hands that have an onset of the other hand within 0, 0.05 or 0.10 quarter notes.
- **Per measure:** `simultaneous_groups` and `nonsimultaneous_groups`, summarised per piece as
  `simultaneous_onset_groups_per_measure_*` and `nonsimultaneous_onset_groups_per_measure_*`.

### A2. Alternation

- **Alternation:** a step from an L group to an R group, or from R to L. B groups break the chain and are not counted.
- **Totals and rates:**
  - `cross_hand_alternations` (count);
  - `cross_hand_alternation_rate_per_measure`;
  - `cross_hand_alternation_rate_per_quarter`.
- **`cross_hand_alternation_switch_share`** = alternations / (onset groups − 1).
- **`cross_hand_alternation_longest_run`**: the most onset groups in one unbroken alternating chain.
- **Per measure:** `alternations` (counted in the measure of the second group), summarised per piece as
  `alternations_per_measure_*`.

### A3. Rhythmic independence

These are computed per measure from each hand's set of onset positions.

- **One-hand and both-hand shares:**
  - `one_hand_share` = positions used by one hand only / positions used by either;
  - `both_share` = 1 − `one_hand_share`.
- **Similarity:** `similarity` = |R ∩ L| / √(|R| · |L|). This is the cosine (Ochiai) similarity of the hands' binary
  onset patterns, so it doesn't depend on measure length or time signature.
  - Dissimilarity = 1 − similarity.
  - It is defined only when both hands play in the measure.
- **Piece-level summaries:** `cross_hand_rhythm_independence_{one_hand_share,both_share,similarity,dissimilarity}_*`.
- **Onset density difference:** `onset_density_diff` = (RH − LH onset groups) / (RH + LH onset groups), and
  `onset_density_diff_abs`.
- **Onset groups per hand:** `rh_distinct_onsets` and `lh_distinct_onsets` (counts).
- **Inter-onset intervals:** IOI = the gap between a hand's successive onset groups, in quarter notes.
  `{rh,lh}_ioi_{mean,median,std,p95,max,cv}`, where CV = std / mean.

### A4. Subdivision mismatch

- **Grid** of a hand in a measure: the smallest subdivision of the quarter note that holds all its onset positions.
  It is the lcm of the exact denominators: 1 = quarter, 2 = eighth, 3 = triplet eighth, 4 = 16th, 6 = 16th triplet,
  and so on. Positions are never rounded. Per measure: `rh_grid`, `lh_grid`.
- **Mismatch measures**, over the measures where both hands play:
  - `grids_differ`: the two grids are unequal;
  - `cross_rhythm`: neither grid divides the other, e.g. 3 against 4;
  - `subdivision_gap` = |log2(grid R / grid L)|.
- **Piece-level features:**
  - `cross_hand_subdivision_mismatch_grids_differ_share`;
  - `cross_hand_subdivision_mismatch_cross_rhythm_share`;
  - `cross_hand_subdivision_mismatch_gap_*`;
  - `{rh,lh}_grid_{median,max}`.

### A5. Onset density and simultaneous sounding

- **Onset density per hand:** `right_hand_onsets_per_measure`, `left_hand_onsets_per_measure`.
- **Density ratio:** `cross_hand_onset_density_ratio` = the higher / the lower hand's onset groups.
  - Convention: when one hand has no onsets, the ratio is missing, and the case is counted by
    `single_hand_measure_share` (measures where exactly one hand has onsets).
  - Per measure: `density_ratio`, summarised as `density_ratio_per_measure_*`.
- **Sounding time:** a hand is sounding while a written note lasts (tied continuations included) and resting
  otherwise. This is how rests enter the analysis.
  - `cross_hand_simultaneous_activity_ratio` = share of the written timeline with both hands sounding.
  - `rh_rest_share` and `lh_rest_share` = share of the timeline during which that hand is silent.
  - Per measure: `both_sounding_share` (summarised as `both_sounding_share_per_measure_*`), `rh_rest_share`,
    `lh_rest_share`.
- **Simultaneous onset vs. simultaneous sounding:** the simultaneous-onset ratio (A1) and the simultaneous-sounding
  ratio are kept separate.

### A6. Share of measures beyond thresholds

| Feature | Measures with … |
|---|---|
| `share_measures_alternations_ge4` | 4 or more alternations |
| `share_measures_one_hand_share_ge0.5` | `one_hand_share` ≥ 0.5 |
| `share_measures_similarity_lt0.5` | `similarity` < 0.5 |
| `share_measures_density_ratio_ge3` | `density_ratio` ≥ 3 |
| `share_measures_both_sounding_share_lt0.25` | `both_sounding_share` < 0.25 |

## 5. Group B: pitch span (`hand_features.py`)

- **B1. Span of everything struck at one onset:** for each onset (both hands together), span = highest − lowest MIDI
  pitch.
  - `simultaneous_pitch_span_*`;
  - `simultaneous_pitch_span_ge{12,19,24}` = share of onsets with a span ≥ the threshold.
  - This is not a single-hand stretch.
- **B2. Span within one hand:** the same span for each hand's onset groups.
  - `hand_pitch_span_*` (both hands pooled), `rh_pitch_span_*`, `lh_pitch_span_*`;
  - `_gt{12,19,24}` = share of groups with a span above the threshold, i.e. the rate per attack.
- **B3. Large spans:** `large_hand_span_attacks_gt{12,19,24}` (count) and
  `large_hand_span_attacks_per_measure_gt{12,19,24}`.

## 6. Group C: black keys

A black key is a sounding pitch class in {1, 3, 6, 8, 10}, so C♯ and D♭ are the same key. Tied continuations are
excluded.

- **`black_key_note_ratio`** and its `_rh` and `_lh` versions: black-key notes / notes.
- **`black_key_attack_ratio`** and its `_rh` and `_lh` versions: the share of a hand's onset groups with at least one
  black key.
- **Counts and rates:** `black_key_notes` (count), `black_key_notes_per_second`, `black_key_notes_per_measure`,
  `black_key_notes_per_quarter`.

## 7. Group D: hand displacement

- **D1. Displacement:** for each hand, each onset group is placed at its median MIDI pitch. Displacement is the
  absolute change between successive groups.
  - `hand_displacement_semitones_*` (both hands pooled), `rh_displacement_*`, `lh_displacement_*`;
  - `_gt{7,12,19}` = share of transitions above the threshold.
- **D2. Centroid version:** the same with the mean pitch of each group, `hand_displacement_centroid_*`.
- **D3. Large leaps:**
  - `large_hand_leaps_gt{7,12,19}` (count) and `large_hand_leaps_per_measure_gt{7,12,19}`. The rate per transition is
    the D1 share.
  - `longest_large_leap_run_gt12` = the longest run of consecutive leaps over 12 semitones within one hand.
- **D4. Displacement rate:** total displacement divided by duration, as `hand_displacement_per_second`,
  `_per_measure` and `_per_quarter`.

## 8. Group E: hand crossing (proxy)

These are counted at onsets where both hands attack.

- **Crossing:** the LH median pitch is above the RH median pitch. Features: `hand_crossing_count` (count),
  `hand_crossing_per_measure`, `hand_crossing_per_minute`.
- **Overlap:** the LH top note is above the RH bottom note. Features: `hand_overlap_count` (count),
  `hand_overlap_per_measure`.

## 9. Aggregation, units and missing values

- **Distribution statistics:** each distribution feature has six statistics, `*_mean`, `*_median`, `*_std`, `*_iqr`,
  `*_p95` and `*_max`. The IOI features have five, without the IQR, plus a CV.
- **Units:**
  - pitch in semitones;
  - time in quarter notes, or in seconds at the stored tempo;
  - shares in 0–1.
- **Raw counts:** they are kept in their own columns, separate from the rates. They are not used in the models.
- **Missing values:** a value is missing when it can't be defined: no second hand, no transitions, or a hand with no
  onsets for a ratio. It is never filled with zero in the outputs. The models impute the training-fold median.
- **Complements:** `both_share` and `dissimilarity` are stored but not correlated, because they are exact complements
  of `one_hand_share` and `similarity` (ρ only changes sign).
- **Determinism:** the same files give the same values. Bootstrap and fold seeds are fixed.

## 10. Outputs (`nina_stats/results/`)

| File | Content |
|---|---|
| `coordination.csv`, `hand_features.csv` | one row per score: the features above, plus `measures`, `quarters` (and `seconds` for B–E) and `hand_confidence` |
| `coordination_per_measure.csv` | per measure: `rh_onsets`, `lh_onsets`, `simultaneous_groups`, `nonsimultaneous_groups`, `alternations`, `one_hand_share`, `both_share`, `similarity`, `density_ratio`, `single_hand`, `rh_grid`, `lh_grid`, `grids_differ`, `cross_rhythm`, `subdivision_gap`, `both_sounding_share`, `rh_rest_share`, `lh_rest_share`, plus the measure's start and length in quarters |
| `hand_features_per_measure.csv` | per measure: `notes`, `simultaneous_span_max`, `hand_span_max`, `hand_spans_gt12`, `black_key_notes`, `displacement_sum`, `displacement_max`, `leaps_gt12`, `crossings`, plus start, length in quarters and seconds. A transition belongs to the measure of its second attack. |
| `*_correlations.csv` | correlation results per feature |
| `*_validation.csv` | held-out comparison per model |
| `COORDINATION.md`, `HAND_FEATURES.md` | reports |
| `figures/coord_*.png`, `figures/hf_*.png` | figures |

## 11. Validation

1. **Checks:**
   - Two hand-built MusicXML test scores with known answers. The first has chords, an accidental, two hands and a
     tempo. The second has triplets against 16ths across two measures. Every listed feature matches the hand
     calculation.
   - On every run, the per-measure sums are checked against the piece totals: black-key notes, crossings,
     displacement, alternations and onsets per hand.
2. **Correlation with grade, per feature:**
   - Spearman ρ with a 95% bootstrap CI (1000 resamples) and Kendall τc.
   - Partial ρ holding note count, and partial ρ holding the five controls: note count, notes per second at the stored
     tempo, stored tempo, written duration and mean polyphony (all from `extract.py`). Partials are computed on rank
     residuals.
   - ρ on the 561 scores with a printed metronome mark.
3. **Held-out prediction:**
   - Model: ridge regression on quantile-normalised features, prediction rounded half-up to a grade 1–9.
   - Folds: 5-fold CV with **composer-disjoint** folds (no composer in both training and test). Random 5-fold is run
     for comparison.
   - Metrics: quadratic weighted κ, acc9, balanced acc9, acc ±1 and MAE, each with a 95% bootstrap CI over the
     held-out scores. Every model's difference from the baseline is paired (same resamples).
   - Models compared:
     - Groups B–E: the baseline (the five controls); baseline + each group; baseline + all groups; each group alone;
       all groups alone.
     - Group A: baseline + each of 25 named metrics, one at a time; baseline + the whole group; the group alone.

## 12. Limitations

- **Hands** come from staves. Notes written on the other hand's staff, or cross-staff passages, are assigned by staff.
- **Span** measures pitch distance, not physical stretch: wide chords may be arpeggiated or split between the hands.
- **Displacement** is pitch distance, not hand travel.
- **Crossing and overlap** are candidates: overlapping pitch ranges don't prove that the hands cross.
- **Black-key usage** depends on key, fingering and context. It is not a direct measure of difficulty.
- **The coordination features** describe the written parts, not how accurately a pianist plays them.
- **Durations** are the written length at the stored tempo. Repeats are not unfolded.
