#!/usr/bin/env python3
"""CIPI difficulty training from MusicXML scores -- ALL-IN-ONE for Google Colab (CUDA-ready).

Trains the CIPI ordinal-regression difficulty head(s) -- the SAME architecture as
cipi_trainer/cipi/models.py (3-layer BiGRU -> context attention -> FC(9), OrdinalLoss) -- on the
806 scores of `806_prefinal_score_dataset_difficulty_bpm.xlsx`, whose labels ("Сложность (число)",
1..9) and 5-fold split are EMBEDDED below, so on Colab you only need the score files. The table is
the one of onset_colab_train_806*.py (same scores, grades and buckets): both models are trained and
tested on exactly the same 5 folds.

Everything lives in this one file (no `cipi` import, no vendored CIPI/pyScoreParser stack). It
depends only on torch / numpy / scikit-learn, all preinstalled in Colab.

What is trained
---------------
  p         (default) the 88-dim pitch representation, computed HERE from the MusicXML. The
            MusicXML -> note-sequence parser below re-implements pyScoreParser's note extraction
            and CIPI's `get_pitches` (repeat expansion, chords, grace notes, ties, hidden notes,
            trills, voice-based hand split, (onset, grace_order, -pitch) ordering, one-hot(midi-21)).
            Verified against the CIPI-computed tensors of the 688 scores of the earlier dataset: identical.
            A score written as two single-hand piano parts (or with an empty extra piano part) is
            first merged into one two-staff part, as scripts/predict_xml.py does before embedding.
  argnn, virtuoso   optional -- their embeddings need the AGPL CIPI stack (ArGNN fingering model,
            VirtuosoNet), which cannot be bundled in one file. If you have them precomputed
            (cipi_trainer/.cache/xml_embeddings/<stem>-<hash>/{embed.pt,virtuoso.pt}, as written by
            scripts/predict_dataset.py), upload that folder and pass --emb-dir; then
            --reps p,argnn,virtuoso trains all three heads and reports the ensemble. Scores without
            an embedding are left out of that head (reported as missing); the ensemble is
            scored on the test scores that every head graded.

Recipe (as cipi/train.py): Adam(lr=1e-3), dropout 0.2, OrdinalLoss (MSE over cumulative ordinal
encoding) with per-grade weights for class imbalance, grad-clip 1.0, FULL-LENGTH pieces batched by
length (bounded by --token-budget), early stopping on validation macro-MSE / Acc±1, 5 folds
(60/20/20: fold k tests on bucket k, validates on bucket k+1).

Per fold, every requested head is first trained (train part, epoch chosen on the val part); then the
fold's TEST part is graded by each head and by the ensemble (rounded mean of the heads' grades).
After all folds: the test metrics as mean ± std over the folds (np.std), as in the onset files:
  acc9   balanced accuracy over the 9 grades (mean per-grade recall),
  acc±1  balanced Acc±1 (mean over the grades of the within-one-grade rate),
  acc3   balanced accuracy over the coarse buckets 1-3 / 4-6 / 7+,
each also as a plain rate (fraction of all test scores). Written to <out-dir>/fold_results.json;
the pooled out-of-fold predictions + confusion matrix go to <out-dir>/oof_predictions.csv.
Checkpoints are saved as `<rep>_ordinal-<fold>.pt` in the same format as cipi/train.py, so
cipi.infer / scripts/predict_xml.py load them unchanged.

------------------------------------------------------------------------------------------
COLAB QUICKSTART
------------------------------------------------------------------------------------------
1) Runtime -> Change runtime type -> GPU (T4 is fine).
2) Mount Drive and make the score files available (the 806 `score_<id>.musicxml`; .xml / .mxl
   variants of the same stems are found too):
     from google.colab import drive; drive.mount('/content/drive')
3) Upload this file (or %%writefile it), then:

   # quick pipeline test: 60 scores, cut to 300 notes, 3 epochs, fold 0 only
   !python cipi_colab_train_musicxml.py --scores-dir /content/drive/MyDrive/train_data_tatiana \
        --out-dir /content/checkpoints --smoke

   # full training of the pitch head, 5 folds + OOF evaluation
   !python cipi_colab_train_musicxml.py --scores-dir /content/drive/MyDrive/train_data_tatiana \
        --out-dir /content/checkpoints

   # all three heads from precomputed argnn/virtuoso embeddings (5 folds, then mean ± std);
   # after a disconnect, re-run the same command with --resume to skip the folds already trained
   !python cipi_colab_train_musicxml.py --scores-dir ... --emb-dir /content/drive/MyDrive/xml_embeddings \
        --reps p,argnn,virtuoso

   # grade particular scores with the trained heads: local files or https links ...
   !python cipi_colab_train_musicxml.py --out-dir /content/checkpoints --predict /content/some_score.mxl,https://host/x.mxl
   # ... or a whole list from a file (one per line: <link or path> [label 1-9] [name]); results as
   # a table + metrics when labels are given + <out-dir>/eval/eval_results.csv. Works while a
   # training run is going on in another cell (it only READS the checkpoints written so far);
   # add --train to train first and grade the list with the fresh checkpoints in one go.
   !python cipi_colab_train_musicxml.py --out-dir /content/checkpoints --eval-list /content/songs.txt
   !python cipi_colab_train_musicxml.py --out-dir /content/checkpoints --eval-list /content/songs.txt \
        --eval-reps p,argnn,virtuoso --emb-dir /content/drive/MyDrive/xml_embeddings_for_colab

4) Colab storage is EPHEMERAL -- copy checkpoints to Drive to keep them:
     !cp -r /content/checkpoints /content/drive/MyDrive/

Parsed features are cached under --cache-dir (default <out-dir>/features), so re-runs skip the
parsing. Point --cache-dir at Drive to keep the cache across sessions.

Test / speed switches: --smoke (preset), --limit-scores N (only the first N dataset scores),
--max-len N (cut every score to N notes: a random window for training, the tail for
evaluation), --epochs, --folds 0, --parse-only, --resume.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
import random
import re
import sys
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

N_CLASSES = 9  # grades 1..9  <->  ordinal targets 0..8
REPS = ("p", "argnn", "virtuoso")
N_BUCKETS = 5
DATASET_TAG = "806"   # split / labels of onset_colab_train_806*.py; stored in the checkpoints (--resume checks it)
SCORE_EXTS = (".musicxml", ".xml", ".mxl")


# =========================================================================================
# 1. Dataset: labels + fold split (806 scores, 806_prefinal_score_dataset_difficulty_bpm.xlsx)
# =========================================================================================
# One line per score:  <score id>  <grade "Сложность (число)" 1..9>  <bucket 0..4>  <n_notes>  [np]
#   file      = score_<id>.musicxml (in --scores-dir; .xml / .mxl also accepted)
#   bucket    = the frozen 5-fold split of onset_colab_train_806*.py (copied verbatim, so both models
#               are trained and tested on exactly the same folds); every grade is spread over the
#               5 buckets with max - min <= 1
#   n_notes   = notes seen by the parser (informational; 0 = file not available when the table was made)
#   np        = the workbook's "не для ф-но" (not for piano) flag; kept unless --exclude-non-piano
# Fold k (k = 0..4): test = bucket k, val = bucket (k+1) % 5, train = the other three (60/20/20).
DATASET_TABLE = """\
1 9 3 0
2 9 3 0
3 9 1 0
4 8 0 0
5 9 4 0
6 8 2 0
7 9 3 0
8 9 1 0
9 9 2 0
10 9 4 0
11 9 3 0
12 7 1 0
13 9 1 0
14 8 0 0
15 7 1 0
16 9 2 0
17 8 3 0
18 9 0 0
19 7 2 0
20 8 2 0
21 9 4 0
22 9 3 0
23 9 1 0
24 8 0 0
25 9 2 0
26 9 0 0
27 9 4 0
28 8 0 1298
29 9 4 8228
30 4 1 1087
31 7 2 368
32 1 0 154
33 3 2 2652
34 2 4 760
35 3 2 608
36 3 2 485
37 4 4 500
38 2 2 514
39 6 0 1400
40 3 3 352
41 3 0 708
42 2 4 535
43 2 2 569
44 4 3 341
45 2 2 471
46 2 1 438
47 3 1 472
48 4 0 484
49 2 0 161
50 4 4 1882
51 4 3 1446
52 2 4 354
53 2 2 132
54 3 3 340
55 4 4 518
56 3 1 475
57 2 3 1282
58 3 1 560 np
59 4 0 1643
60 2 1 243
61 1 4 145
62 3 4 405
63 2 2 633
64 7 1 646
65 1 4 167
66 3 0 1108 np
67 2 3 486
68 6 4 2384
69 3 1 1605
70 2 3 388
71 1 2 177
72 4 4 2583
73 2 2 546
74 2 0 146
75 4 0 1109
76 2 4 1547
77 8 2 1285
78 1 4 155
79 2 4 202
80 9 0 751
81 2 1 745
82 7 1 2158
83 6 0 865
84 5 3 467
85 3 3 398
86 3 3 674
87 5 2 1240
88 2 1 299
89 4 1 2244
90 4 4 346
91 5 4 2173
92 3 3 986
93 2 3 709
94 2 3 181
95 5 3 1910
96 5 4 3400
97 3 4 983
98 4 1 931
99 3 2 1082
100 4 2 1512
101 7 4 3776
102 7 4 2118
103 7 4 3298
104 3 1 801
105 3 3 1258
106 3 2 1298
107 5 2 1508
108 2 0 414
109 4 4 805
110 5 0 1396
111 3 1 673
112 3 3 2323
113 4 3 1794
114 3 0 1027
115 1 4 157
116 3 4 415
117 6 3 3203
118 3 3 614
119 3 1 1183
120 2 0 970
121 3 0 949
122 7 2 1395
123 2 3 217
124 2 1 581
125 2 1 463
126 3 0 2296
127 3 3 2367
128 2 4 260
129 3 4 622
130 7 2 2516
131 3 3 3672
132 1 3 205
133 2 4 386
134 2 2 223
135 3 2 6768
136 2 1 145
137 3 1 384
138 1 0 211
139 5 4 1662
140 2 4 177
141 5 2 597
142 3 0 1542
143 4 4 1099
144 4 2 1402
145 1 2 224
146 2 3 431
147 3 2 1453
148 4 4 2134
149 3 2 849
150 2 0 173
151 3 0 305
152 5 3 1560
153 4 0 2899
154 3 3 2130
155 3 2 1770
156 2 2 1118
157 2 0 389
158 3 1 901
159 4 4 595
160 3 1 700
161 5 1 2706
162 3 3 904
163 3 2 359
164 5 0 1247
165 4 0 967
166 2 0 594
167 3 1 907
168 3 1 463
169 2 0 358
170 3 1 445
171 5 3 2623
172 3 4 446
173 5 1 1873
174 6 3 2540
175 4 3 2681
176 4 0 1814
177 3 0 639
178 8 4 6716
179 3 3 1373
180 3 4 601
181 3 0 578
182 5 0 2766
183 5 1 997
184 4 4 1028
185 4 3 496
186 4 4 1095
187 3 4 571
188 4 4 874
189 3 3 452
190 4 0 2728
191 2 2 189
192 5 2 1390
193 4 2 2226
194 2 1 187
195 1 0 121
196 5 4 2257
197 3 1 469
198 4 1 297
199 3 1 594
200 4 3 560
201 1 1 157
202 5 2 2418
203 2 4 261
204 3 3 479
205 2 2 204
206 5 4 2019
207 2 1 225
208 5 0 1716
209 4 3 995
210 8 1 1296
211 3 2 1240
212 2 2 152
213 4 3 699
214 8 2 764
215 2 1 194
216 3 0 234
217 4 0 933
218 4 1 987
219 3 1 574
220 3 0 247
221 3 0 213
222 3 1 228
223 4 1 467
224 2 0 205
225 5 3 648
226 4 0 804
227 5 2 2155
228 3 0 438
229 1 0 122
230 4 3 1216
231 2 0 197
232 3 4 318
233 4 0 1730
234 2 4 182
235 4 1 756
236 4 4 683
237 4 3 523
238 4 4 566
239 4 2 338
240 3 2 533
241 3 1 436
242 4 2 816
243 6 1 1866
244 5 4 4069
245 5 2 1826
246 3 4 1094
247 2 1 304
248 5 0 2327
249 8 1 1851
250 6 2 1098
251 2 0 233
252 5 1 554
253 4 1 397
254 2 1 250
255 3 0 558
256 2 1 204
257 4 3 1496
258 3 3 213
259 7 4 3403
260 3 2 542
261 4 4 1272
262 3 0 209
263 5 3 1290
264 3 0 1718
265 7 0 910
266 3 1 224
267 3 4 418
268 4 2 1395
269 5 0 1527
270 7 3 2362
271 3 3 326
272 3 0 559
273 2 2 355
274 6 3 1791
275 5 2 423
276 4 4 817
277 7 3 823
278 6 2 2207
279 3 2 785
280 2 4 181
281 3 1 876
282 5 3 763
283 7 3 3214
284 2 3 209
285 4 3 614
286 3 2 580
287 4 1 956
288 7 3 1849
289 6 0 2277
290 2 0 238
291 4 2 1030
292 3 0 292
293 3 3 534
294 3 2 732
295 1 3 151
296 6 4 3796
297 2 1 242
298 3 1 808
299 4 1 330
300 3 0 1844
301 8 1 1893
302 3 2 377
303 7 2 1501
304 2 4 148
305 7 4 2929
306 5 0 1539
307 5 3 398
308 3 4 240
309 5 3 1135
310 4 1 1387
311 6 1 2654
312 3 0 775
313 6 4 1600
314 1 2 131
315 4 2 1080
316 2 3 123
317 6 3 849
318 5 3 2119
319 3 1 312
320 6 4 2312
321 1 2 94
322 6 1 967
323 2 1 251
324 4 0 802
325 1 0 143
326 5 1 1028
327 4 4 83
328 4 3 1038
329 3 2 468
330 4 0 1145
331 2 2 191
332 3 0 354
333 4 1 941
334 5 0 2422
335 4 1 1190
336 4 2 2468
337 5 2 1194
338 4 1 1427
339 5 4 1488
340 3 2 118
341 2 0 178
342 7 2 613
343 4 2 1762
344 5 2 1657
345 2 0 166
346 4 3 585
347 6 3 1994
348 6 4 1815
349 3 2 186
350 4 3 1528
351 2 1 98
352 5 3 1402
353 2 0 396
354 4 2 736
355 6 2 2074
356 2 1 138
357 3 3 299
358 2 3 195
359 8 0 2070
360 4 1 558
361 6 4 1269
362 3 3 353
363 5 0 1871
364 3 4 338
365 5 1 2548
366 2 4 222
367 4 4 2120
368 4 1 1281
369 2 2 194
370 4 2 170
371 4 2 1771
372 5 3 1074
373 6 3 2726
374 6 4 1326
375 4 0 1312
376 5 2 1171
377 3 3 877
378 5 1 1130
379 3 4 591
380 2 1 202
381 5 2 1247
382 1 1 79
383 6 3 1923
384 2 1 149
385 1 0 195
386 1 3 104
387 4 0 523
388 3 4 363
389 7 1 2588
390 5 3 1961
391 4 1 1607
392 6 2 1819
393 4 2 386
394 2 3 189
395 2 4 166
396 4 2 1168
397 7 1 1564
398 2 4 250
399 1 3 144
400 3 4 291
401 6 1 737
402 4 2 1506
403 5 0 1919
404 2 1 177
405 6 1 2218
406 3 4 188
407 6 0 2385
408 2 3 251
409 3 2 166 np
410 2 0 243
411 4 1 949
412 4 3 1598
413 3 0 332
414 5 4 855
415 1 4 167
416 9 1 4529
417 7 1 924
418 2 0 202
419 2 0 165
420 6 2 1784
421 5 2 2568
422 7 1 2829
423 6 1 2911
424 1 1 247
425 4 3 264
426 5 0 973
427 5 1 1093
428 6 1 1578
429 4 3 790
430 6 1 3572
431 4 1 2079
432 6 3 1941
433 1 4 80
434 2 2 195
435 5 4 1251
436 6 1 2017
437 2 1 181
438 2 2 213
439 3 0 71
440 2 2 451
441 2 4 220
442 3 4 170
443 2 0 224 np
444 7 3 3677
445 5 4 2455
446 2 4 198
447 5 4 717
448 5 1 1413
449 7 0 2250
450 2 4 219
451 1 3 261
452 3 4 1093
453 7 2 2624
454 2 2 167
455 6 1 1330
456 2 0 196
457 5 2 1512
458 4 1 1137
459 8 3 4206
460 7 2 2232
461 3 1 362
462 2 3 274
463 3 4 168
464 6 2 2097
465 1 2 130
466 3 2 278
467 5 4 1451
468 2 0 170
469 2 4 162
470 3 3 290
471 5 1 1480
472 6 0 3191
473 2 4 210 np
474 2 1 140
475 5 1 1940
476 6 4 2056
477 5 0 1707
478 3 3 150
479 4 2 846
480 4 3 1056
481 4 2 470
482 4 3 1579
483 1 4 239
484 6 3 3200
485 2 3 349
486 2 0 179
487 5 1 1069
488 3 0 397
489 4 2 1146
490 2 0 194
491 2 0 170
492 2 4 165
493 6 0 1764
494 6 4 2763
495 3 2 288
496 4 4 1298
497 3 4 178
498 2 3 274
499 5 4 1609
500 2 4 276
501 4 2 887
502 2 2 294
503 3 2 312
504 3 3 533
505 2 3 183
506 9 0 4472
507 1 2 108
508 6 2 2066
509 6 3 4547
510 2 1 168
511 3 0 1301
512 7 1 2442
513 5 4 2561
514 3 0 434
515 5 0 1801
516 6 1 2171
517 5 4 2476
518 3 3 238
519 5 0 2453
520 5 1 3136
521 2 4 299
522 2 2 155
523 2 2 131
524 4 1 2392
525 7 4 2799
526 2 3 206
527 6 0 1909
528 2 1 164
529 2 2 267
530 5 2 2796
531 5 1 2931
532 3 2 307
533 5 3 1860
534 3 1 344
535 2 1 408
536 2 0 136
537 7 4 3113
538 5 3 1643
539 6 2 3465
540 6 3 1717
541 4 1 2266
542 6 4 2427
543 6 4 1397
544 3 1 263
545 2 2 127
546 5 3 1142
547 2 0 142
548 3 0 1297
549 3 3 375
550 7 1 1857
551 8 4 2835
552 5 3 2284
553 6 3 2934
554 3 1 151
555 2 2 181
556 5 2 1039
557 2 3 323
558 2 2 272
559 6 0 2992
560 7 0 167
561 5 4 2481
562 3 3 219
563 9 0 4742
564 8 4 1821
565 5 2 1869
566 2 3 174
567 8 2 5421
568 2 4 164
569 3 4 160
570 6 0 1791
571 6 4 1630
572 6 3 1593
573 6 3 1456
574 2 4 342
575 4 2 882
576 3 0 272
577 5 4 2085
578 6 3 2254
579 2 3 234
580 5 2 1651
581 2 1 184
582 6 2 2363
583 3 4 235
584 2 1 170
585 2 4 231
586 6 4 2790
587 2 1 178
588 2 3 124
589 3 4 280
590 3 3 194
591 4 0 777
592 6 1 865
593 3 1 348
594 2 3 162
595 3 2 396
596 2 2 146
597 3 1 383
598 3 2 330
599 5 0 1722
600 6 1 970
601 7 0 1871
602 1 3 180
603 5 0 3159
604 6 4 2273
605 6 0 1168
606 7 0 1233
607 5 1 2158
608 5 3 3177
609 5 0 5099
610 6 3 2045
611 6 0 3100
612 7 1 1598
613 6 4 2995
614 5 3 2786
615 6 2 827
616 6 4 4387
617 5 1 2832
618 4 2 684
619 5 4 1419
620 7 0 1266
621 5 4 6316
622 4 3 1976
623 5 3 2930
624 6 0 1587
625 7 3 1659
626 4 0 1667
627 6 0 2952
628 7 0 975
629 6 4 2621
630 2 1 190
631 5 0 1916
632 6 2 2572
633 6 0 2655
634 5 1 2868
635 5 2 1468
636 1 1 141
637 6 1 2245
638 6 4 2558
639 7 3 2313
640 6 0 1656
641 7 4 2888
642 4 3 1831
643 6 3 1741
644 6 2 3313
645 7 4 1996
646 7 3 3231
647 6 1 1619
648 5 3 1878
649 3 4 220
650 6 0 2825
651 7 0 1604
652 3 4 97 np
653 5 3 2846
654 6 2 2316
655 3 0 2252
656 5 0 2398
657 5 4 2469
658 2 3 408
659 6 4 1976
660 5 2 971
661 5 0 1707
662 3 4 180
663 6 2 2404
664 5 4 1486
665 6 1 2225
666 5 0 1185
667 6 3 1608
668 5 2 2142
669 6 1 1923
670 6 1 2064
671 6 3 1839
672 5 2 1815
673 3 4 547
674 6 2 2274
675 4 4 1838
676 4 4 1651
677 2 1 133
678 6 1 1762
679 7 2 2888
680 4 0 1330
681 6 2 2738
682 6 2 2237
683 5 0 2367
684 4 2 1184
685 4 2 1484
686 6 1 2700
687 6 0 2525
688 5 1 1998
689 6 3 1431
690 6 3 1877
691 5 4 2086
692 4 0 701
693 4 1 1874
694 7 0 2729
695 6 4 1554
696 6 2 1813
697 7 0 2243
698 5 0 1859
699 8 1 1023
700 6 3 1616
701 6 2 1671
702 6 0 2806
703 7 3 1406
704 8 4 1528
705 6 4 2068
706 4 4 1872
707 5 1 2703
708 6 4 1615
709 6 2 1520
710 6 2 2034
711 7 4 1919
712 4 0 1764
713 6 3 2800
714 4 0 2140
715 5 1 1923
716 5 3 1489
717 5 1 2039
718 4 4 1185
719 3 2 117
720 7 0 3707
721 6 2 2508
722 4 0 1321
723 4 3 2450
724 5 4 1872
725 6 0 1358
726 5 1 1806
727 6 0 2169
728 7 3 4712
729 1 0 140
730 6 0 1947
731 3 2 1674
732 6 4 1636
733 6 0 2049
734 6 1 4029
735 7 3 1068
736 5 2 1480
737 5 1 2607
738 4 4 1605
739 6 1 2109
740 5 3 1800
741 6 0 2442
742 4 0 4464
743 9 2 2542
744 6 0 2112
745 5 3 3471
746 8 3 4593
747 5 4 3416
748 6 0 2607
749 7 3 1497
750 7 4 870
751 8 3 525
752 6 4 2360
753 5 1 1495
754 3 4 513
755 4 0 2073
756 6 1 2784
757 5 0 1272
758 5 0 921
759 5 2 2651
760 7 1 1363
761 7 2 1654
762 4 3 1315
763 7 2 1977
764 6 2 898
765 6 3 1868
766 6 4 2512
767 6 1 1655
768 5 4 1034
769 6 3 1115
770 6 2 1753
771 7 0 2919
772 6 1 1986
773 2 2 400
774 2 2 153
775 1 1 118
776 2 0 129
777 2 3 179
778 7 4 1768
779 2 3 129
780 7 2 7748
781 8 3 3482
782 2 3 215
783 1 3 92
784 2 0 154
785 2 3 134
786 3 3 260
787 7 0 4830
788 3 1 154
789 2 3 175
790 2 4 132
791 2 4 189
792 8 1 5109
793 2 3 164
794 7 3 3169
795 1 1 83
796 2 4 126
797 9 2 5459
798 2 0 169
799 2 2 152
800 2 3 192
801 6 2 2619
802 2 2 234
803 8 4 3738
804 6 2 3001
805 1 1 203
806 2 0 132
"""


def load_dataset(args) -> list:
    """-> [{'stem', 'grade', 'bucket', 'n_notes', 'not_piano'}, ...] sorted by id."""
    items = []
    for line in DATASET_TABLE.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        sid, grade, bucket, n_notes = parts[:4]
        items.append({"stem": f"score_{sid}", "grade": int(grade), "bucket": int(bucket),
                      "n_notes": int(n_notes), "not_piano": "np" in parts[4:]})
    if args.xlsx:
        items = _apply_xlsx_labels(items, args.xlsx)
    if args.exclude_non_piano:
        items = [it for it in items if not it["not_piano"]]
    if args.limit_scores:
        items = items[: args.limit_scores]
    return items


def _apply_xlsx_labels(items: list, xlsx_path: str) -> list:
    """Optional: re-read the labels from the workbook (column "Сложность (число)"; the score file
    is the basename of "musicxml_path", or score_<id> from the "Score url" when the path is empty).
    The buckets stay those of the embedded table; workbook scores that are not in the table are
    skipped, so the folds stay identical to onset_colab_train_806*.py."""
    try:
        import openpyxl
    except ImportError:
        sys.exit("--xlsx needs openpyxl (pip install openpyxl)")
    ws = openpyxl.load_workbook(xlsx_path, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    ci, pi = header.index("Сложность (число)"), header.index("musicxml_path")
    ui = header.index("Score url") if "Score url" in header else None
    by_stem = {it["stem"]: it for it in items}
    seen, not_in_table, changed = set(), 0, 0
    for r in rows[1:]:
        if r[ci] is None:
            continue
        if r[pi]:
            stem = os.path.splitext(os.path.basename(str(r[pi])))[0]
        else:                                  # no musicxml_path yet: the score id of the URL names the file
            mid = re.search(r"/scores/(\d+)", str(r[ui] or "")) if ui is not None else None
            if not mid:
                continue
            stem = f"score_{mid.group(1)}"
        grade = int(r[ci])
        if not 1 <= grade <= 9:
            print(f"  xlsx: skipping {stem}: grade {grade} out of 1..9")
            continue
        if stem not in by_stem:
            not_in_table += 1
            continue
        seen.add(stem)
        if by_stem[stem]["grade"] != grade:
            by_stem[stem]["grade"] = grade
            changed += 1
    dropped = [it for it in items if it["stem"] not in seen]
    items = [it for it in items if it["stem"] in seen]
    print(f"labels from {xlsx_path}: {len(items)} scores ({changed} relabelled, {not_in_table} not in the "
          f"embedded table -> skipped, {len(dropped)} not in workbook -> dropped)")
    return sorted(items, key=lambda it: it["stem"])


def fold_split(items: list, fold: int) -> dict:
    test_b, val_b = fold % N_BUCKETS, (fold + 1) % N_BUCKETS
    out = {"train": [], "val": [], "test": []}
    for it in items:
        part = "test" if it["bucket"] == test_b else "val" if it["bucket"] == val_b else "train"
        out[part].append(it)
    return out


# =========================================================================================
# 2. MusicXML -> note sequence -> `p` (re-implementation of pyScoreParser + CIPI get_pitches)
# =========================================================================================
STEP_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


class ScoreParseError(RuntimeError):
    pass


def _read_score_root(path: str) -> ET.Element:
    if str(path).lower().endswith(".mxl"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            target = None
            if "META-INF/container.xml" in names:
                cont = ET.fromstring(z.read("META-INF/container.xml"))
                for rf in cont.iter("rootfile"):
                    fp = rf.get("full-path")
                    if fp and fp in names:
                        target = fp
                        break
            if target is None:
                cands = [n for n in names if n.lower().endswith((".xml", ".musicxml"))
                         and not n.startswith("META-INF")]
                if not cands:
                    raise ScoreParseError(f"{path}: no MusicXML inside the .mxl container")
                target = cands[0]
            root = ET.fromstring(z.read(target))
    else:
        root = ET.parse(path).getroot()
    if root.tag != "score-partwise":
        raise ScoreParseError(f"{path}: not a score-partwise MusicXML document ({root.tag})")
    return root


PIANO_NAME_RE = re.compile(r"piano|klavier|clavier|fortepiano|pianoforte|фортепиано|рояль|клавир|钢琴|ピアノ|\bpf\b", re.I)
PIANO_MIDI_PROGRAMS = set(range(1, 9))  # General MIDI piano family, 1-based as MusicXML writes it


def _is_piano_part(score_part) -> bool:
    text = " ".join(t for t in (score_part.findtext("part-name"), score_part.findtext("part-abbreviation"),
                                score_part.findtext("score-instrument/instrument-name")) if t)
    if PIANO_NAME_RE.search(text):
        return True
    prog = (score_part.findtext("midi-instrument/midi-program") or "").strip()
    return prog.isdigit() and int(prog) in PIANO_MIDI_PROGRAMS


def _staves_with_notes(part) -> set:
    staves = set()
    for n in part.iter("note"):
        if n.find("rest") is None and n.find("pitch") is not None:
            staves.add(int(n.findtext("staff") or 1))
    return staves


def _first_clef_sign(part):
    for m in part.findall("measure"):
        for cl in m.iter("clef"):
            return cl.findtext("sign")
    return None


def _measure_cursor(measure) -> tuple:
    """(end, furthest) cursor position of a measure's content, in its divisions units."""
    pos, furthest = 0, 0
    for ch in measure:
        if ch.tag == "note":
            if ch.find("chord") is None and ch.find("grace") is None:
                pos += int(float(ch.findtext("duration") or 0))
        elif ch.tag == "backup":
            pos -= int(float(ch.findtext("duration") or 0))
        elif ch.tag == "forward":
            pos += int(float(ch.findtext("duration") or 0))
        furthest = max(furthest, pos)
    return pos, furthest


def _normalize_divisions(part, target: int) -> None:
    """Rescale every duration / offset of the part so that <divisions> == target throughout."""
    cur = None
    for m in part.findall("measure"):
        for ch in m:
            if ch.tag == "attributes":
                d = ch.find("divisions")
                if d is not None:
                    cur = int(float(d.text))
                    d.text = str(target)
                continue
            if cur is None or cur == target:
                continue
            for el in ch.iter():
                if el.tag in ("duration", "offset") and el.text and el.text.strip():
                    el.text = str(int(round(float(el.text) * target / cur)))
        if cur is None:
            cur = 1


_NOTE_AFTER_STAFF = ("beam", "notations", "lyric", "play", "listen")


def _set_staff(el, staff: int) -> None:
    """Set <staff> on a <note> or <direction>, creating it at its schema position if absent."""
    st = el.find("staff")
    if st is None:
        st = ET.Element("staff")
        children = list(el)
        if el.tag == "note":
            idx = next((k for k, ch in enumerate(children) if ch.tag in _NOTE_AFTER_STAFF), len(children))
        else:  # direction: staff precedes <sound>
            idx = next((k for k, ch in enumerate(children) if ch.tag in ("sound", "listening")), len(children))
        el.insert(idx, st)
    st.text = str(staff)


def _lcm(a: int, b: int) -> int:
    from math import gcd
    return a * b // gcd(a, b)


def merge_piano_parts(root):
    """If the score is TWO piano parts holding one hand each (or one of them holding no notes at
    all), rewrite it in place as ONE two-staff piano part and return a one-line description;
    otherwise return None and leave the tree untouched.

    CIPI's hand split is voice-based and per part (voices 1-4 of part 0 -> right hand, everything
    else -> left hand, pyScoreParser adds 10 x part-index to every voice), so an empty first part
    sends the whole piece to the left hand and a piece written as two single-staff parts never
    forms one two-hand score. After the merge: the treble part becomes staff 1 with its voices,
    the other part becomes staff 2 with its voices shifted by 4 (5..8, the MuseScore convention);
    both parts' durations are rescaled to a common <divisions>; the second part's clefs are added
    with renumbered staves; its barlines, prints and chord symbols are dropped (the first part's
    apply). Directions of the second part are kept with their staff renumbered.
    """
    parts = root.findall("part")
    if len(parts) != 2:
        return None
    part_list = root.find("part-list")
    if part_list is None:
        return None
    score_parts = {sp.get("id"): sp for sp in part_list.iter("score-part")}
    if not all(p.get("id") in score_parts and _is_piano_part(score_parts[p.get("id")]) for p in parts):
        return None
    names = [(score_parts[p.get("id")].findtext("part-name") or "").strip() or p.get("id") for p in parts]
    a, b = parts
    staves_a, staves_b = _staves_with_notes(a), _staves_with_notes(b)
    if not staves_a and not staves_b:
        return None
    if not staves_a or not staves_b:                       # one part is empty: keep the other alone
        empty = a if not staves_a else b
        root.remove(empty)
        part_list.remove(score_parts[empty.get("id")])
        return (f"2 piano parts {names}: part {empty.get('id')} has no notes and was dropped, "
                f"the other part is now part 0")
    if len(staves_a) != 1 or len(staves_b) != 1:
        return None                                         # not "one hand each"
    if len(a.findall("measure")) != len(b.findall("measure")):
        return None
    if _first_clef_sign(a) == "F" and _first_clef_sign(b) == "G":
        a, b = b, a                                         # the treble part becomes the right hand
        names.reverse()
    n_staves_a = max(1, max(int(x.text) for x in a.iter("staves")) if a.find(".//staves") is not None else 1)
    voice_shift = 4 * n_staves_a
    staff_shift = n_staves_a
    divisions = [int(float(d.text)) for p in (a, b) for d in p.iter("divisions") if d.text]
    target = 1
    for d in divisions:
        target = _lcm(target, d)
    _normalize_divisions(a, target)
    _normalize_divisions(b, target)

    for i, (m1, m2) in enumerate(zip(a.findall("measure"), b.findall("measure"))):
        end1, far1 = _measure_cursor(m1)
        end2, far2 = _measure_cursor(m2)
        # clefs (and, in measure 1, the staff count) of the second part go into the first part's attributes
        clefs_b = [cl for attr in m2.findall("attributes") for cl in attr.findall("clef")]
        if i == 0 or clefs_b:
            attr = m1.find("attributes")
            if attr is None:
                attr = ET.Element("attributes")
                m1.insert(0, attr)
            if i == 0:
                for cl in attr.findall("clef"):
                    if cl.get("number") is None:
                        cl.set("number", "1")
                staves_el = attr.find("staves")
                if staves_el is None:
                    staves_el = ET.Element("staves")
                    anchor = [k for k, ch in enumerate(list(attr)) if ch.tag in ("divisions", "key", "time")]
                    attr.insert((anchor[-1] + 1) if anchor else 0, staves_el)
                staves_el.text = str(staff_shift + max(1, max(staves_b)))
            for cl in clefs_b:
                cl2 = copy.deepcopy(cl)
                cl2.set("number", str(int(cl.get("number") or 1) + staff_shift))
                attr.append(cl2)
        for el in m1:                                   # first part: make its staff explicit
            if el.tag in ("note", "direction") and el.find("staff") is None:
                _set_staff(el, 1)
        if end1 > 0:
            bk = ET.SubElement(m1, "backup")
            ET.SubElement(bk, "duration").text = str(end1)
        for ch in list(m2):
            if ch.tag in ("attributes", "barline", "print", "harmony", "figured-bass"):
                continue
            el = copy.deepcopy(ch)
            if el.tag in ("note", "direction"):
                _set_staff(el, int(el.findtext("staff") or 1) + staff_shift)
            if el.tag == "note":
                voice = el.find("voice")
                if voice is None:
                    voice = ET.Element("voice")
                    voice.text = "1"
                    dur = el.find("duration")
                    el.insert(list(el).index(dur) + 1 if dur is not None else len(list(el)), voice)
                voice.text = str(int(voice.text) + voice_shift)
            m1.append(el)
        furthest = max(far1, far2)
        if end2 < furthest:
            fw = ET.SubElement(m1, "forward")
            ET.SubElement(fw, "duration").text = str(furthest - end2)
    root.remove(b)
    part_list.remove(score_parts[b.get("id")])
    return (f"2 single-hand piano parts {names} merged into one part: {names[0]} -> staff 1 (right hand), "
            f"{names[1]} -> staff 2 (left hand)")


class _Note:
    __slots__ = ("midi", "onset", "voice", "grace", "grace_order", "chord_index", "tied_stop",
                 "print_object", "wavy", "measure", "duration")

    def __init__(self):
        self.midi = None
        self.onset = Fraction(0)
        self.voice = 1
        self.grace = False
        self.grace_order = 0
        self.chord_index = 0
        self.tied_stop = False
        self.print_object = True
        self.wavy = None  # (type, number)
        self.measure = 0
        self.duration = Fraction(0)


def _parse_note(el: ET.Element, state: dict, chord_state: dict):
    """One <note>; None for rests / unpitched. Moves the onset cursor like pyScoreParser."""
    n = _Note()
    n.print_object = el.get("print-object") != "no"
    is_rest = is_chord = unpitched = False
    dur_raw = None
    for ch in el:
        t = ch.tag
        if t == "chord":
            is_chord = True
            chord_state["idx"] += 1
            n.chord_index = chord_state["idx"]
        elif t == "duration":
            dur_raw = int(float(ch.text))
        elif t == "pitch":
            step, alter, octave = ch.findtext("step"), ch.findtext("alter"), ch.findtext("octave")
            if step not in STEP_TO_PC:
                raise ScoreParseError(f"bad pitch step {step!r}")
            n.midi = 12 + STEP_TO_PC[step] + (int(float(alter)) if alter else 0) + int(octave) * 12
        elif t == "rest":
            is_rest = True
        elif t == "voice":
            n.voice = int(ch.text)
        elif t == "unpitched":
            unpitched = True
        elif t == "grace":
            n.grace = True
        elif t == "notations":
            for nt in ch:
                if nt.tag == "tied" and nt.get("type") == "stop":
                    n.tied_stop = True
                elif nt.tag == "ornaments":
                    for orn in nt:
                        if orn.tag == "wavy-line":
                            n.wavy = (orn.get("type"), orn.get("number") or "1")
    if not is_chord:
        chord_state["idx"] = 0
    div = state["divisions"]
    if n.grace:                                  # grace: no duration, cursor stays
        n.onset = state["pos"]
    else:
        d = Fraction(dur_raw or 0, div) if div else Fraction(0)
        if is_chord:                             # chord note: onset of the previous note
            n.duration, n.onset = state["prev_dur"], state["prev_pos"]
        else:
            n.duration, n.onset = d, state["pos"]
            state["pos"] += d
    state["prev_dur"], state["prev_pos"] = n.duration, n.onset   # rests count as "previous" too
    n.measure = state["measure"]
    if is_rest or unpitched or n.midi is None:
        return None
    return n


def _parse_measure(m: ET.Element, state: dict) -> dict:
    out = {"notes": [], "repeat": None, "dacapo": False, "fine": False,
           "first_ending_start": False, "first_ending_stop": False}
    chord_state = {"idx": 0}
    for ch in m:
        t = ch.tag
        if t == "attributes":
            d = ch.findtext("divisions")
            if d:
                state["divisions"] = int(float(d))
        elif t == "backup":
            state["pos"] -= Fraction(int(float(ch.findtext("duration") or 0)), state["divisions"])
        elif t == "forward":
            state["pos"] += Fraction(int(float(ch.findtext("duration") or 0)), state["divisions"])
        elif t == "barline":
            rep = ch.find("repeat")
            if rep is not None:
                out["repeat"] = {"forward": "start", "backward": "jump"}.get(rep.get("direction"))
            end = ch.find("ending")
            if end is not None and end.get("number") == "1":
                if end.get("type") == "start":
                    out["first_ending_start"] = True
                elif end.get("type") == "stop":
                    out["first_ending_stop"] = True
                elif end.get("type") == "discontinue":
                    state["first_ending_discontinue"] = True
        elif t == "direction":
            for sub in ch:
                if sub.tag == "sound":
                    if sub.get("tempo") is not None:
                        pass
                    elif sub.get("dacapo") is not None:
                        out["dacapo"] = True
                    elif sub.get("fine") is not None:
                        out["fine"] = True
                if state.get("first_ending_discontinue") and sub.tag == "direction-type":
                    for b in sub:
                        if b.tag == "bracket" and b.get("type") == "stop":
                            out["first_ending_stop"] = True
                            state["first_ending_discontinue"] = False
        elif t == "note":
            n = _parse_note(ch, state, chord_state)
            if n is not None:
                out["notes"].append(n)
    return out


def _expand_part(part: ET.Element) -> list:
    """Measures in performance order: repeats / first endings / D.C. al Fine expanded, every
    jump taken at most once (pyScoreParser's Part._parse incl. its local patches)."""
    xml_measures = part.findall("measure")
    n_meas = len(xml_measures)
    state = {"pos": Fraction(0), "divisions": 1, "prev_dur": Fraction(0), "prev_pos": Fraction(0),
             "measure": 0, "first_ending_discontinue": False}
    measures, cur = [], 0
    forward_repeats, resolved_repeats, resolved_dacapo = [], [], []
    resolved_first_ending, end_of_first_ending = [], []
    fine_activated = False
    max_measures = 20 * n_meas + 100
    while cur < n_meas:
        if len(measures) > max_measures:
            warnings.warn(f"repeat expansion exceeded {max_measures} measures, stopping")
            break
        state["measure"] = cur
        old_state = dict(state)
        pm = _parse_measure(xml_measures[cur], state)
        if pm["fine"] and fine_activated:
            measures.append(pm)
            break
        if pm["first_ending_start"]:
            if cur in resolved_first_ending:
                cur = end_of_first_ending[resolved_first_ending.index(cur)] + 1
                state.clear(); state.update(old_state)
                continue
            resolved_first_ending.append(cur)
        measures.append(pm)
        if pm["first_ending_stop"]:
            end_of_first_ending.append(cur)
        if pm["repeat"] == "start":
            forward_repeats.append(cur)
            cur += 1
        elif pm["repeat"] == "jump" and cur not in resolved_repeats:
            resolved_repeats.append(cur)
            if len(resolved_first_ending) != len(end_of_first_ending):
                end_of_first_ending.append(cur)
            cur = forward_repeats.pop() if forward_repeats else 0
        elif pm["dacapo"] and cur not in resolved_dacapo:
            resolved_dacapo.append(cur)
            cur = 0
            fine_activated = True
        else:
            cur += 1
    return measures


def _binary_index(sorted_vals, target):
    """pyScoreParser.binary_index: index of the last element <= target (0 if none)."""
    lo, hi = 0, len(sorted_vals) - 1
    if target < sorted_vals[0]:
        return 0
    if target >= sorted_vals[-1]:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sorted_vals[mid] <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _playable_notes(measures: list, part_index: int) -> list:
    """pyScoreParser.get_playable_notes for one part -- the steps that decide WHICH notes exist
    and in what order (grace order, tie continuations, hidden notes, written-out trills)."""
    notes = []
    for mi, m in enumerate(measures):
        for n in m["notes"]:
            n.voice += part_index * 10          # CIPI: voices 1-4 of part 0 -> rh, all else -> lh
            n.measure = mi + 1
            notes.append(n)
    grace_tmp = []                              # classify_notes: grace_order -1, -2, ... backwards
    for n in notes:
        if n.grace:
            grace_tmp.append(n)
        elif grace_tmp:
            rest, order = [], -1
            for g in reversed(grace_tmp):
                if g.voice == n.voice:
                    g.grace_order = order
                    if g.chord_index == 0:
                        order -= 1
                else:
                    rest.append(g)
            grace_tmp = rest
    notes = [n for n in notes if not n.tied_stop]        # apply_tied_notes
    notes.sort(key=lambda x: (x.onset, x.grace_order, -x.midi))
    if not notes:
        return []
    # omit_trill_notes: hidden notes, then notes repeating a trilled pitch under its wavy line
    wavy = [[n.wavy[0], n.wavy[1], n.onset, n.midi, None] for n in notes if n.wavy is not None]
    notes = [n for n in notes if n.print_object]
    for i in range(len(wavy) - 1, -1, -1):
        w = wavy[i]
        if w[0] == "stop":
            for j in range(i - 1, -1, -1):
                if wavy[j][0] == "start" and wavy[j][1] == w[1]:
                    wavy[j][4] = w[2]
                    break
            wavy.pop(i)
    wavy = [w for w in wavy if w[4]]
    if wavy and notes:
        onsets = [n.onset for n in notes]
        omit = set()
        for w in wavy:
            idx = _binary_index(onsets, w[2])
            while abs(notes[idx].midi - w[3]) > 3 and idx > 0 and notes[idx - 1].onset == notes[idx].onset:
                idx -= 1
            trill_pitch, k = notes[idx].midi, idx + 1
            while k < len(notes) and notes[k].onset < w[4]:
                if notes[k].midi == trill_pitch:
                    omit.add(k)
                k += 1
        notes = [n for i, n in enumerate(notes) if i not in omit]
    return notes


def score_notes(path: str, return_info: bool = False):
    """Every playable note of the score in CIPI's order -> [(midi, hand, measure), ...]
    (plus a small info dict with return_info=True)."""
    root = _read_score_root(path)
    merged = merge_piano_parts(root)          # two single-hand piano parts -> one part (see there)
    parts = root.findall("part")
    if not parts:
        raise ScoreParseError(f"{path}: no <part>")
    notes = []
    for pi, part in enumerate(parts):
        notes.extend(_playable_notes(_expand_part(part), pi))
    notes.sort(key=lambda x: (x.onset, x.grace_order, -x.midi))
    if not notes:
        raise ScoreParseError(f"{path}: score contains no notes")
    out = [(n.midi, "rh" if n.voice < 5 else "lh", n.measure) for n in notes]
    if return_info:
        return out, {"merged_parts": merged, "n_parts": len(parts)}
    return out


def p_indices(notes: list) -> torch.Tensor:
    """CIPI get_pitches as a compact (T,) index tensor: one-hot column midi-21 for 21<=midi<=108,
    -1 for the all-zero row it writes for out-of-range pitches."""
    return torch.tensor([(m - 21) if 20 < m <= 108 else -1 for m, _, _ in notes], dtype=torch.int16)


def p_onehot(idx: torch.Tensor) -> torch.Tensor:
    """(T,) index tensor -> (T, 88) float piano-roll rows."""
    x = torch.zeros(idx.shape[0], 88)
    keep = idx >= 0
    x[keep, idx[keep].long()] = 1.0
    return x


def parse_score_file(path: str) -> dict:
    """Worker: one score -> {'idx': int16 (T,), 'n_notes', 'n_measures', 'n_rh', 'n_lh'}."""
    notes, info = score_notes(path, return_info=True)
    return {"idx": p_indices(notes), "n_notes": len(notes),
            "n_measures": max(m for _, _, m in notes),
            "n_rh": sum(h == "rh" for _, h, _ in notes), "n_lh": sum(h == "lh" for _, h, _ in notes),
            "n_parts": info["n_parts"], "merged_parts": info["merged_parts"]}


def _parse_job(job):
    stem, path = job
    try:
        return stem, parse_score_file(path), None
    except Exception as exc:  # one bad file must not kill a long parse
        return stem, None, f"{type(exc).__name__}: {exc}"


# =========================================================================================
# 3. Features: locate score files, parse (cached), load precomputed argnn / virtuoso
# =========================================================================================
def resolve_score(scores_dir: str, stem: str):
    for ext in SCORE_EXTS:
        p = os.path.join(scores_dir, stem + ext)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(scores_dir, "**", stem + ".*"), recursive=True)
    hits = [h for h in hits if h.lower().endswith(SCORE_EXTS)]
    return sorted(hits)[0] if hits else None


def build_p_features(items: list, args) -> dict:
    """stem -> parsed record (see parse_score_file). Cached in <cache-dir>/p.pt keyed by stem."""
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, "p.pt")
    feats = {}
    if os.path.exists(cache_path):
        feats = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"feature cache: {len(feats)} scores from {cache_path}")
    todo = [it["stem"] for it in items if it["stem"] not in feats]
    if not todo:
        return feats
    if not args.scores_dir or not os.path.isdir(args.scores_dir):
        sys.exit(f"--scores-dir {args.scores_dir!r} is not a directory (mount the score files first)")
    jobs, missing = [], []
    for stem in todo:
        path = resolve_score(args.scores_dir, stem)
        if path:
            jobs.append((stem, path))
        else:
            missing.append(stem)
    if missing:
        print(f"WARNING: {len(missing)} score files not found under {args.scores_dir} "
              f"(e.g. {missing[:5]}) -- skipped")
    print(f"parsing {len(jobs)} scores with {args.workers} worker(s) ...")
    t0, errors, done = time.time(), [], 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = ex.map(_parse_job, jobs, chunksize=4)
            for stem, rec, err in results:
                done += 1
                if rec is not None:
                    feats[stem] = rec
                else:
                    errors.append((stem, err))
                if done % 50 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)}  ({time.time() - t0:.0f}s)")
                    torch.save(feats, cache_path)
    else:
        for job in jobs:
            stem, rec, err = _parse_job(job)
            done += 1
            if rec is not None:
                feats[stem] = rec
            else:
                errors.append((stem, err))
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  ({time.time() - t0:.0f}s)")
                torch.save(feats, cache_path)
    torch.save(feats, cache_path)
    if errors:
        print(f"WARNING: {len(errors)} scores failed to parse and are skipped:")
        for stem, err in errors[:20]:
            print(f"    {stem}: {err[:160]}")
    print(f"parsed {len(jobs) - len(errors)} scores in {time.time() - t0:.0f}s -> {cache_path}")
    return feats


def _as_TD(t: torch.Tensor, dim: int) -> torch.Tensor:
    t = t.detach().cpu().float()
    while t.dim() > 2:
        sq = [i for i, s in enumerate(t.shape) if s == 1]
        if not sq:
            raise ValueError(f"cannot reduce {tuple(t.shape)} to (T,{dim})")
        t = t.squeeze(sq[0])
    if t.dim() != 2 or t.shape[1] != dim:
        raise ValueError(f"expected (T,{dim}), got {tuple(t.shape)}")
    return t


def build_precomputed_features(items: list, rep: str, args) -> dict:
    """argnn -> stem: (rh (T,64), lh (T,64));  virtuoso -> stem: (T,64). Read from --emb-dir in the
    layout of cipi_trainer/.cache/xml_embeddings/<stem>-<hash>/{embed.pt, virtuoso.pt}; cached in
    <cache-dir>/<rep>.pt. Scores without the embedding (e.g. a hand with no notes) are skipped."""
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"{rep}.pt")
    feats = {}
    if os.path.exists(cache_path):
        feats = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"feature cache: {len(feats)} {rep} scores from {cache_path}")
    todo = [it["stem"] for it in items if it["stem"] not in feats]
    if not todo:
        return feats
    if not args.emb_dir or not os.path.isdir(args.emb_dir):
        sys.exit(f"rep {rep!r} needs precomputed embeddings: pass --emb-dir DIR "
                 f"(layout: DIR/<stem>-<hash>/embed.pt + virtuoso.pt)")
    fname = "embed.pt" if rep == "argnn" else "virtuoso.pt"
    missing, skipped, t0 = [], [], time.time()
    for i, stem in enumerate(todo, 1):
        dirs = sorted(d for d in glob.glob(os.path.join(args.emb_dir, f"{stem}-*"))
                      if os.path.exists(os.path.join(d, fname)))
        if not dirs:
            missing.append(stem)
            continue
        d = torch.load(os.path.join(dirs[-1], fname), map_location="cpu", weights_only=False)
        try:
            if rep == "argnn":
                if d.get("argnn_rh") is None or d.get("argnn_lh") is None:
                    skipped.append(stem)      # a hand with no notes has no fingering embedding
                    continue
                feats[stem] = (_as_TD(d["argnn_rh"], 64), _as_TD(d["argnn_lh"], 64))
            else:
                feats[stem] = _as_TD(d["total_note_cat"], 64)
        except (KeyError, ValueError) as exc:
            skipped.append(stem)
            print(f"  {stem}: unusable {fname}: {exc}")
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)")
    torch.save(feats, cache_path)
    if missing:
        print(f"WARNING {rep}: {len(missing)} scores have no {fname} under {args.emb_dir} (e.g. {missing[:5]})")
    if skipped:
        print(f"NOTE {rep}: {len(skipped)} scores skipped (empty hand / unusable tensor): {', '.join(skipped)}")
    print(f"{rep}: {len(feats)} scores -> {cache_path}")
    return feats


def feature_length(rep: str, f) -> int:
    if rep == "p":
        return int(f["idx"].shape[0])
    if rep == "argnn":
        return int(max(f[0].shape[0], f[1].shape[0]))
    return int(f.shape[0])


# =========================================================================================
# 4. Models  (identical to cipi_trainer/cipi/models.py)
# =========================================================================================
class ContextAttention(nn.Module):
    """Context-attention pooling over the time axis, masking zero-padded steps."""

    def __init__(self, size: int, num_head: int = 1):
        super().__init__()
        if size % num_head != 0:
            raise ValueError(f"size {size} must be divisible by num_head {num_head}")
        if size // num_head == 1:
            raise ValueError("head_size == 1 falls into the degenerate branch; use fewer heads")
        self.size, self.num_head, self.head_size = size, num_head, size // num_head
        self.attention_net = nn.Linear(size, size)
        self.context_vector = nn.Parameter(torch.empty(num_head, self.head_size, 1))
        nn.init.uniform_(self.context_vector, -1.0, 1.0)

    def forward(self, x: torch.Tensor, return_weights: bool = False):  # (B, T, size) -> (B, size)
        attn = torch.tanh(self.attention_net(x))
        if self.head_size != 1:
            split = torch.stack(attn.split(self.head_size, dim=2), dim=0)
            sim = torch.bmm(split.view(self.num_head, -1, self.head_size), self.context_vector)
            sim = sim.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)  # (B, T, H)
            sim[x.sum(-1) == 0] = -1e10  # mask padded timesteps
            weight = torch.softmax(sim, dim=1)
            x_split = torch.stack(x.split(self.head_size, dim=2), dim=2)
            attn = (x_split * weight.unsqueeze(-1)).view(x.shape[0], x.shape[1], x.shape[-1])
        else:
            weight = torch.softmax(attn, dim=1)
            attn = weight * x
        out = attn.sum(dim=1)
        return (out, weight) if return_weights else out


class PackedBiGRU(nn.Module):
    """Dropout -> length-packed 3-layer bidirectional GRU -> padded output."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.in_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.in_dropout(x.float())
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return out  # (B, T, 2*hidden)


class OrdinalHead(nn.Module):
    """Single-branch head for the pitch ("p") and virtuoso representations."""

    def __init__(self, input_size: int, n_classes: int = N_CLASSES, num_head: int = 1):
        super().__init__()
        size = 64
        self.branch = PackedBiGRU(input_size, size // 2, num_layers=3)
        self.attn = ContextAttention(size, num_head=num_head)
        self.act = nn.ReLU()
        self.fc = nn.Linear(size, n_classes)

    def forward(self, x, lengths):
        return self.fc(self.act(self.attn(self.branch(x, lengths))))


class OrdinalHeadArgnn(nn.Module):
    """Two-branch (right/left hand) head for the argnn fingering representation."""

    def __init__(self, input_size: int = 64, n_classes: int = N_CLASSES, num_head: int = 1):
        super().__init__()
        hidden = 64
        size = hidden * 2  # 128
        self.branch_rh = PackedBiGRU(input_size, hidden, num_layers=3)
        self.branch_lh = PackedBiGRU(input_size, hidden, num_layers=3)
        self.attn_rh = ContextAttention(size, num_head=num_head)
        self.attn_lh = ContextAttention(size, num_head=num_head)
        self.act = nn.ReLU()
        self.fc = nn.Linear(size * 2, n_classes)

    def forward(self, rh, lh, rh_len, lh_len):
        srh = self.attn_rh(self.branch_rh(rh, rh_len))
        slh = self.attn_lh(self.branch_lh(lh, lh_len))
        return self.fc(self.act(torch.cat([srh, slh], dim=1)))


class OrdinalLoss(nn.Module):
    """MSE over a cumulative ordinal encoding; optional per-grade sample weights; mean over batch."""

    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        enc = torch.zeros_like(logits)
        for i, t in enumerate(targets):
            enc[i, : int(t) + 1] = 1.0
        se = F.mse_loss(logits, enc, reduction="none").mean(dim=1)
        if self.class_weights is not None:
            w = self.class_weights[targets]
            return (w * se).sum() / w.sum().clamp(min=1e-8)
        return se.mean()


def prediction2label(logits: torch.Tensor) -> torch.Tensor:
    """Ordinal decode: raw scores (B, K) -> label 0..K-1 (or -1 if the first score <= 0.5)."""
    return (logits > 0.5).cumprod(dim=1).sum(dim=1) - 1


def build_head(rep: str, num_head: int = 1) -> nn.Module:
    if rep == "p":
        return OrdinalHead(input_size=88, num_head=num_head)
    if rep == "virtuoso":
        return OrdinalHead(input_size=64, num_head=num_head)
    if rep == "argnn":
        return OrdinalHeadArgnn(input_size=64, num_head=num_head)
    raise ValueError(f"unknown representation: {rep!r}")


# =========================================================================================
# 5. Data pipeline  (as cipi/data.py, over in-memory features)
# =========================================================================================
class PieceDataset(Dataset):
    """Items = [(stem, grade)]; features from the in-memory dict. Item -> (features, target 0..8).
    max_len > 0 cuts every piece to that many notes: random window when training, tail otherwise."""

    def __init__(self, items: list, feats: dict, rep: str, max_len: int = 0, train: bool = False):
        self.items = [(it["stem"], int(it["grade"])) for it in items if it["stem"] in feats]
        self.feats, self.rep, self.max_len, self.train = feats, rep, max_len, train

    def __len__(self):
        return len(self.items)

    def _window(self, x):
        if not self.max_len or x.shape[0] <= self.max_len:
            return x
        t = x.shape[0]
        start = random.randint(0, t - self.max_len) if self.train else t - self.max_len
        return x[start:start + self.max_len]

    def features(self, stem: str):
        f = self.feats[stem]
        if self.rep == "p":
            return p_onehot(self._window(f["idx"]))                  # (T, 88)
        if self.rep == "virtuoso":
            return self._window(f)                                   # (T, 64)
        return self._window(f[0]), self._window(f[1])                # (T_rh, 64), (T_lh, 64)

    def __getitem__(self, i):
        stem, grade = self.items[i]
        return self.features(stem), grade - 1


def collate_single(batch):
    feats, ys = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    return pad_sequence(feats, batch_first=True), lengths, torch.tensor(ys, dtype=torch.long)


def collate_argnn(batch):
    feats, ys = zip(*batch)
    rh, lh = [f[0] for f in feats], [f[1] for f in feats]
    rl = torch.tensor([t.shape[0] for t in rh], dtype=torch.long)
    ll = torch.tensor([t.shape[0] for t in lh], dtype=torch.long)
    return (pad_sequence(rh, batch_first=True), pad_sequence(lh, batch_first=True), rl, ll), \
        torch.tensor(ys, dtype=torch.long)


def collate_for(rep):
    return collate_argnn if rep == "argnn" else collate_single


def balanced_class_weights(items: list, n_classes: int = N_CLASSES) -> torch.Tensor:
    """Inverse-frequency per-grade weights, normalized so the present grades average 1."""
    counts = [0] * n_classes
    for it in items:
        counts[int(it["grade"]) - 1] += 1
    inv = [0.0 if c == 0 else 1.0 / c for c in counts]
    present = [w for w in inv if w > 0]
    mean = (sum(present) / len(present)) if present else 1.0
    return torch.tensor([w / mean for w in inv], dtype=torch.float)


class LengthBucketBatchSampler:
    """Batches of similar-length pieces; a batch grows until `max_batch` items or the token budget
    (max_len_in_batch * n_items > budget). Bounds memory without truncating anything."""

    def __init__(self, lengths, token_budget: int, max_batch: int, shuffle: bool):
        self.shuffle = shuffle
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches, cur, cur_max = [], [], 0
        for i in order:
            nmax = max(cur_max, lengths[i])
            if cur and (nmax * (len(cur) + 1) > token_budget or len(cur) >= max_batch):
                self.batches.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, lengths[i])
        if cur:
            self.batches.append(cur)

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(order)
        for b in order:
            yield self.batches[b]

    def __len__(self):
        return len(self.batches)


def make_loader(items, feats, rep, args, train):
    ds = PieceDataset(items, feats, rep, max_len=args.max_len, train=train)
    lengths = [min(feature_length(rep, feats[s]), args.max_len or 10**9) for s, _ in ds.items]
    sampler = LengthBucketBatchSampler(lengths, args.token_budget, args.batch_size, shuffle=train)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_for(rep))


# =========================================================================================
# 6. Metrics  (cipi/metrics.py)
# =========================================================================================
def summary(y_true, y_pred) -> dict:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    per = [mean_squared_error(yt[yt == c], yp[yt == c]) for c in np.unique(yt)]
    return {"acc9": round(float(balanced_accuracy_score(yt, yp)), 4),
            "acc_pm1": round(float(np.mean(np.abs(yt - yp) <= 1)), 4),
            "mse": round(float(mean_squared_error(yt, yp)), 4),
            "macro_mse": round(float(np.mean(per)), 4) if per else float("nan")}


COARSE_EDGES = (3, 6)      # acc3 buckets of 1-based grades: 1-3, 4-6, 7 and above
TEST_METRICS = ("acc9", "acc_pm1", "acc3", "acc9_plain", "acc_pm1_plain", "acc3_plain")


def coarse_bucket(grades):
    """1-based grades -> 0 (grades 1-3), 1 (4-6), 2 (7 and above)."""
    g = np.asarray(grades)
    return (g > COARSE_EDGES[0]).astype(int) + (g > COARSE_EDGES[1]).astype(int)


def _per_class_mean(y, hit) -> float:
    """Mean over the classes present in y of the hit rate within that class (class-weighted)."""
    rates = [np.mean(hit[y == c]) for c in np.unique(y)]
    return float(np.mean(rates)) if rates else float("nan")


def test_summary(y_true, y_pred) -> dict:
    """Test metrics, defined as in onset_colab_train_806*.py (0-based labels in, 1-based grades for acc3):
    acc9    = balanced accuracy over the 9 grades (mean per-grade recall),
    acc_pm1 = balanced Acc±1 (mean over the grades of P(|pred - true| <= 1)),
    acc3    = balanced accuracy over the coarse buckets 1-3 / 4-6 / 7+.
    The *_plain keys are the same hits as a plain fraction of the scores (weighted by class size)."""
    y, p = np.asarray(y_true, int), np.asarray(y_pred, int)
    exact, pm1 = p == y, np.abs(p - y) <= 1
    cy, cp = coarse_bucket(y + 1), coarse_bucket(p + 1)
    return {"n": int(len(y)),
            "acc9": round(_per_class_mean(y, exact), 4),
            "acc_pm1": round(_per_class_mean(y, pm1), 4),
            "acc3": round(_per_class_mean(cy, cy == cp), 4),
            "acc9_plain": round(float(np.mean(exact)), 4),
            "acc_pm1_plain": round(float(np.mean(pm1)), 4),
            "acc3_plain": round(float(np.mean(cy == cp)), 4)}


def mean_std(per_fold: list) -> dict:
    """[test_summary per fold] -> {metric: (mean, std over the folds)} (np.std, as the onset files)."""
    return {k: (round(float(np.mean([m[k] for m in per_fold])), 4), round(float(np.std([m[k] for m in per_fold])), 4))
            for k in TEST_METRICS}


# =========================================================================================
# 7. Training  (cipi/train.py recipe)
# =========================================================================================
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def forward_batch(model, rep, batch, device):
    if rep == "argnn":
        (rh, lh, rl, ll), y = batch
        return model(rh.to(device), lh.to(device), rl, ll), y.to(device)
    x, lengths, y = batch
    return model(x.to(device), lengths), y.to(device)


@torch.no_grad()
def evaluate(model, rep, loader, device, criterion=None) -> dict:
    model.eval()
    y_true, y_pred, total, n = [], [], 0.0, 0
    for batch in loader:
        logits, y = forward_batch(model, rep, batch, device)
        if criterion is not None:
            total += float(criterion(logits, y)); n += 1
        y_pred += prediction2label(logits.cpu()).clamp(min=0).tolist()  # -1 sentinel -> 0
        y_true += y.cpu().tolist()
    m = summary(y_true, y_pred)
    if criterion is not None:
        m["loss"] = round(total / max(n, 1), 4)
    return m


def train_one(rep: str, fold: int, items: list, feats: dict, args) -> str:
    """Train one head on the fold's TRAIN part and keep the best epoch on its VAL part. The TEST part
    is not touched here (see test_fold). -> checkpoint path. With --resume an existing checkpoint
    of this dataset is reused instead of training again."""
    out = os.path.join(args.out_dir, f"{rep}_ordinal-{fold}.pt")
    if args.resume and os.path.exists(out):
        ck = torch.load(out, map_location="cpu", weights_only=False)
        if ck.get("dataset") == DATASET_TAG and ck.get("n_items") == len(items):
            print(f"\n[{rep} f{fold}] resume: using {out} (best epoch {ck.get('epoch')})")
            return out
        print(f"\n[{rep} f{fold}] resume: {out} is from another dataset / subset -- training again")
    set_seed(args.seed)
    device = pick_device(args.device)
    split = fold_split(items, fold)
    train_loader = make_loader(split["train"], feats, rep, args, train=True)
    val_loader = make_loader(split["val"], feats, rep, args, train=False)
    if not len(train_loader.dataset) or not len(val_loader.dataset):
        sys.exit(f"[{rep} f{fold}] empty train/val set -- not enough scores with features")

    model = build_head(rep, num_head=args.num_head).to(device)
    train_items = [it for it in split["train"] if it["stem"] in feats]
    class_w = balanced_class_weights(train_items).to(device) if args.balanced else None
    criterion = OrdinalLoss(class_weights=class_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = {"macro_mse": float("inf"), "acc_pm1": float("-inf"), "state": None, "epoch": -1, "val": None}
    stale = 0
    n_test = sum(it["stem"] in feats for it in split["test"])
    print(f"\n=== training rep={rep} fold={fold} device={device} lr={args.lr} max_len={args.max_len} "
          f"(train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={n_test}) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, t0 = 0.0, time.time()
        for batch in train_loader:
            logits, y = forward_batch(model, rep, batch, device)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total += float(loss.detach())
        val = evaluate(model, rep, val_loader, device, criterion)
        by_mse = val["macro_mse"] < best["macro_mse"]
        by_pm1 = val["acc_pm1"] > best["acc_pm1"]
        print(f"[{rep} f{fold}] epoch {epoch:>3} | train_loss {total / max(len(train_loader), 1):.4f} "
              f"| val {val} | {time.time() - t0:.0f}s{' *' if (by_mse or by_pm1) else ''}")
        if by_mse or by_pm1:
            best = {"macro_mse": val["macro_mse"] if by_mse else best["macro_mse"],
                    "acc_pm1": val["acc_pm1"] if by_pm1 else best["acc_pm1"],
                    "epoch": epoch, "val": val,
                    "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[{rep} f{fold}] early stop")
                break

    print(f"[{rep} f{fold}] BEST epoch {best['epoch']} | val {best['val']}")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"state_dict": best["state"], "representation": rep, "fold": fold,
                "num_head": args.num_head, "max_len": args.max_len,
                "val_metrics": best["val"], "test_metrics": {}, "epoch": best["epoch"],
                "source": "cipi_colab_train_musicxml.py", "dataset": DATASET_TAG, "n_items": len(items),
                "label_column": "Сложность (число)", "n_train": len(train_loader.dataset)}, out)
    print(f"[{rep} f{fold}] saved -> {out}")
    return out


# =========================================================================================
# 8. Test evaluation: per fold (all heads + ensemble on the fold's TEST part), then mean ± std
# =========================================================================================
def load_checkpoint_model(path: str, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    heads = {int(v.shape[0]) for k, v in ck["state_dict"].items() if k.endswith("context_vector")}
    num_head = heads.pop() if len(heads) == 1 else int(ck.get("num_head", 1))
    model = build_head(ck["representation"], num_head=num_head)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), int(ck.get("max_len", 0))


@torch.no_grad()
def predict_split(rep: str, ckpt: str, split_items: list, feats: dict, args) -> dict:
    """stem -> predicted label 0..8 (the -1 'no prediction' sentinel counts as 0) for every score of
    split_items that has this head's features."""
    device = pick_device(args.device)
    model, max_len = load_checkpoint_model(ckpt, device)
    ds = PieceDataset(split_items, feats, rep, max_len=max_len, train=False)
    lengths = [min(feature_length(rep, feats[s]), max_len or 10**9) for s, _ in ds.items]
    sampler = LengthBucketBatchSampler(lengths, args.token_budget, args.batch_size, shuffle=False)
    collate = collate_for(rep)
    out = {}
    for idxs in sampler:
        logits, _ = forward_batch(model, rep, collate([ds[j] for j in idxs]), device)
        for j, lab in zip(idxs, prediction2label(logits.cpu()).clamp(min=0).tolist()):
            out[ds.items[j][0]] = lab
    return out


def ensemble_preds(preds: dict) -> dict:
    """{rep: {stem: label}} -> {stem: rounded mean label} over the scores every head predicted."""
    common = set.intersection(*[set(p) for p in preds.values()]) if preds else set()
    return {s: int(round(sum(p[s] for p in preds.values()) / len(preds))) for s in common}


def _metrics_row(name: str, m: dict) -> str:
    return (f"  {name:<28} {m['n']:>4}   {m['acc9']:.4f} {m['acc_pm1']:.4f} {m['acc3']:.4f}   | "
            f"{m['acc9_plain']:.4f} {m['acc_pm1_plain']:.4f} {m['acc3_plain']:.4f}")


METRICS_HEADER = (f"  {'':<28} {'':>4}   {'balanced':<20}   | plain (fraction of scores)\n"
                  f"  {'head':<28} {'n':>4}   {'acc9':<6} {'acc±1':<6} {'acc3':<6}   | "
                  f"{'acc9':<6} {'acc±1':<6} {'acc3':<6}")


def test_fold(fold: int, ckpts: dict, items: list, all_feats: dict, args):
    """Grade the fold's TEST part with every head (+ the ensemble when there are several heads), print
    the metrics and store them in the checkpoints. -> ({head: test_summary}, {rep: {stem: label}}, ensemble)."""
    grade_of = {it["stem"]: it["grade"] for it in items}
    test_items = fold_split(items, fold)["test"]
    preds = {rep: predict_split(rep, ck, test_items, all_feats[rep], args) for rep, ck in ckpts.items()}
    preds = {rep: p for rep, p in preds.items() if p}
    metrics = {}
    for rep, p in preds.items():
        metrics[rep] = test_summary([grade_of[s] - 1 for s in p], list(p.values()))
        ck = torch.load(ckpts[rep], map_location="cpu", weights_only=False)
        ck["test_metrics"] = metrics[rep]
        torch.save(ck, ckpts[rep])
    ens = ensemble_preds(preds) if len(preds) > 1 else {}
    if ens:
        metrics["ensemble"] = test_summary([grade_of[s] - 1 for s in ens], list(ens.values()))
    print(f"\n===== fold {fold} TEST (bucket {fold % N_BUCKETS}, {len(test_items)} scores) =====")
    print(METRICS_HEADER)
    for name, m in metrics.items():
        print(_metrics_row(f"ensemble({'+'.join(preds)})" if name == "ensemble" else name, m))
    return metrics, preds, ens


def report_over_folds(fold_metrics: dict) -> dict:
    """{fold: {head: test_summary}} -> {head: {metric: (mean, std)}}; prints the table."""
    heads = []
    for m in fold_metrics.values():
        heads += [h for h in m if h not in heads]
    out = {}
    print(f"\n===== TEST metrics over folds {sorted(fold_metrics)}: mean ± std (np.std over the folds) =====")
    print(f"  {'head':<10} {'folds':>5}  " + "  ".join(f"{k:<15}" for k in ("acc9", "acc±1", "acc3"))
          + "  |  " + "  ".join(f"{k + ' plain':<15}" for k in ("acc9", "acc±1", "acc3")))
    for h in heads:
        per = [m[h] for m in fold_metrics.values() if h in m]
        out[h] = mean_std(per)
        cells = [f"{out[h][k][0]:.4f} ± {out[h][k][1]:.4f}" for k in TEST_METRICS]
        print(f"  {h:<10} {len(per):>5}  " + "  ".join(cells[:3]) + "  |  " + "  ".join(cells[3:]))
    return out


def oof_report(preds: dict, ens: dict, fold_of: dict, items: list, args) -> None:
    """Pooled out-of-fold view: every score graded by the fold where it was in TEST."""
    grade_of = {it["stem"]: it["grade"] for it in items}
    print("\n===== out-of-fold (all test parts pooled) =====")
    print(METRICS_HEADER)
    for rep, p in preds.items():
        print(_metrics_row(rep, test_summary([grade_of[s] - 1 for s in p], list(p.values()))))
    if ens:
        print(_metrics_row(f"ensemble({'+'.join(preds)})", test_summary([grade_of[s] - 1 for s in ens], list(ens.values()))))
    final = ens or next(iter(preds.values()), {})
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=int)   # rows = true grade, cols = predicted
    for s, lab in final.items():
        conf[grade_of[s] - 1, lab] += 1
    print(f"confusion of the {'ensemble' if ens else 'head'} (rows: true 1..9, cols: predicted 1..9):")
    for g in range(N_CLASSES):
        print(f"  {g + 1}: " + " ".join(f"{c:3d}" for c in conf[g]))
    csv_path = os.path.join(args.out_dir, "oof_predictions.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "grade", "fold"] + [f"pred_{r}" for r in preds] + ["pred_ensemble"])
        for s in sorted(set().union(*[set(p) for p in preds.values()])):
            per_rep = [preds[r][s] + 1 if s in preds[r] else "" for r in preds]   # -> grades 1..9
            w.writerow([s, grade_of[s], fold_of[s]] + per_rep + [ens[s] + 1 if s in ens else ""])
    print(f"per-score OOF predictions -> {csv_path}")


# =========================================================================================
# 9. Grade particular scores: --predict (paths / links) and --eval-list FILE (one score per line)
# =========================================================================================
EVAL_SUBDIR = "eval"
_UA = "Mozilla/5.0 (cipi_colab_train_musicxml)"


def is_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def direct_download_url(url: str) -> str:
    """Turn common share links into direct-download links (Google Drive, Dropbox)."""
    m = re.match(r"https?://drive\.google\.com/file/d/([^/]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    m = re.match(r"https?://drive\.google\.com/open\?id=([^&]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    if "dropbox.com" in url:
        return re.sub(r"[?&]dl=0", lambda mm: mm.group(0)[0] + "dl=1", url) if "dl=0" in url else url
    return url


def _sniff_extension(head: bytes):
    """'.mxl' for a zip container, '.musicxml' for XML, None for anything else (e.g. an HTML page)."""
    if head.startswith(b"PK\x03\x04"):
        return ".mxl"
    text = head.lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if text.startswith(b"<?xml") or text.startswith(b"<score-partwise") or text.startswith(b"<!doctype score-partwise"):
        return ".musicxml"
    return None


def download_score(url: str, dest_dir: str, force: bool = False) -> str:
    """Download a score link into dest_dir (once; re-used afterwards) and return the local path.
    The saved name is <8 chars of the url hash>_<basename>, with the extension fixed from the
    content (zip -> .mxl, xml -> .musicxml) when the link does not carry one."""
    import hashlib
    import urllib.parse
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    tag = hashlib.sha1(url.encode()).hexdigest()[:8]
    base = os.path.basename(urllib.parse.unquote(urllib.parse.urlparse(url).path)) or "score"
    existing = glob.glob(os.path.join(dest_dir, f"{tag}_*"))
    if existing and not force:
        return existing[0]
    req = urllib.request.Request(direct_download_url(url), headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
        cd = resp.headers.get("Content-Disposition") or ""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    if m:
        base = os.path.basename(urllib.parse.unquote(m.group(1).strip()))
    ext = _sniff_extension(data[:512])
    if ext is None:
        raise ScoreParseError(f"{url}: the download is not a MusicXML file or .mxl container "
                              f"(got {len(data)} bytes starting with {data[:40]!r}); is it a direct link?")
    stem, cur_ext = os.path.splitext(base)
    if cur_ext.lower() not in SCORE_EXTS:
        stem, cur_ext = base, ext
    path = os.path.join(dest_dir, f"{tag}_{stem}{cur_ext}")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def entry_stem(source: str) -> str:
    """Name used to look the score up in --emb-dir and to label the output rows."""
    import urllib.parse
    base = os.path.basename(urllib.parse.unquote(urllib.parse.urlparse(source).path)) if is_url(source) \
        else os.path.basename(source)
    return os.path.splitext(base)[0] or source


def read_eval_list(path: str) -> list:
    """One score per line: <https link or local path> [label 1..9] [name ...].
    Separator: tab, ';' or ',' if the file uses one, otherwise whitespace. Lines starting with '#'
    and a header line (first field 'url', 'link', 'path', 'file', 'source' or 'score') are skipped."""
    with open(path, encoding="utf-8-sig") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    delim = next((d for d in ("\t", ";", ",") if any(d in ln for ln in lines)), None)
    rows = list(csv.reader(lines, delimiter=delim)) if delim else [ln.split() for ln in lines]
    entries = []
    for row in rows:
        row = [c.strip() for c in row if c is not None]
        row = [c for c in row if c != ""]
        if not row:
            continue
        if row[0].lower() in ("url", "link", "path", "file", "source", "score") and not entries:
            continue
        source, label, name = row[0], None, None
        rest = row[1:]
        if rest and re.fullmatch(r"[1-9](\.0)?", rest[0]):
            label, rest = int(float(rest[0])), rest[1:]
        elif rest and re.fullmatch(r"[1-9](\.0)?", rest[-1]):
            label, rest = int(float(rest[-1])), rest[:-1]
        if rest:
            name = " ".join(rest)
        entries.append({"source": source, "label": label, "name": name or entry_stem(source)})
    return entries


def checkpoint_dirs(args) -> list:
    """--out-dir plus every --ckpt-dir folder (comma-separated), in that order."""
    dirs = [args.out_dir] + [d.strip() for d in (args.ckpt_dir or "").split(",") if d.strip()]
    return [os.path.abspath(d) for d in dirs]


def find_checkpoints(dirs, rep: str) -> list:
    """<rep>_ordinal-<fold>.pt files across dirs; the first folder that has a fold wins."""
    seen, out = set(), []
    for d in dirs:
        for ck in sorted(glob.glob(os.path.join(d, f"{rep}_ordinal-*.pt"))):
            fold = os.path.basename(ck).rsplit("-", 1)[1][:-3]
            if fold not in seen:
                seen.add(fold)
                out.append(ck)
    return out


def load_heads(dirs, reps, device: str) -> dict:
    """rep -> [(model, max_len, fold)] for every readable <rep>_ordinal-<fold>.pt in dirs.
    A checkpoint that cannot be read (e.g. being written by a training run) is skipped."""
    heads = {}
    for rep in reps:
        models = []
        for ck in find_checkpoints(dirs, rep):
            try:
                model, max_len = load_checkpoint_model(ck, device)
            except Exception as exc:
                print(f"WARNING: cannot read {ck} ({type(exc).__name__}: {exc}); skipped")
                continue
            models.append((model, max_len, int(os.path.basename(ck).rsplit("-", 1)[1][:-3])))
        if models:
            heads[rep] = models
    return heads


def load_precomputed_for_stem(emb_dir: str, rep: str, stem: str):
    """The argnn (rh, lh) or virtuoso features of one score from --emb-dir, or None."""
    fname = "embed.pt" if rep == "argnn" else "virtuoso.pt"
    dirs = sorted(d for d in glob.glob(os.path.join(emb_dir, f"{stem}-*")) if os.path.exists(os.path.join(d, fname)))
    if not dirs:
        return None
    d = torch.load(os.path.join(dirs[-1], fname), map_location="cpu", weights_only=False)
    try:
        if rep == "argnn":
            if d.get("argnn_rh") is None or d.get("argnn_lh") is None:
                return None
            return _as_TD(d["argnn_rh"], 64), _as_TD(d["argnn_lh"], 64)
        return _as_TD(d["total_note_cat"], 64)
    except (KeyError, ValueError):
        return None


def _tail(x: torch.Tensor, max_len: int) -> torch.Tensor:
    return x if not max_len or x.shape[0] <= max_len else x[-max_len:]


@torch.no_grad()
def grade_with_heads(rep: str, feats, models, device: str) -> list:
    """Hard grade 1..9 per fold model (None where the ordinal decode gives no prediction)."""
    grades = []
    for model, max_len, _fold in models:
        if rep == "argnn":
            rh, lh = _tail(feats[0], max_len).to(device), _tail(feats[1], max_len).to(device)
            logits = model(rh.unsqueeze(0), lh.unsqueeze(0), torch.tensor([rh.shape[0]]), torch.tensor([lh.shape[0]]))
        else:
            x = _tail(feats, max_len).to(device)
            logits = model(x.unsqueeze(0), torch.tensor([x.shape[0]]))
        lab = int(prediction2label(logits.cpu()).item())
        grades.append(None if lab < 0 else lab + 1)
    return grades


def grade_entries(entries: list, args, reps) -> list:
    """Download / parse every entry and grade it with each requested head that has checkpoints.
    argnn / virtuoso need the score's embedding under --emb-dir (looked up by file stem); the
    pitch head is computed here from the MusicXML itself."""
    device = pick_device(args.device)
    dirs = checkpoint_dirs(args)
    heads = load_heads(dirs, reps, device)
    missing = [r for r in reps if r not in heads]
    if missing:
        print(f"NOTE: no checkpoints for {missing} in {dirs}; graded with {list(heads) or 'nothing'}")
    if not heads:
        sys.exit(f"no <rep>_ordinal-*.pt checkpoints in {dirs} -- train first, or add --ckpt-dir FOLDER")
    for rep, models in heads.items():
        print(f"{rep}: {len(models)} fold model(s) {[f for _, _, f in models]}")
    if "p" not in heads:
        print("WARNING: no pitch-head checkpoints (p_ordinal-*.pt) found. A MusicXML file or link can only be "
              "graded by the pitch head here -- argnn / virtuoso need the score's precomputed embedding under "
              "--emb-dir. Train the pitch head (--reps p) or add its folder with --ckpt-dir.")
    dl_dir = os.path.join(args.out_dir, EVAL_SUBDIR, "scores")
    results = []
    for i, e in enumerate(entries, 1):
        res = {"name": e["name"], "source": e["source"], "label": e.get("label")}
        try:
            path = download_score(e["source"], dl_dir, args.force_download) if is_url(e["source"]) else e["source"]
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            res["file"] = path
            rec = parse_score_file(path)
        except Exception as exc:
            res["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [{i}/{len(entries)}] {e['name']}: ERROR {res['error'][:160]}")
            results.append(res)
            continue
        res.update({"n_notes": rec["n_notes"], "n_measures": rec["n_measures"], "merged_parts": rec["merged_parts"]})
        stem = entry_stem(e["source"])
        per_rep, notes = {}, []
        for rep, models in heads.items():
            if rep == "p":
                feats = p_onehot(rec["idx"])
            else:
                feats = load_precomputed_for_stem(args.emb_dir, rep, stem) if args.emb_dir else None
                if feats is None:
                    notes.append(f"{rep}: no embedding for '{stem}' under --emb-dir" if args.emb_dir
                                 else f"{rep}: needs --emb-dir with this score's embedding")
                    continue
            grades = grade_with_heads(rep, feats, models, device)
            ok = [g for g in grades if g is not None]
            per_rep[rep] = {"folds": grades, "mean": round(sum(ok) / len(ok), 2) if ok else None}
        means = [v["mean"] for v in per_rep.values() if v["mean"] is not None]
        res["per_rep"] = per_rep
        res["ensemble_mean"] = round(sum(means) / len(means), 3) if means else None
        res["grade"] = int(round(sum(means) / len(means))) if means else None
        if res["label"] is not None and res["grade"] is not None:
            res["diff"] = res["grade"] - res["label"]
        if per_rep and not means:
            notes.append("every fold hit the ordinal 'no prediction' sentinel (untrained or unsuitable model)")
        if not per_rep:
            notes.insert(0, "NOT GRADED: no head could score this file (see the reasons below)")
        if notes:
            res["notes"] = notes
        shown = " ".join(f"{r}={v['mean']}" for r, v in per_rep.items())
        print(f"  [{i}/{len(entries)}] {e['name'][:50]}: grade {res['grade']} ({shown}) "
              f"notes={rec['n_notes']} measures={rec['n_measures']}"
              + (f" label {res['label']}" if res["label"] is not None else "")
              + (f" | {rec['merged_parts']}" if rec["merged_parts"] else ""))
        for n in notes:
            print(f"        - {n}")
        results.append(res)
    return results


def _metrics_block(y_true, y_pred) -> dict:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    m = summary(yt - 1, yp - 1)
    m["mae"] = round(float(np.mean(np.abs(yt - yp))), 4)
    m["n"] = int(len(yt))
    return m


def evaluate_entries(entries: list, args, reps, tag: str = "eval") -> list:
    """Grade the entries, print a table (+ metrics when labels are given) and write
    <out-dir>/eval/<tag>_results.csv and .json."""
    results = grade_entries(entries, args, reps)
    reps_used = sorted({r for res in results for r in res.get("per_rep", {})}, key=REPS.index)
    print(f"\n{'name':<40} {'label':>5} {'notes':>6} " + " ".join(f"{r:>9}" for r in reps_used) + f" {'grade':>5} {'diff':>5}  folds")
    for res in results:
        if "error" in res:
            print(f"{res['name'][:40]:<40} {res['label'] if res['label'] is not None else '':>5}  ERROR {res['error'][:80]}")
            continue
        cols = " ".join(f"{str(res['per_rep'].get(r, {}).get('mean', '')):>9}" for r in reps_used)
        folds = ";".join(f"{r}:" + ",".join("-" if g is None else str(g) for g in v["folds"]) for r, v in res["per_rep"].items())
        print(f"{res['name'][:40]:<40} {res['label'] if res['label'] is not None else '':>5} {res.get('n_notes', ''):>6} {cols} "
              f"{res['grade'] if res['grade'] is not None else '':>5} {res.get('diff', ''):>5}  {folds}")
        for n in res.get("notes", []):
            print(f"{'':<40}   - {n}")
    labelled = [r for r in results if r.get("label") is not None and r.get("grade") is not None]
    if labelled:
        print(f"\nmetrics over {len(labelled)} labelled scores (acc9 = balanced accuracy, acc_pm1 = within +-1 grade):")
        print(f"  ensemble : {_metrics_block([r['label'] for r in labelled], [r['grade'] for r in labelled])}")
        for rep in reps_used:
            sub = [r for r in labelled if r["per_rep"].get(rep, {}).get("mean") is not None]
            if sub:
                print(f"  {rep:<9}: {_metrics_block([r['label'] for r in sub], [int(round(r['per_rep'][rep]['mean'])) for r in sub])}")
    out_dir = os.path.join(args.out_dir, EVAL_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{tag}_results.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "source", "label", "grade", "ensemble_mean"] + [f"{r}_mean" for r in reps_used]
                   + [f"{r}_folds" for r in reps_used] + ["diff", "n_notes", "n_measures", "merged_parts", "file", "notes", "error"])
        for res in results:
            pr = res.get("per_rep", {})
            w.writerow([res["name"], res["source"], res.get("label", ""), res.get("grade", ""), res.get("ensemble_mean", "")]
                       + [pr.get(r, {}).get("mean", "") for r in reps_used]
                       + [",".join("-" if g is None else str(g) for g in pr[r]["folds"]) if r in pr else "" for r in reps_used]
                       + [res.get("diff", ""), res.get("n_notes", ""), res.get("n_measures", ""), res.get("merged_parts") or "",
                          res.get("file", ""), " | ".join(res.get("notes", [])), res.get("error", "")])
    with open(os.path.join(out_dir, f"{tag}_results.json"), "w") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)
    print(f"\nresults -> {csv_path} (+ .json)")
    return results


def eval_reps_for(args, reps_trained) -> list:
    if args.eval_reps == "auto":
        dirs = checkpoint_dirs(args)
        found = [r for r in REPS if find_checkpoints(dirs, r)]
        print(f"heads with checkpoints in {dirs}: {found or 'none'}")
        return found or list(reps_trained)
    bad = [r for r in args.eval_reps.split(",") if r.strip() and r.strip() not in REPS]
    if bad:
        sys.exit(f"--eval-reps: unknown representation(s) {bad}; choose from {list(REPS)}")
    return [r.strip() for r in args.eval_reps.split(",") if r.strip()]


# =========================================================================================
# Main
# =========================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", default="/content/drive/MyDrive/train_data_tatiana",
                    help="folder with the score files (score_<id>.musicxml / .xml / .mxl)")
    ap.add_argument("--out-dir", default="/content/checkpoints")
    ap.add_argument("--cache-dir", default=None, help="parsed-feature cache (default <out-dir>/features)")
    ap.add_argument("--emb-dir", default=None,
                    help="precomputed argnn/virtuoso embeddings: DIR/<stem>-<hash>/{embed.pt,virtuoso.pt}")
    ap.add_argument("--reps", default="p", help="comma-separated heads to train: p,argnn,virtuoso")
    ap.add_argument("--folds", default="0,1,2,3,4", help="comma-separated folds 0..4")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64, help="max pieces per batch")
    ap.add_argument("--token-budget", type=int, default=120000,
                    help="max (padded_len * items) per batch; bounds memory without truncation")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip")
    ap.add_argument("--max-len", type=int, default=0,
                    help="cut every score to N notes (train: random window, eval: tail). 0 = full "
                         "length (default, faithful to the paper). Use e.g. 300 to test the pipeline fast.")
    ap.add_argument("--limit-scores", type=int, default=0,
                    help="use only the first N dataset scores (quick pipeline test; 0 = all)")
    ap.add_argument("--smoke", action="store_true",
                    help="preset for a quick test: --limit-scores 60 --max-len 300 --epochs 3 --patience 3 --folds 0")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--num-head", type=int, default=1,
                    help="context-attention heads (64%%h==0 for p/virtuoso, 128%%h==0 for argnn)")
    ap.add_argument("--no-balanced", dest="balanced", action="store_false",
                    help="disable per-grade loss weights")
    ap.add_argument("--exclude-non-piano", action="store_true",
                    help="drop the 6 scores flagged 'не для ф-но' in the workbook")
    ap.add_argument("--xlsx", default=None,
                    help="optional: re-read labels from the workbook instead of the embedded table")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="parallel parsers")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parse-only", action="store_true", help="parse + cache the features, then exit")
    ap.add_argument("--no-oof", dest="oof", action="store_false",
                    help="skip the pooled out-of-fold report (confusion + oof_predictions.csv); the per-fold "
                         "test metrics and their mean ± std are always printed")
    ap.add_argument("--resume", action="store_true",
                    help="reuse <rep>_ordinal-<fold>.pt checkpoints already in --out-dir (trained on this dataset) "
                         "instead of training them again, e.g. after a Colab disconnect")
    ap.add_argument("--predict", default=None,
                    help="comma-separated score files or https links to grade with the trained heads (no training "
                         "unless --train)")
    ap.add_argument("--eval-list", default=None,
                    help="text/CSV file with one score per line: <https link or path> [label 1-9] [name]; every "
                         "score is downloaded, parsed and graded with the trained heads (no training unless --train)")
    ap.add_argument("--eval-reps", default="auto",
                    help="heads used by --predict / --eval-list: 'auto' = every head with checkpoints in --out-dir, "
                         "or e.g. p or p,argnn,virtuoso (argnn/virtuoso need --emb-dir entries for the scores)")
    ap.add_argument("--train", action="store_true",
                    help="with --predict / --eval-list: train first (as usual), then grade with the new checkpoints")
    ap.add_argument("--ckpt-dir", default=None,
                    help="extra folder(s) with <rep>_ordinal-<fold>.pt checkpoints for --predict / --eval-list, "
                         "comma-separated (e.g. the pitch head trained in another run); --out-dir is always included")
    ap.add_argument("--force-download", action="store_true", help="re-download linked scores even if cached")
    args = ap.parse_args()
    try:                      # line-buffered logs also when stdout is a file / pipe (`| tee`)
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    if args.smoke:
        args.limit_scores = args.limit_scores or 60
        args.max_len = args.max_len or 300
        args.epochs, args.patience, args.folds = min(args.epochs, 3), min(args.patience, 3), "0"
        print("SMOKE preset: limit_scores=%d max_len=%d epochs=%d folds=%s"
              % (args.limit_scores, args.max_len, args.epochs, args.folds))
    args.out_dir = os.path.abspath(args.out_dir)
    args.cache_dir = os.path.abspath(args.cache_dir or os.path.join(args.out_dir, "features"))
    reps = [r.strip() for r in args.reps.split(",") if r.strip()]
    bad = [r for r in reps if r not in REPS]
    if bad:
        ap.error(f"unknown representation(s) {bad}; choose from {list(REPS)}")
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    if any(f < 0 or f >= N_BUCKETS for f in folds):
        ap.error(f"folds must be within 0..{N_BUCKETS - 1}")

    dev = pick_device(args.device)
    print(f"torch {torch.__version__} | device: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    eval_entries = []
    if args.eval_list:
        eval_entries = read_eval_list(args.eval_list)
        print(f"eval list: {len(eval_entries)} scores from {args.eval_list} "
              f"({sum(e['label'] is not None for e in eval_entries)} with labels)")
    predict_entries = [{"source": p.strip(), "label": None, "name": entry_stem(p.strip())}
                       for p in (args.predict or "").split(",") if p.strip()]
    if (eval_entries or predict_entries) and not args.train:
        if predict_entries:
            print(json.dumps(grade_entries(predict_entries, args, eval_reps_for(args, reps)), indent=2, ensure_ascii=False))
        if eval_entries:
            evaluate_entries(eval_entries, args, eval_reps_for(args, reps))
        return

    items = load_dataset(args)
    grades = [it["grade"] for it in items]
    print(f"dataset: {len(items)} scores | grade counts 1..9: "
          f"{[grades.count(g) for g in range(1, 10)]} | buckets: {[sum(it['bucket'] == b for it in items) for b in range(N_BUCKETS)]}")

    all_feats = {}
    for rep in reps:
        all_feats[rep] = build_p_features(items, args) if rep == "p" \
            else build_precomputed_features(items, rep, args)
        n = sum(it["stem"] in all_feats[rep] for it in items)
        print(f"{rep}: features for {n}/{len(items)} scores")
    if args.parse_only:
        return

    # per fold: train every head on train (+ val for the epoch choice), THEN grade the fold's test part
    fold_metrics, oof_preds, oof_ens, fold_of = {}, {rep: {} for rep in reps}, {}, {}
    results_path = os.path.join(args.out_dir, "fold_results.json")
    for fold in folds:
        ckpts = {rep: train_one(rep, fold, items, all_feats[rep], args) for rep in reps}
        metrics, preds, ens = test_fold(fold, ckpts, items, all_feats, args)
        fold_metrics[fold] = metrics
        for rep, p in preds.items():
            oof_preds[rep].update(p)
            fold_of.update({s: fold for s in p})
        oof_ens.update(ens)
        with open(results_path, "w") as fh:     # rewritten after every fold
            json.dump({"args": vars(args), "folds": {str(f): m for f, m in fold_metrics.items()}},
                      fh, indent=2, ensure_ascii=False)
    mean_std_by_head = report_over_folds(fold_metrics)
    with open(results_path, "w") as fh:
        json.dump({"args": vars(args), "folds": {str(f): m for f, m in fold_metrics.items()},
                   "mean_std": mean_std_by_head}, fh, indent=2, ensure_ascii=False)
    print(f"per-fold and mean ± std test metrics -> {results_path}")
    if args.oof:
        oof_report({r: p for r, p in oof_preds.items() if p}, oof_ens, fold_of, items, args)
    if predict_entries:
        print("\n===== --predict with the new checkpoints =====")
        print(json.dumps(grade_entries(predict_entries, args, eval_reps_for(args, reps)), indent=2, ensure_ascii=False))
    if eval_entries:
        print("\n===== --eval-list with the new checkpoints =====")
        evaluate_entries(eval_entries, args, eval_reps_for(args, reps))


if __name__ == "__main__":
    main()
