#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
onset_colab_train_806_grid.py -- onset_colab_train_806.py + explicit regularisation flags + a ONE-FOLD
HYPER-PARAMETER GRID (--grid) that trains many settings on the same fold(s), ranks them and writes a
pasteable summary; every (setting, seed, fold) is cached, so the run resumes after an interruption.

FLAGS ADDED
  --in-dropout P / --rnn-dropout P   dropout on the input matrix / between the GRU layers (default 0.2 / 0.2)
  --weight-decay W                    AdamW weight decay (0 = plain Adam)
  --quiet                             print only every 10th / improving epoch
  --grid oat|random|both|full         oat    = one-at-a-time: every axis value with the other axes at the
                                               baseline (= the command-line flags)             ~16 runs
                                      random = --grid-n random settings from the full product (seeded)
                                      both   = oat + random (default when --grid is given)
                                      full   = the whole product (prints the count and asks nothing)
  --grid-folds 0        the fold(s) every setting is trained on (default: bucket 0 only)
  --grid-seeds 42       the seed(s)
  --grid-n 30           random settings for random / both
  --grid-axes ...       the axes, 'name=v1,v2,...;name=...' (defaults below)
  Summary -> <out-dir>/grid/<tag>/summary.md + summary.json, ranked by acc9, with:
    acc9  balanced accuracy over the 9 grades (the headline),
    acc8m the SAME predictions scored with grades 8 and 9 merged -- comparable with the 8-grade runs,
    acc±1, acc3, macro-MSE, recall of grade 8 and grade 9, best epoch, Δ to the baseline.
  CAVEAT: one fold = ~160 test scores, a few of grade 9; the fold-to-fold std of acc9 was 0.03-0.05 in
  earlier runs, so the grid can only tell apart differences of ~0.05 and more. Use it to shortlist, then
  confirm the top settings with all folds x 3 seeds (plain run, --seed 42/43/44).

onset_colab_train_806.py -- onset_colab_train_779.py plus the 27 scores that the 806 workbook
(806_prefinal_score_dataset_difficulty_bpm.xlsx) adds: rows with a grade, a Score url and a tempo but
no musicxml_path yet (3 of grade 7, 6 of grade 8, 18 of grade 9), with NINE grades again. 2026-10-19:
  * grades 1..9 (N_CLASSES 9, MERGE_GRADES empty): grade 9 is a class of its own again, the ordinal
    head has 9 outputs and every metric / confusion matrix runs over 1..9;
  * the 779 earlier rows carry the grades of the 806 workbook (19 of them changed there) and
    KEEP their buckets, except 6 rows whose grade changed, moved so that every grade is spread
    evenly over the 5 buckets (max - min <= 1 per grade; listed above DATASET_TABLE);
  * the 27 new rows are dealt per grade into the buckets holding the fewest scores of that grade
    (ties in a seeded random order); their note count is 0 in the table because their files were
    not available when the table was made;
  * a workbook row without musicxml_path is named after its Score url (score_<id>) at run time, so
    the new scores take part as soon as score_<id>.musicxml / .xml / .mxl is in --scores-dir
    (a missing file is reported as skipped, the run goes on without it);
  * --bpm-xlsx defaults to the 806 workbook; checkpoints carry the dataset tag "806".

GRADES: 9. MERGE_GRADES is empty; the merge mechanism of the 779 file is kept for experiments
(set MERGE_GRADES = {9: 8} and N_CLASSES = 8 to fold grade 9 into 8 again). Checkpoints record n_classes.

onset_colab_train.py -- the ONSET difficulty model as ONE self-contained file for Google Colab.

What it does (the same pipeline as cipi_trainer/onset_model/, nothing imported from there):
  1. reads the dataset workbook (score_dataset_difficulty_musicxml_bpm.xlsx): the score file of every
     row (basename of "musicxml_path"), its label "Сложность (число)" (1..9; 9 is folded into 8), the "не для ф-но" flag,
     and its START TEMPO: column "bpm_db", or "bpm" when bpm_db is empty. A score without a
     usable tempo is skipped with a big warning;
  2. encodes every score with music21 into an event matrix: one row per onset, 88 right-hand key
     bits + 88 left-hand key bits + 12 + 12 local attack-rate bits (one one-hot per hand over
     log-spaced "attacks per minute" bins) + extras -- NO tempo-class bits (encoder version 3): the
     tempo map (start tempo = workbook value, every later mark of the file honoured) only drives
     the seconds clock behind the rates and intervals; one CSV per score is kept
     (<out-dir>/features/csv/) so the arrays can be validated. --rate-from ioi (default) derives
     the rate from the time to the hand's next onset, --rate-from dur from the written note
     duration. Onsets are merged on a 1e-4-quarter grid;
  3. trains on those matrices a 3-layer UNIDIRECTIONAL GRU (64 wide; --bidirectional for the
     CIPI-style BiGRU) -> context attention -> 9-way ordinal head, OrdinalLoss with per-grade
     weights, Adam 1e-3, grad-clip 1,
     full pieces in batches of --batch-size similar-length pieces, early stopping on validation
     macro-MSE / balanced Acc±1 (every Acc±1 in this file is the mean per-grade within-one rate,
     the plain rate is printed next to it) -- on the workbook's 5 stratified folds (embedded below; identical to the
     local cipi.train --dataset tatiana split) or on --n-folds N re-dealt folds;
  4. grades every score out-of-fold (by the fold in whose test bucket it lies) and prints the
     comparison with the labels (balanced Acc-9, Acc±1, acc3 = balanced accuracy over the coarse
     buckets 1-3 / 4-6 / 7+, exact/within-1/within-2, MAE, RMSE, bias,
     Spearman, per-grade table, confusion matrix); writes oof_<variant>.csv / .json;
  5. --predict FILE ... grades any MusicXML files with the trained fold ensemble.

------------------------------------------------------------------------------------------
COLAB QUICKSTART
------------------------------------------------------------------------------------------
1) Runtime -> Change runtime type -> GPU (optional; the model is small, a CPU run is ~1 min/epoch).
2) Mount Drive with the score files and the bpm workbook:
     from google.colab import drive; drive.mount('/content/drive')
3) Install the two non-default packages once per runtime:
     !pip -q install music21 openpyxl
4) Run (features are built on the first run and cached in --out-dir; writing 700 CSVs straight to
   Drive is slow, so use a local --out-dir and copy it to Drive afterwards, or accept ~5 min):
     !python /content/drive/MyDrive/onset_colab_train.py \\
         --scores-dir /content/drive/MyDrive/train_data_tatiana \\
         --bpm-xlsx  /content/drive/MyDrive/806_prefinal_score_dataset_difficulty_bpm.xlsx \\
         --out-dir   /content/onset_out --device auto
     !cp -r /content/onset_out /content/drive/MyDrive/
   Variants:  --features core+timing --transpose 3      (timing extras + key-shift augmentation)
              --rate-from dur                            (rate bits from the written durations; the
                    features are re-encoded and the checkpoints get a "_dur" suffix)
              --bidirectional --hidden 32                (the CIPI BiGRU; checkpoints get a "_bi" suffix)
              --n-folds 2                                (2-fold cross-validation)
              --smoke                                    (60 scores, 3 epochs: pipeline test)
              --predict /content/drive/MyDrive/x.musicxml --start-bpm 120   (grade a file)
              --no-train --predict-dir /content/drive/MyDrive/new_scores     (grade every score under a
                    folder, recursively, with the folds already trained; results -> <out-dir>/predictions/
                    <folder>.csv; start tempo: workbook row if the stem has one, else the file's own mark)

Outputs in --out-dir:
  features/csv/<stem>.csv        the event matrix of every score (validate these)
  features/index.json            per score: grade, bucket, start tempo + source, events, warnings
  checkpoints/<variant>_s<seed>/onset_ordinal-<fold>.pt   the heads (best epoch, written on every improvement;
                                                  one folder per --seed, so several seeds can be evaluated and
                                                  ensembled with onset_colab_eval.py)
  checkpoints/<variant>/oof_<variant>.csv/.json   out-of-fold predictions and metrics
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import traceback
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(line_buffering=True)   # progress lines show up immediately in a notebook
except Exception:  # noqa: BLE001
    pass

# =========================================================================================
# 1. The dataset table: score id, workbook grade (1..9), fold bucket (0..4), note count
#    (cipi.musicxml.parse_score_file; 0 = file not available when the table was made), "np" = not
#    for piano. 806 scores = the 779 rows of onset_colab_train_779.py (buckets kept, grades as in
#    the 806 workbook) plus the 27 new rows of 806_prefinal_score_dataset_difficulty_bpm.xlsx dealt
#    into the least-filled bucket of their grade. The bucket column IS the split used
#    (embedded_buckets() reads it); every grade is spread over the 5 buckets with max - min <= 1.
#    Grade changes taken from the 806 workbook (id old grade->new grade): 33 4->3, 83 7->6, 101 6->7, 110 6->5, 124 3->2, 130 8->7, 135 4->3, 160 4->3, 166 3->2, 177 4->3, 249 9->8, 578 7->6, 603 6->5, 615 7->6, 616 7->6, 626 5->4, 646 8->7, 651 8->7, 655 4->3.
#    Rows moved to keep the per-grade balance after those changes:
#      score_124 (grade 2): bucket 2 -> 1
#      score_33 (grade 3): bucket 1 -> 2
#      score_603 (grade 5): bucket 3 -> 0
#      score_83 (grade 6): bucket 1 -> 0
#      score_578 (grade 6): bucket 2 -> 3
#      score_616 (grade 6): bucket 1 -> 4
#    Labels are re-read from the workbook at run time; a workbook score missing here is skipped.
# =========================================================================================
EMBEDDED_TABLE_ONLY = True      # skip workbook scores that are not in DATASET_TABLE
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
N_BUCKETS = 5
DATASET_TAG = "806"                       # goes into the checkpoint folder name (checkpoints/<variant>)
SPLIT_SEED = 42                           # seed of the frozen re-deal and of the new rows' tie-breaks
MERGE_GRADES = {}                         # workbook grade -> grade used here (empty: all nine grades kept)
N_CLASSES = 9                             # grades 1..9 (ordinal head width)
URL_COLUMN = "Score url"                  # names a score (score_<id>) when musicxml_path is still empty
LABEL_COLUMN = "Сложность (число)"
PATH_COLUMN = "musicxml_path"
NOT_PIANO_COLUMN = "не для ф-но"
BPM_COLUMNS = ("bpm_db", "bpm")          # first non-empty wins


def merge_grade(grade) -> int:
    """The grade the model is trained on: the workbook grade with MERGE_GRADES applied (9 -> 8)."""
    g = int(grade)
    return MERGE_GRADES.get(g, g)


def embedded_table() -> dict:
    """stem -> merged grade for every row of DATASET_TABLE."""
    out = {}
    for line in DATASET_TABLE.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out[f"score_{parts[0]}"] = merge_grade(parts[1])
    return out


def embedded_buckets() -> dict:
    """stem -> bucket 0..4 from the bucket column of DATASET_TABLE: the frozen split described in the
    file docstring (stratified_buckets() is only used for --n-folds != 5)."""
    out = {}
    for line in DATASET_TABLE.strip().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out[f"score_{parts[0]}"] = int(parts[2])
    return out


def read_workbook(xlsx_path: str) -> list:
    """-> [{'stem', 'grade', 'bucket', 'not_piano', 'bpm', 'bpm_source'}] for every row with a file
    and a grade 1..9; the grade is merged (9 -> 8, MERGE_GRADES) and the bucket comes from the
    re-dealt split (embedded_buckets). bpm is None (with the reason in bpm_source) when the row has
    no usable tempo."""
    try:
        import openpyxl
    except ImportError:
        sys.exit("reading the workbook needs openpyxl:  !pip install openpyxl")
    ws = openpyxl.load_workbook(xlsx_path, read_only=True).worksheets[0]
    rows = ws.iter_rows(values_only=True)
    header = [str(c) if c is not None else "" for c in next(rows)]
    col = {name: header.index(name) for name in (LABEL_COLUMN, PATH_COLUMN, NOT_PIANO_COLUMN) + BPM_COLUMNS
           if name in header}
    for need in (LABEL_COLUMN, PATH_COLUMN):
        if need not in col:
            sys.exit(f"{xlsx_path}: column {need!r} not found; columns are {header}")
    if not any(c in col for c in BPM_COLUMNS):
        sys.exit(f"{xlsx_path}: needs a tempo column ({' / '.join(BPM_COLUMNS)})")
    buckets = embedded_buckets()
    url_col = header.index(URL_COLUMN) if URL_COLUMN in header else None
    items, bad_grade, not_in_table, n_merged, n_from_url = [], 0, 0, 0, 0
    for r in rows:
        p, g = r[col[PATH_COLUMN]], r[col[LABEL_COLUMN]]
        if g is None:
            continue
        if p:
            stem = os.path.splitext(os.path.basename(str(p)))[0]
        else:                                  # no musicxml_path yet: the score id of the URL names the file
            mid = re.search(r"/scores/(\d+)", str(r[url_col] or "")) if url_col is not None else None
            if not mid:
                continue
            stem = f"score_{mid.group(1)}"
            n_from_url += 1
        if EMBEDDED_TABLE_ONLY and stem not in buckets:
            not_in_table += 1
            continue
        try:
            raw_grade = int(g)
        except (TypeError, ValueError):
            bad_grade += 1
            continue
        grade = merge_grade(raw_grade)
        if not 1 <= grade <= N_CLASSES:
            bad_grade += 1
            continue
        n_merged += grade != raw_grade
        bpm, src = None, f"no value in {' / '.join(BPM_COLUMNS)}"
        for c in BPM_COLUMNS:
            if c in col and r[col[c]] not in (None, ""):
                try:
                    v = float(r[col[c]])
                except (TypeError, ValueError):
                    src = f"{c}={r[col[c]]!r} is not a number"
                    continue
                if v > 0:
                    bpm, src = v, c
                    break
                src = f"{c}={v:g} is not a positive tempo"
        items.append({"stem": stem, "grade": grade,
                      "bucket": buckets.get(stem, zlib.crc32(stem.encode()) % N_BUCKETS),
                      "not_piano": bool(r[col[NOT_PIANO_COLUMN]]) if NOT_PIANO_COLUMN in col else False,
                      "bpm": bpm, "bpm_source": src})
    items.sort(key=lambda it: it["stem"])
    n_new = sum(it["stem"] not in buckets for it in items)
    print(f"workbook {os.path.basename(xlsx_path)}: {len(items)} scores with a grade (1..{N_CLASSES}"
          + (f"; {n_merged} scores of grade {', '.join(map(str, MERGE_GRADES))} counted as grade "
             f"{', '.join(map(str, MERGE_GRADES.values()))}" if n_merged else "") + ")"
          + (f", {bad_grade} rows with an unusable grade skipped" if bad_grade else "")
          + (f", {n_from_url} rows without musicxml_path named after their Score url" if n_from_url else "")
          + (f", {n_new} not in the embedded split (bucket by hash)" if n_new else "")
          + (f", {not_in_table} workbook scores not in the embedded table skipped" if not_in_table else "")
          + f"; start tempo available for {sum(it['bpm'] is not None for it in items)}")
    return items


def describe(items: list) -> str:
    grades = [it["grade"] for it in items]
    return "grades: " + " ".join(f"{g}:{grades.count(g)}" for g in range(1, N_CLASSES + 1)) + \
        " | buckets: " + " ".join(f"{b}:{sum(it['bucket'] == b for it in items)}" for b in range(N_BUCKETS))


# =========================================================================================
# 2. The encoder: MusicXML -> event matrix (encode_score.py)
# =========================================================================================
RATE_EDGES = [30, 45, 70, 100, 150, 220, 330, 500, 750, 1100, 1600]   # attacks per minute, 12 bins
RATE_NAMES = ["lt30", "30_45", "45_70", "70_100", "100_150", "150_220", "220_330", "330_500",
              "500_750", "750_1100", "1100_1600", "gt1600"]
RATE_FROM = ("ioi", "dur")         # local rate from the inter-onset interval or from the written duration
ENCODER_VERSION = 3                # 1-2: with 11 tempo-class bits; 3: keys + rate bits only
# ---- staff selection overrides: stem -> (right-hand staff, left-hand staff), 0-based indices into ALL
#      staves of the file (music21 parts, empty ones included). For scores whose first staves are an ossia,
#      an inserted system or an empty part, so that "first two staves with notes" picks the wrong pair.
STAFF_OVERRIDE = {
    "score_17": (2, 3),    # staves 1-2 = ossia of bars 7-8 (30 notes); the piece is on staves 3-4
    "score_531": (1, 2),   # staff 1 = ossia of 4 bars (46 notes); hands on staves 2-3
    "score_67": (1, 2),    # staff 1 = empty part; hands on staves 2-3 (the notes filter handles it too)
    "score_14": (1, 2),    # staff 1 = 2-bar insert (148 notes); hands on staves 2-3
}
# ---- scores whose internal tempo marks are playback automation (dozens of bare <metronome> marks hidden
#      in MuseScore), not notation: only the START tempo is used for the whole piece
SINGLE_TEMPO_SCORES = {
    "score_26",            # Rachmaninoff Op. 23/9: 68 marks 120<->154 per bar + rit./accel. automation; printed: Presto 152
}
BLACK = {1, 3, 6, 8, 10}          # pitch classes of black keys
LO, HI = 21, 108                   # MIDI range of the 88 keys
DEFAULT_BPM = 100.0                # --file-tempo fallback when the file has no mark at all
KEY_COLUMNS = [f"rh_k{i:02d}" for i in range(1, 89)] + [f"lh_k{i:02d}" for i in range(1, 89)]
RATE_COLUMNS = [f"rh_rate_{n}" for n in RATE_NAMES] + [f"lh_rate_{n}" for n in RATE_NAMES]
CORE_COLUMNS = KEY_COLUMNS + RATE_COLUMNS                     # the 200 model inputs
EXTRA_COLUMNS = ["offset_quarters", "t_sec", "bpm", "beat_in_bar", "is_downbeat", "ts_num", "ts_den",
                 "rh_playing", "lh_playing", "both",
                 "rh_n_notes", "lh_n_notes", "rh_span", "lh_span", "rh_black", "lh_black",
                 "ioi_prev_sec", "rh_ioi_next_sec", "lh_ioi_next_sec",
                 "rh_dur_quarters", "lh_dur_quarters", "rh_rate_npm", "lh_rate_npm"]
HEADER = ["measure"] + CORE_COLUMNS + EXTRA_COLUMNS
SANITIZE_TAGS = ("harmony", "arpeggiate", "non-arpeggiate")   # music21 8.x crashes on these; no notes in them


def rate_onehot(rate):
    v = [0] * (len(RATE_EDGES) + 1)
    if rate is None:
        return v
    i = sum(1 for e in RATE_EDGES if rate >= e)
    v[i] = 1
    return v


def warn_skip(src, reason):
    bar = "!" * 88
    print(f"\n{bar}\n!!  SKIPPED: {os.path.basename(src)}\n!!  {reason}\n{bar}\n", file=sys.stderr, flush=True)


def _read_tree(src):
    import xml.etree.ElementTree as ET
    if str(src).lower().endswith(".mxl"):
        with zipfile.ZipFile(src) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".xml", ".musicxml")) and not n.startswith("META-INF")]
            try:
                container = ET.fromstring(z.read("META-INF/container.xml"))
                names = [container.find(".//rootfile").get("full-path")] + names
            except Exception:  # noqa: BLE001
                pass
            return ET.ElementTree(ET.fromstring(z.read(names[0])))
    return ET.parse(src)


def _metronome_direction(bpm: str):
    import xml.etree.ElementTree as ET
    d = ET.Element("direction")
    dt = ET.SubElement(d, "direction-type")
    m = ET.SubElement(dt, "metronome")
    ET.SubElement(m, "beat-unit").text = "quarter"
    ET.SubElement(m, "per-minute").text = bpm
    return d


def normalize_sound_tempo(root) -> int:
    """Tempo marks independent of the music21 version: MuseScore writes the playback tempo as
    <sound tempo="X"/> (quarter-note bpm) next to a printed <metronome> or, for a text-only tempo
    such as "Allegro", on its own; music21 8.x ignores it, 9+ turns it into a MetronomeMark. Every
    <sound tempo> without a printed <metronome> in the same direction becomes an explicit
    quarter-note <metronome> direction at the same place; the tempo attribute is dropped everywhere
    (a printed mark stays the single source). Returns the number of marks created."""
    n = 0
    for measure in root.iter("measure"):
        for el in list(measure):
            if el.tag == "direction":
                sound = el.find("sound")
                if sound is None or not sound.get("tempo"):
                    continue
                if el.find("direction-type/metronome") is None:
                    dt = _metronome_direction(sound.get("tempo"))[0]
                    after = [j for j, c in enumerate(list(el)) if c.tag == "direction-type"]
                    el.insert((after[-1] + 1) if after else 0, dt)
                    n += 1
                del sound.attrib["tempo"]
            elif el.tag == "sound" and el.get("tempo"):          # standalone <sound tempo/> in the measure
                measure.insert(list(measure).index(el), _metronome_direction(el.get("tempo")))
                del el.attrib["tempo"]
                n += 1
    return n


def drop_bad_display_steps(root) -> int:
    """Remove the <display-step>/<display-octave> of rests and unpitched notes whose step is not
    A..G (some exports write "?" or nothing): they only place the glyph vertically, carry no onset
    or pitch, and make music21 raise "Cannot make a step out of '?'". Returns the count removed."""
    n = 0
    for el in root.iter():
        if el.tag not in ("rest", "unpitched"):
            continue
        step = el.find("display-step")
        if step is not None and (step.text or "").strip().upper() not in list("ABCDEFG"):
            for child in list(el):
                if child.tag in ("display-step", "display-octave"):
                    el.remove(child)
                    n += 1
    return n


def sanitized_copy(src, tags=SANITIZE_TAGS):
    """(path_to_parse, {tag: count}); the original path when nothing had to change, else a
    temporary .musicxml copy without chord symbols / arpeggio brackets and with playback-only
    tempos turned into explicit metronome marks (key 'sound_tempo_to_metronome')."""
    tree = _read_tree(src)
    removed = {}
    for parent in tree.getroot().iter():
        for child in list(parent):
            if child.tag in tags:
                parent.remove(child)
                removed[child.tag] = removed.get(child.tag, 0) + 1
    n_disp = drop_bad_display_steps(tree.getroot())
    if n_disp:
        removed["bad_display_step"] = n_disp
    n_tempo = normalize_sound_tempo(tree.getroot())
    if n_tempo:
        removed["sound_tempo_to_metronome"] = n_tempo
    if not removed:
        return src, removed
    fd, path = tempfile.mkstemp(prefix=os.path.splitext(os.path.basename(src))[0] + "_", suffix=".musicxml")
    os.close(fd)
    tree.write(path, encoding="UTF-8", xml_declaration=True)
    return path, removed


def encode(src, out=None, start_bpm=None, file_tempo=False, sanitize=True, rate_from="ioi"):
    """One MusicXML file -> (rows, HEADER, marks, info), CSV written to `out` when given; None when
    the score is skipped (no start tempo, fewer than two staves with notes).
    start_bpm: the workbook tempo (quarter-note bpm) in force at offset 0; later marks of the file
    are kept. file_tempo=True: the file's own first mark instead (DEFAULT_BPM when none).
    rate_from: "ioi" (60 / seconds to the hand's next onset) or "dur" (bpm / written duration)."""
    try:
        from music21 import converter, meter, tempo
    except ImportError:
        sys.exit("music21 is needed to encode scores:  !pip install music21")
    if rate_from not in RATE_FROM:
        raise ValueError(f"rate_from must be one of {RATE_FROM}, not {rate_from!r}")
    info = {"file": src, "warnings": [], "rate_from": rate_from, "encoder_version": ENCODER_VERSION}
    if start_bpm is None and not file_tempo:
        warn_skip(src, "no start tempo given (workbook row missing or without bpm_db / bpm)")
        return None
    parse_path, removed = sanitized_copy(src) if sanitize else (src, {})
    if removed:
        info["sanitized"] = removed
    try:
        s = converter.parse(parse_path)
    finally:
        if parse_path != src:
            os.remove(parse_path)

    # hands = the first two staves that contain notes (an empty part, e.g. a dropped instrument,
    # is ignored; two single-hand parts land as rh / lh in file order) -- unless the score is in
    # STAFF_OVERRIDE, which names the two staves explicitly
    all_parts = list(s.parts)
    stem = os.path.splitext(os.path.basename(src))[0]
    info["n_staves"] = len(all_parts)
    if stem in STAFF_OVERRIDE and max(STAFF_OVERRIDE[stem]) < len(all_parts):
        i, j = STAFF_OVERRIDE[stem]
        rh_part, lh_part = all_parts[i], all_parts[j]
        info["staves_used"] = [i + 1, j + 1]
        info["warnings"].append(f"STAFF_OVERRIDE: using staves {i + 1} (rh) and {j + 1} (lh) of {len(all_parts)}")
    else:
        if stem in STAFF_OVERRIDE:
            info["warnings"].append(f"STAFF_OVERRIDE ignored: the file has only {len(all_parts)} staves")
        parts = [p for p in all_parts if len(p.recurse().notes) > 0]
        if len(parts) < 2:
            warn_skip(src, f"expected two staves with notes (right hand / left hand), found {len(parts)}")
            return None
        if len(parts) > 2:
            info["warnings"].append(f"{len(parts)} staves/parts with notes: using the first two as rh / lh")
        rh_part, lh_part = parts[0], parts[1]
        info["staves_used"] = [all_parts.index(rh_part) + 1, all_parts.index(lh_part) + 1]

    # ---- tempo map (offset in quarters -> quarter-note bpm); a later mark at the same offset wins
    Q = lambda x: round(float(x), 4)          # 1e-4-quarter grid  # noqa: E731
    tm = {}
    for mm in s.recurse().getElementsByClass(tempo.MetronomeMark):
        if mm.getQuarterBPM():
            tm[Q(mm.getOffsetInHierarchy(s))] = float(mm.getQuarterBPM())
    file_marks = sorted(tm.items())
    if stem in SINGLE_TEMPO_SCORES and len(file_marks) > 1:
        info["warnings"].append(f"SINGLE_TEMPO_SCORES: {len(file_marks) - 1} internal tempo marks ignored, "
                                f"one tempo for the whole piece")
        file_marks = file_marks[:1]
    if start_bpm is None:                                # --file-tempo
        marks = file_marks or [(0.0, DEFAULT_BPM)]
        if marks[0][0] != 0:
            marks.insert(0, (0.0, marks[0][1]))
        bpm_src = "file"
    else:
        replaced = [b for o, b in file_marks if o == 0]
        marks = [(0.0, float(start_bpm))] + [(o, b) for o, b in file_marks if o > 0]
        bpm_src = "workbook"
        if replaced and abs(replaced[-1] - start_bpm) > 0.5:
            info["warnings"].append(f"file's initial mark {replaced[-1]:g} bpm replaced by {start_bpm:g}")
    info.update({"start_bpm": marks[0][1], "start_bpm_source": bpm_src,
                 "file_marks": [(o, b) for o, b in file_marks], "n_marks": len(marks)})

    def sec_at(t):
        total = 0.0
        for (o, b), (o2, _) in zip(marks, marks[1:] + [(None, None)]):
            end = t if o2 is None or o2 > t else o2
            if end > o:
                total += float(end - o) * 60 / b
            if o2 is None or o2 >= t:
                break
        return total

    def bpm_at(t):
        return [b for o, b in marks if o <= t][-1]

    # Bars split across a system break (MuseScore exports the two halves as two consecutive <measure>
    # elements: the first ends with an invisible barline, both are shorter than the bar) are merged back
    # into one bar, so the measure number, the downbeat and the beat position stay right.
    bars, ts_map, n_merged = {}, {}, 0
    prev, bar_len, number = None, None, 0
    for m in rh_part.getElementsByClass('Measure'):
        off = Q(m.offset)
        ts = m.getElementsByClass(meter.TimeSignature)
        if ts:
            bar_len = float(ts[0].barDuration.quarterLength)
            ts_map[off] = (ts[0].numerator, ts[0].denominator)
        cont = (prev is not None and bar_len is not None
                and any(b.type == "none" for b in prev.getElementsByClass('Barline')
                        if getattr(b, "location", "right") in ("right", None))
                and float(prev.duration.quarterLength) < bar_len - 1e-6
                and float(prev.duration.quarterLength) + float(m.duration.quarterLength) <= bar_len + 1e-6)
        if cont:
            n_merged += 1                                 # second half of a split bar: no new bar starts here
        else:
            number += 1
            bars[off] = number
        prev = m
    bar_starts = sorted(bars)
    info["n_merged_split_bars"] = n_merged
    if n_merged:
        info["warnings"].append(f"{n_merged} bars split across a system break were merged back")

    def bar_info(t):
        b = [k for k in bar_starts if k <= t][-1]
        ts = [ts_map[k] for k in sorted(ts_map) if k <= b]
        return bars[b], t - b, (ts[-1] if ts else (4, 4))

    def events(part):                                  # offset -> pitches, offset -> shortest written duration
        d, dur = {}, {}
        for x in part.stripTies(matchByPitch=False).flatten().notes:
            ql = float(x.duration.quarterLength)
            if ql == 0:                                 # grace notes
                continue
            off = Q(x.offset)
            d.setdefault(off, set()).update(p.midi for p in x.pitches if LO <= p.midi <= HI)
            dur[off] = min(dur.get(off, ql), ql)
        return {k: v for k, v in d.items() if v}, dur

    (rh, rh_dur), (lh, lh_dur) = events(rh_part), events(lh_part)
    rh_keys, lh_keys = sorted(rh), sorted(lh)
    if not rh and not lh:
        warn_skip(src, "no notes in either staff")
        return None

    def next_onset(keys, t):
        later = [k for k in keys if k > t]
        return later[0] if later else None

    rows, prev_t = [], None
    for t in sorted(set(rh) | set(lh)):
        r, l = rh.get(t, set()), lh.get(t, set())
        r_vec = [1 if (k + LO - 1) in r else 0 for k in range(1, 89)]
        l_vec = [1 if (k + LO - 1) in l else 0 for k in range(1, 89)]
        bpm = bpm_at(t)
        t_sec = sec_at(t)
        bar, beat, (tsn, tsd) = bar_info(t)
        rn, ln = (next_onset(rh_keys, t) if r else None), (next_onset(lh_keys, t) if l else None)
        r_ioi = (sec_at(rn) - t_sec) if rn else None
        l_ioi = (sec_at(ln) - t_sec) if ln else None
        if rate_from == "ioi":
            r_rate = 60 / r_ioi if r_ioi else None
            l_rate = 60 / l_ioi if l_ioi else None
        else:
            r_rate = bpm / rh_dur[t] if r else None
            l_rate = bpm / lh_dur[t] if l else None
        f = lambda x, n=4: round(x, n) if x is not None else ""  # noqa: E731
        rows.append([bar] + r_vec + l_vec + rate_onehot(r_rate) + rate_onehot(l_rate) + [
            f(t, 4), f(t_sec), bpm, f(float(beat), 3), int(abs(beat) < 1e-6), tsn, tsd,
            int(bool(r)), int(bool(l)), int(bool(r) and bool(l)),
            len(r), len(l),
            (max(r) - min(r)) if r else "", (max(l) - min(l)) if l else "",
            sum(1 for p in r if p % 12 in BLACK), sum(1 for p in l if p % 12 in BLACK),
            f(t_sec - prev_t) if prev_t is not None else "",
            f(r_ioi), f(l_ioi),
            f(rh_dur[t], 4) if r else "", f(lh_dur[t], 4) if l else "", f(r_rate, 1), f(l_rate, 1)])
        prev_t = t_sec
    info["n_events"] = len(rows)
    if out:
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HEADER)
            w.writerows(rows)
    return rows, HEADER, marks, info


# =========================================================================================
# 3. Feature building over the dataset (build_features.py): CSV per score + index.json
# =========================================================================================
SCORE_EXTS = (".musicxml", ".xml", ".mxl")
FEATURES_README = """Event matrices of the workbook dataset, one CSV per score (see the file docstring).
Columns: measure | rh_k01..rh_k88 | lh_k01..lh_k88 | rh_rate_* (12 bits) | lh_rate_* (12 bits) |
extras (offset_quarters, t_sec, bpm, beat_in_bar, is_downbeat, ts_num, ts_den, rh_playing, lh_playing, both,
rh_n_notes, lh_n_notes, rh_span, lh_span, rh_black, lh_black, ioi_prev_sec, rh_ioi_next_sec, lh_ioi_next_sec,
rh_dur_quarters, lh_dur_quarters, rh_rate_npm, lh_rate_npm).
The rate bits are the one-hot class of the hand's local attack rate in attacks per minute (log-spaced bins
<30 .. >1600; from the time to the hand's next onset with --rate-from ioi, from the written duration with
--rate-from dur); index.json["encoder"] records the settings used. No tempo class is in the matrix (encoder
version 3): the tempo in force is only the extras column bpm and acts through the seconds-based intervals.
The start tempo is the workbook's bpm_db (else bpm); later tempo marks of the file are honoured.
"""


def resolve_score(scores_dir: str, stem: str):
    for ext in SCORE_EXTS:
        p = os.path.join(scores_dir, stem + ext)
        if os.path.exists(p):
            return p
    hits = [h for h in glob.glob(os.path.join(scores_dir, "**", stem + ".*"), recursive=True)
            if h.lower().endswith(SCORE_EXTS)]
    return sorted(hits)[0] if hits else None


def _encode_job(stem: str, path: str, out_csv: str, bpm, bpm_source: str, rate_from: str = "ioi") -> dict:
    rec = {"stem": stem, "source": path}
    try:
        if bpm is None:
            warn_skip(path, f"no start tempo in the workbook: {bpm_source}")
            rec.update({"status": "skipped", "reason": f"no start tempo: {bpm_source}"})
            return rec
        res = encode(path, out_csv, start_bpm=bpm, rate_from=rate_from)
        if res is None:
            rec.update({"status": "skipped", "reason": "not two staves with notes (see the warning above)"})
            return rec
        rows, _, marks, info = res
        rec.update({"status": "ok", "n_events": len(rows), "start_bpm": info["start_bpm"],
                    "start_bpm_source": bpm_source, "file_marks": info["file_marks"], "n_marks": len(marks),
                    "warnings": info["warnings"], "sanitized": info.get("sanitized", {}),
                    "n_staves": info.get("n_staves"), "staves_used": info.get("staves_used"),
                    "n_merged_split_bars": info.get("n_merged_split_bars", 0),
                    "csv": os.path.join("csv", os.path.basename(out_csv))})
    except Exception as exc:  # noqa: BLE001
        rec.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:]})
    return rec


def build_features(items: list, scores_dir: str, features_dir: str, workers: int,
                   force: bool = False, rate_from: str = "ioi") -> dict:
    """Encodes every dataset score that has no CSV yet; returns the index {'encoder','scores','skipped',
    'errors'}. A folder encoded with other settings (version, rate_from) is redone."""
    csv_dir = os.path.join(features_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    with open(os.path.join(features_dir, "README.txt"), "w") as fh:
        fh.write(FEATURES_README)
    encoder = {"version": ENCODER_VERSION, "rate_from": rate_from, "n_columns": len(HEADER)}
    index_path = os.path.join(features_dir, "index.json")
    index = {"scores": {}, "skipped": {}, "errors": {}}
    if os.path.exists(index_path) and not force:
        try:
            old = json.load(open(index_path))
            if old.get("encoder") == encoder:
                index = old
            else:
                print(f"features: encoder settings changed ({old.get('encoder', 'version 1')} -> {encoder}): "
                      f"re-encoding every score")
                force = True
        except json.JSONDecodeError:
            pass
    index["encoder"] = encoder
    labels = {it["stem"]: it for it in items}
    jobs = []
    for it in items:
        path = resolve_score(scores_dir, it["stem"])
        if path is None:
            index["skipped"][it["stem"]] = {"status": "skipped", "reason": f"no score file in {scores_dir}"}
            continue
        out_csv = os.path.join(csv_dir, it["stem"] + ".csv")
        if not force and it["stem"] in index["scores"] and os.path.exists(out_csv):
            continue
        jobs.append((it["stem"], path, out_csv, it["bpm"], it["bpm_source"], rate_from))
    print(f"features: {len(index['scores'])} scores already encoded in {features_dir}; encoding {len(jobs)} "
          f"with {workers} worker(s) (encoder v{ENCODER_VERSION}, rate from {rate_from})")

    def record(rec):
        stem = rec.pop("stem")
        it = labels[stem]
        rec.update({"grade": it["grade"], "bucket": it["bucket"], "not_piano": it["not_piano"]})
        for k in ("scores", "skipped", "errors"):
            index[k].pop(stem, None)
        index[{"ok": "scores", "skipped": "skipped", "error": "errors"}[rec["status"]]][stem] = rec

    t0, done = time.time(), 0
    if workers > 1 and len(jobs) > 1:
        try:
            with ProcessPoolExecutor(workers) as ex:
                for fut in as_completed([ex.submit(_encode_job, *j) for j in jobs]):
                    record(fut.result())
                    done += 1
                    if done % 25 == 0 or done == len(jobs):
                        print(f"  [{done}/{len(jobs)}] {time.time() - t0:.0f}s")
                        json.dump(index, open(index_path, "w"), indent=1, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001  (a notebook without a __main__ module cannot spawn)
            print(f"  parallel encoding failed ({type(exc).__name__}); continuing serially")
            workers = 1
    if workers <= 1 or (jobs and done < len(jobs)):
        for j in jobs:
            if j[0] in index["scores"] or j[0] in index["skipped"] or j[0] in index["errors"]:
                continue
            record(_encode_job(*j))
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {time.time() - t0:.0f}s")
                json.dump(index, open(index_path, "w"), indent=1, ensure_ascii=False)
    json.dump(index, open(index_path, "w"), indent=1, ensure_ascii=False)
    n_ok, n_skip, n_err = len(index["scores"]), len(index["skipped"]), len(index["errors"])
    print(f"features: {n_ok} scores encoded, {n_skip} skipped, {n_err} errors -> {index_path}")
    for stem, rec in list(index["skipped"].items())[:20]:
        print(f"  SKIPPED {stem}: {rec.get('reason')}")
    for stem, rec in list(index["errors"].items())[:20]:
        print(f"  ERROR   {stem}: {rec.get('reason')}")
    if n_ok:
        ev = sorted(r["n_events"] for r in index["scores"].values())
        print(f"  events per score: min {ev[0]} median {ev[len(ev) // 2]} max {ev[-1]}; "
              f"{sum(1 for r in index['scores'].values() if r.get('sanitized'))} sanitized")
    return index


# =========================================================================================
# 4. CSV -> tensors, feature sets, dataset, batching (data.py)
# =========================================================================================
TIMING_COLUMNS = ["log_ioi_prev", "log_rh_ioi_next", "log_lh_ioi_next", "rh_n_notes", "lh_n_notes",
                  "rh_span", "lh_span", "rh_black", "lh_black", "both", "is_downbeat", "beat_frac"]
FEATURE_SETS = {   # name -> (one-hot CSV columns, timing values appended?)
    "core": (KEY_COLUMNS + RATE_COLUMNS, False),           # 200
    "core+timing": (KEY_COLUMNS + RATE_COLUMNS, True),     # 212 (no log-bpm value: no tempo enters the model directly)
}
FEATURE_DIMS = {name: len(cols) + (len(TIMING_COLUMNS) if timing else 0) for name, (cols, timing) in FEATURE_SETS.items()}
N_KEYS = 88
_COL = {name: i for i, name in enumerate(HEADER)}
_IDX = {name: [_COL[c] for c in cols] for name, (cols, _) in FEATURE_SETS.items()}


def resolve_feature_set(meta: dict) -> str:
    """The feature set a checkpoint was trained with, after checking it was trained on this encoder
    layout (encoder_version 3); checkpoints of versions 1-2 (tempo-class bits) are refused."""
    version = meta.get("encoder_version", 1)
    if version != ENCODER_VERSION:
        sys.exit(f"checkpoint trained with encoder version {version} (features {meta.get('features')!r}, "
                 f"{meta.get('input_size')} inputs: tempo-class bits); this file is encoder version {ENCODER_VERSION} "
                 f"without the tempo block -- re-train")
    name, size = meta.get("features", "core"), meta.get("input_size")
    if name not in FEATURE_DIMS or (size is not None and FEATURE_DIMS[name] != size):
        sys.exit(f"checkpoint features {name!r} with {size} inputs match no feature set {FEATURE_DIMS}")
    return name


def ensemble_encoder(metas: list) -> dict:
    """Encoder settings (rate_from, version) the checkpoints were trained with; they must agree."""
    cfgs = {(m.get("rate_from", "ioi"), m.get("encoder_version", 1)) for m in metas}
    if len(cfgs) > 1:
        sys.exit(f"the checkpoints disagree on the encoder settings (rate_from, version): {sorted(cfgs)}")
    rate_from, version = next(iter(cfgs))
    return {"rate_from": rate_from, "version": version}


def _f(v) -> float:
    return float(v) if v not in ("", None) else 0.0


def timing_values(row) -> list:
    g = lambda name: _f(row[_COL[name]])  # noqa: E731
    tsn, tsd = g("ts_num"), g("ts_den")
    bar_len = tsn * 4.0 / tsd if tsn and tsd else 4.0
    return [math.log1p(g("ioi_prev_sec")), math.log1p(g("rh_ioi_next_sec")), math.log1p(g("lh_ioi_next_sec")),
            g("rh_n_notes") / 4, g("lh_n_notes") / 4, g("rh_span") / 12, g("lh_span") / 12,
            g("rh_black") / 4, g("lh_black") / 4, g("both"), g("is_downbeat"),
            min(g("beat_in_bar") / bar_len, 1.0)]


def matrix_from_rows(rows: list, feature_set: str) -> torch.Tensor:
    idx, timing = _IDX[feature_set], FEATURE_SETS[feature_set][1]
    core = [[_f(r[i]) for i in idx] for r in rows]
    if timing:
        core = [c + timing_values(r) for c, r in zip(core, rows)]
    return torch.tensor(core, dtype=torch.float32)


def read_csv(path: str, feature_set: str) -> torch.Tensor:
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != HEADER:
            raise ValueError(f"{path}: unexpected columns (another encoder version? use --force-features)")
        return matrix_from_rows(list(reader), feature_set)


def load_features(features_dir: str, index: dict, feature_set: str) -> dict:
    """stem -> (T, D) tensor for every encoded score; cached in features/cache_<set>.pt together with
    the encoder settings, so a cache from other settings or another width is rebuilt."""
    recs, enc = index["scores"], index.get("encoder", {"version": 1})
    if enc.get("version") != ENCODER_VERSION:
        sys.exit(f"{features_dir} was built by encoder version {enc.get('version')}, this file is version "
                 f"{ENCODER_VERSION}: run again with --force-features")
    dim = FEATURE_DIMS[feature_set]
    cache_path = os.path.join(features_dir, f"cache_{feature_set.replace('+', '_')}.pt")
    feats, stale = {}, []
    if os.path.exists(cache_path):
        try:
            cached = torch.load(cache_path, weights_only=False)
        except Exception as exc:  # noqa: BLE001  (a cache cut short by an interrupted save or a Drive sync)
            print(f"  cache {os.path.basename(cache_path)} is unreadable ({type(exc).__name__}): rebuilding it from the CSVs")
            cached = {}
        cached = cached.get("feats", {}) if isinstance(cached, dict) and cached.get("encoder") == enc else {}
        for s in recs:
            if s in cached and tuple(cached[s].shape) == (recs[s]["n_events"], dim):
                feats[s] = cached[s]
            else:
                stale.append(s)
    else:
        stale = list(recs)
    for i, s in enumerate(stale):
        feats[s] = read_csv(os.path.join(features_dir, recs[s]["csv"]), feature_set)
        if (i + 1) % 100 == 0:
            print(f"  reading CSVs: {i + 1}/{len(stale)}")
    if stale:
        torch.save({"encoder": enc, "feats": feats}, cache_path + ".tmp")
        os.replace(cache_path + ".tmp", cache_path)          # atomic: never leaves a half-written cache behind
    print(f"features '{feature_set}' ({dim} dims, encoder v{enc['version']}, rate from {enc.get('rate_from', 'ioi')}): "
          f"{len(feats)} scores ({len(stale)} read from CSV, rest from cache)")
    return feats


def transpose(x: torch.Tensor, k: int) -> torch.Tensor:
    """Shift both hands' key vectors by k semitones (notes leaving the keyboard are dropped)."""
    if k == 0:
        return x
    y = x.clone()
    y[:, : 2 * N_KEYS] = 0
    for h in (0, N_KEYS):
        if k > 0:
            y[:, h + k: h + N_KEYS] = x[:, h: h + N_KEYS - k]
        else:
            y[:, h: h + N_KEYS + k] = x[:, h - k: h + N_KEYS]
    return y


class OnsetDataset(Dataset):
    """(features (T, D), target 0..8). max_len > 0 cuts to that many events (random window when
    training, tail otherwise); transpose > 0 shifts the keys by a random amount in [-k, k] when training."""

    def __init__(self, items: list, feats: dict, max_len: int = 0, train: bool = False, transpose: int = 0):
        self.items = [(it["stem"], int(it["grade"])) for it in items if it["stem"] in feats]
        self.feats, self.max_len, self.train, self.transpose = feats, max_len, train, transpose

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem, grade = self.items[i]
        x = self.feats[stem]
        if self.max_len and x.shape[0] > self.max_len:
            start = random.randint(0, x.shape[0] - self.max_len) if self.train else x.shape[0] - self.max_len
            x = x[start: start + self.max_len]
        if self.train and self.transpose:
            x = transpose(x, random.randint(-self.transpose, self.transpose))
        return x, grade - 1


class LengthBatchSampler:
    """Batches of up to `max_batch` pieces of similar length (sorted by length, so padding stays
    small); the only limits are --batch-size and --max-len."""

    def __init__(self, lengths, max_batch: int, shuffle: bool):
        self.shuffle = shuffle
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches = [order[i: i + max_batch] for i in range(0, len(order), max_batch)]

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(order)
        for b in order:
            yield self.batches[b]

    def __len__(self):
        return len(self.batches)


def collate(batch):
    feats, ys = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    return pad_sequence(feats, batch_first=True), lengths, torch.tensor(ys, dtype=torch.long)


def make_loader(items: list, feats: dict, max_len: int, batch_size: int, train: bool, transpose: int = 0):
    ds = OnsetDataset(items, feats, max_len=max_len, train=train, transpose=transpose)
    lengths = [min(feats[s].shape[0], max_len or 10 ** 9) for s, _ in ds.items]
    return DataLoader(ds, batch_sampler=LengthBatchSampler(lengths, batch_size, shuffle=train), collate_fn=collate)


# =========================================================================================
# 5. Model: BiGRU -> context attention -> ordinal head; OrdinalLoss (CIPI, cipi/models.py)
# =========================================================================================
class ContextAttention(nn.Module):
    def __init__(self, size: int, num_head: int = 1):
        super().__init__()
        assert size % num_head == 0, "size must be divisible by num_head"
        self.size, self.num_head, self.head_size = size, num_head, size // num_head
        self.attention_net = nn.Linear(size, size)
        self.context_vector = nn.Parameter(torch.randn(num_head, self.head_size, 1) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, T, size) -> (B, size)
        B, T, _ = x.shape
        att = torch.tanh(self.attention_net(x)).view(B, T, self.num_head, self.head_size)
        scores = torch.einsum("bthd,hdo->bth", att, self.context_vector)     # (B, T, H)
        scores = scores.masked_fill((x.sum(-1) == 0).unsqueeze(-1), -1e10)  # padded steps get no weight (as CIPI)
        weights = torch.softmax(scores, dim=1)
        heads = x.view(B, T, self.num_head, self.head_size)
        return torch.einsum("bth,bthd->bhd", weights, heads).reshape(B, self.size)


class PackedGRU(nn.Module):
    """Dropout -> length-packed multi-layer GRU (unidirectional by default) -> padded output."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 3, dropout: float = 0.2,
                 bidirectional: bool = False, in_dropout: float = None):
        super().__init__()
        self.in_dropout = nn.Dropout(dropout if in_dropout is None else in_dropout)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.out_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.in_dropout(x.float())
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return out                                                            # (B, T, out_size)


class OnsetHead(nn.Module):
    def __init__(self, input_size: int, hidden: int = 64, num_layers: int = 3, num_head: int = 1,
                 bidirectional: bool = False, n_classes: int = N_CLASSES, dropout: float = 0.2,
                 in_dropout: float = None):
        super().__init__()
        self.branch = PackedGRU(input_size, hidden, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                                in_dropout=in_dropout)
        size = self.branch.out_size
        self.attn = ContextAttention(size, num_head=num_head)
        self.act = nn.ReLU()
        self.fc = nn.Linear(size, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.fc(self.act(self.attn(self.branch(x, lengths))))


class OrdinalLoss(nn.Module):
    """MSE over the cumulative ordinal encoding (target k -> [1]*(k+1) + [0]*(K-k-1)), mean over the
    batch, optional per-grade weights applied per sample."""

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
    """Ordinal decode: (B, K) -> label 0..K-1, or -1 when the first score is <= 0.5."""
    return (logits > 0.5).cumprod(dim=1).sum(dim=1) - 1


def balanced_class_weights(grades: list, n_classes: int = N_CLASSES) -> torch.Tensor:
    counts = torch.tensor([grades.count(g) for g in range(1, n_classes + 1)], dtype=torch.float)
    w = torch.where(counts > 0, counts.sum() / (n_classes * counts.clamp(min=1)), torch.zeros_like(counts))
    return w / w[counts > 0].mean()


def build_model(input_size: int, hidden: int = 64, num_layers: int = 3, num_head: int = 1,
                bidirectional: bool = False, n_classes: int = N_CLASSES, rnn_dropout: float = 0.2,
                in_dropout: float = 0.2) -> OnsetHead:
    return OnsetHead(input_size, hidden=hidden, num_layers=num_layers, num_head=num_head, bidirectional=bidirectional,
                     n_classes=n_classes, dropout=rnn_dropout, in_dropout=in_dropout)


def load_checkpoint(path: str, device: str = "cpu"):
    """Checkpoints before the 8-grade merge carry no 'n_classes' key: they have 9 outputs."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["input_size"], hidden=ck.get("hidden", 32), num_layers=ck.get("num_layers", 3),
                        num_head=ck.get("num_head", 1), bidirectional=ck.get("bidirectional", True),
                        n_classes=ck.get("n_classes", 9), rnn_dropout=ck.get("rnn_dropout", 0.2),
                        in_dropout=ck.get("in_dropout", 0.2))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, {k: v for k, v in ck.items() if k != "state_dict"}


# =========================================================================================
# 6. Metrics (cipi/metrics.py, without scikit-learn)
# =========================================================================================
def balanced_accuracy(y_true, y_pred) -> float:
    y, p = np.asarray(y_true), np.asarray(y_pred)
    recalls = [np.mean(p[y == c] == c) for c in np.unique(y)]
    return float(np.mean(recalls)) if recalls else float("nan")


COARSE_EDGES = (3, 6)      # acc3 buckets of 1-based grades: 1-3, 4-6, 7 and above


def coarse_bucket(grades):
    """1-based grades -> 0 (grades 1-3), 1 (4-6), 2 (7 and above)."""
    g = np.asarray(grades)
    return (g > COARSE_EDGES[0]).astype(int) + (g > COARSE_EDGES[1]).astype(int)


def acc3_balanced(y_true, y_pred) -> float:
    """acc3: balanced accuracy over the three coarse buckets (mean per-bucket recall), 1-based grades.
    A prediction counts as correct when it lands in the label's bucket (1-3 / 4-6 / 7+)."""
    return balanced_accuracy(coarse_bucket(y_true), coarse_bucket(y_pred))


def acc_pm1_balanced(y_true, y_pred) -> float:
    """Mean over the grades present of P(|pred - true| <= 1 | true = grade) -- the balanced Acc±1,
    used for every acc_pm1 in this file (validation, model selection, checkpoints, reports)."""
    y, p = np.asarray(y_true), np.asarray(y_pred)
    per_grade = [np.mean(np.abs(p[y == c] - c) <= 1) for c in np.unique(y)]
    return float(np.mean(per_grade)) if per_grade else float("nan")


def summary(y_true, y_pred) -> dict:
    """0-based labels: balanced Acc-9, balanced Acc±1 (plain rate as acc_pm1_plain), acc3 (balanced over
    the coarse buckets 1-3 / 4-6 / 7+) and acc3_weighted (the same hit rate weighted by the bucket sizes,
    i.e. the plain fraction of scores landing in their label's bucket), MSE, macro MSE."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    per_class = [np.mean((p[y == c] - c) ** 2) for c in np.unique(y)]
    return {"acc9": round(balanced_accuracy(y_true, y_pred), 4),
            "acc_pm1": round(acc_pm1_balanced(y_true, y_pred), 4),
            "acc_pm1_plain": round(float(np.mean(np.abs(y - p) <= 1)), 4),
            "acc3": round(acc3_balanced(y + 1, p + 1), 4),
            "acc3_weighted": round(float(np.mean(coarse_bucket(y + 1) == coarse_bucket(p + 1))), 4),
            "mse": round(float(np.mean((y - p) ** 2)), 4),
            "macro_mse": round(float(np.mean(per_class)), 4) if per_class else float("nan")}


def _rank(x):
    r = np.empty(len(x))
    r[x.argsort()] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def agreement(y_true, y_pred) -> dict:
    """Grades 1..N_CLASSES: exact / within-1 / within-2 rates, balanced accuracy (key 'acc9' kept
    for compatibility: it is the mean per-grade recall over the 8 grades), balanced Acc±1 (acc_pm1 =
    acc_pm1_macro; within1_pct is the plain rate), acc3 (balanced over the coarse buckets 1-3 / 4-6 /
    7+; acc3_weighted is the hit rate weighted by the bucket sizes), MAE, RMSE, bias, Spearman."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    d = p - y
    yi, pi = y.astype(int), np.rint(p).astype(int)
    rho = float(np.corrcoef(_rank(y), _rank(p))[0, 1]) if len(y) > 1 and y.std() and p.std() else float("nan")
    pm1 = round(acc_pm1_balanced(yi, pi), 4)
    return {"n": int(len(y)),
            "exact_pct": round(100 * float(np.mean(pi == yi)), 1),
            "within1_pct": round(100 * float(np.mean(np.abs(pi - yi) <= 1)), 1),
            "within2_pct": round(100 * float(np.mean(np.abs(pi - yi) <= 2)), 1),
            "acc9": round(balanced_accuracy(yi, pi), 4),
            "acc_pm1": pm1, "acc_pm1_macro": pm1,
            "acc3": round(acc3_balanced(yi, pi), 4),
            "acc3_weighted": round(float(np.mean(coarse_bucket(yi) == coarse_bucket(pi))), 4),
            "mae": round(float(np.mean(np.abs(d))), 3), "rmse": round(float(np.sqrt(np.mean(d ** 2))), 3),
            "bias": round(float(np.mean(d)), 3), "spearman": round(rho, 4)}


def confusion(y_true, y_pred, grades=None) -> list:
    """Rows = true grade, columns = predicted grade, both over 1..N_CLASSES by default."""
    grades = list(grades if grades is not None else range(1, N_CLASSES + 1))
    idx = {g: i for i, g in enumerate(grades)}
    m = [[0] * len(grades) for _ in grades]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return m


# =========================================================================================
# 7. Folds
# =========================================================================================
def stratified_buckets(items: list, n: int, seed: int) -> dict:
    """stem -> bucket 0..n-1, stratified by grade (deterministic per seed)."""
    rng = random.Random(seed)
    by_grade = {}
    for it in items:
        by_grade.setdefault(int(it["grade"]), []).append(it["stem"])
    out = {}
    for grade in sorted(by_grade):
        stems = sorted(by_grade[grade])
        rng.shuffle(stems)
        offset = rng.randrange(n)
        for i, stem in enumerate(stems):
            out[stem] = (i + offset) % n
    return out


def split_fold(items: list, fold: int, args) -> dict:
    """--n-folds 5: the re-dealt embedded buckets, test = bucket k, val = bucket k+1, train = the other three.
    --n-folds N != 5: N re-dealt stratified buckets; test = bucket k, a stratified --val-frac of the
    rest validates, the rest trains."""
    if args.n_folds == 5:
        test_b, val_b = fold % N_BUCKETS, (fold + 1) % N_BUCKETS
        out = {"train": [], "val": [], "test": []}
        for it in items:
            out["test" if it["bucket"] == test_b else "val" if it["bucket"] == val_b else "train"].append(it)
        return out
    buckets = stratified_buckets(items, args.n_folds, args.seed)
    test = [it for it in items if buckets[it["stem"]] == fold]
    rest = [it for it in items if buckets[it["stem"]] != fold]
    val_b = stratified_buckets(rest, max(2, int(round(1 / args.val_frac))), args.seed + 1000 + fold)
    return {"train": [it for it in rest if val_b[it["stem"]] != 0],
            "val": [it for it in rest if val_b[it["stem"]] == 0], "test": test}


# =========================================================================================
# 8. Training, out-of-fold evaluation (train.py)
# =========================================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> str:
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


@torch.no_grad()
def evaluate(model, loader, device, criterion=None) -> dict:
    model.eval()
    y_true, y_pred, total, n = [], [], 0.0, 0
    for x, lengths, y in loader:
        logits = model(x.to(device), lengths)
        if criterion is not None:
            total += float(criterion(logits, y.to(device)))
            n += 1
        y_pred += prediction2label(logits.cpu()).clamp(min=0).tolist()
        y_true += y.tolist()
    m = summary(y_true, y_pred)
    if criterion is not None:
        m["loss"] = round(total / max(n, 1), 4)
    return m


@torch.no_grad()
def predict_dataset(model, ds: OnsetDataset, device, batch_size) -> dict:
    """stem -> decoded grade 1..N_CLASSES (None for the ordinal 'no prediction' sentinel)."""
    model.eval()
    lengths = [min(ds.feats[s].shape[0], ds.max_len or 10 ** 9) for s, _ in ds.items]
    out = {}
    for idxs in LengthBatchSampler(lengths, batch_size, shuffle=False):
        x, ln, _ = collate([ds[j] for j in idxs])
        for j, lab in zip(idxs, prediction2label(model(x.to(device), ln).cpu()).tolist()):
            out[ds.items[j][0]] = None if lab < 0 else lab + 1
    return out


def train_one(fold: int, items: list, feats: dict, args, out_dir: str) -> dict:
    set_seed(args.seed)
    device = pick_device(args.device)
    split = split_fold(items, fold, args)
    train_items = split["train"][: args.limit_train] if args.limit_train else split["train"]
    mk = lambda rows, train: make_loader(rows, feats, args.max_len, args.batch_size, train,  # noqa: E731
                                         transpose=args.transpose if train else 0)
    train_loader, val_loader, test_loader = mk(train_items, True), mk(split["val"], False), mk(split["test"], False)
    if not len(train_loader.dataset) or not len(val_loader.dataset):
        sys.exit(f"[fold {fold}] empty train/val set")

    input_size = FEATURE_DIMS[args.features]
    model = build_model(input_size, hidden=args.hidden, num_layers=args.layers, num_head=args.num_head,
                        bidirectional=args.bidirectional, n_classes=N_CLASSES, rnn_dropout=args.rnn_dropout,
                        in_dropout=args.in_dropout).to(device)
    class_w = balanced_class_weights([g for _, g in train_loader.dataset.items]).to(device) if args.balanced else None
    criterion = OrdinalLoss(class_weights=class_w)
    optimizer = (torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                 if args.weight_decay > 0 else torch.optim.Adam(model.parameters(), lr=args.lr))
    quiet = getattr(args, "quiet", False)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"onset_ordinal-{fold}.pt")
    best = {"macro_mse": float("inf"), "acc_pm1": float("-inf"), "state": None, "epoch": -1, "val": None}

    def meta(test=None, partial=False):
        return {"fold": fold, "features": args.features, "input_size": input_size, "hidden": args.hidden,
                "num_layers": args.layers, "num_head": args.num_head, "bidirectional": args.bidirectional,
                "in_dropout": args.in_dropout, "rnn_dropout": args.rnn_dropout, "weight_decay": args.weight_decay,
                "balanced": args.balanced, "lr": args.lr, "batch_size": args.batch_size,
                "n_classes": N_CLASSES, "merge_grades": dict(MERGE_GRADES), "split_seed": SPLIT_SEED,
                "seed": args.seed, "n_folds": args.n_folds, "val_frac": args.val_frac,
                "exclude_non_piano": args.exclude_non_piano,
                "max_len": args.max_len,
                "transpose": args.transpose, "val_metrics": best["val"], "test_metrics": test or {},
                "epoch": best["epoch"], "partial": partial, "dataset": "score_dataset_difficulty_musicxml",
                "n_train": len(train_loader.dataset), "source": "onset_colab_train.py",
                "encoder_version": ENCODER_VERSION, "rate_from": args.rate_from}

    stale = 0
    print(f"\n=== fold {fold} | features={args.features} ({input_size} dims, rate from {args.rate_from}) "
          f"{'bi' if args.bidirectional else 'uni'}-GRU hidden={args.hidden} layers={args.layers} "
          f"heads={args.num_head} in_do={args.in_dropout} rnn_do={args.rnn_dropout} wd={args.weight_decay:g} "
          f"balanced={args.balanced} batch={args.batch_size} transpose=±{args.transpose} device={device} lr={args.lr} "
          f"max_len={args.max_len} "
          f"| train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)} ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, t0 = 0.0, time.time()
        for x, lengths, y in train_loader:
            loss = criterion(model(x.to(device), lengths), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total += float(loss.detach())
        val = evaluate(model, val_loader, device, criterion)
        by_mse = val["macro_mse"] < best["macro_mse"]
        by_pm1 = val["acc_pm1"] > best["acc_pm1"]
        if not quiet or epoch % 10 == 0 or by_mse or by_pm1:
            print(f"[fold {fold}] epoch {epoch:>3} | train_loss {total / max(len(train_loader), 1):.4f} | val {val} "
                  f"| {time.time() - t0:.0f}s{' *' if (by_mse or by_pm1) else ''}")
        if by_mse or by_pm1:
            best = {"macro_mse": val["macro_mse"] if by_mse else best["macro_mse"],
                    "acc_pm1": val["acc_pm1"] if by_pm1 else best["acc_pm1"],
                    "epoch": epoch, "val": val,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
            torch.save({"state_dict": best["state"], **meta(partial=True)}, ckpt_path)   # survives an interrupt
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[fold {fold}] early stop")
                break

    model.load_state_dict(best["state"])
    test = evaluate(model, test_loader, device, criterion) if len(test_loader.dataset) else {}
    preds = predict_dataset(model, test_loader.dataset, device, args.batch_size)
    print(f"[fold {fold}] BEST epoch {best['epoch']} | val {best['val']} | TEST {test}")
    torch.save({"state_dict": best["state"], **meta(test=test, partial=False)}, ckpt_path)
    print(f"[fold {fold}] saved -> {ckpt_path}")
    return {"fold": fold, "epoch": best["epoch"], "val": best["val"], "test": test, "preds": preds}


OUTSIDE_TOLERANCE = 2      # a score is "outside ±2" when |predicted - label| > 2


def outside_tolerance(rows: list, label_key: str = "label", pred_key: str = "pred", tol: int = OUTSIDE_TOLERANCE) -> list:
    """The rows whose prediction misses the label by more than `tol` grades (the complement of the
    within-2 rate), worst first, each with its signed 'diff' = predicted - label."""
    out = [dict(r, diff=r[pred_key] - r[label_key]) for r in rows
           if r.get(pred_key) is not None and r.get(label_key) is not None and abs(r[pred_key] - r[label_key]) > tol]
    return sorted(out, key=lambda r: (-abs(r["diff"]), r["diff"], str(r.get("stem") or r.get("file"))))


def print_outside(rows_out: list, n_total: int, what: str = "scores") -> None:
    """One line per score outside ±OUTSIDE_TOLERANCE: name, label -> predicted, diff, bucket / file."""
    print(f"\n  {len(rows_out)} of {n_total} {what} outside ±{OUTSIDE_TOLERANCE} grades (label -> predicted):")
    for r in rows_out:
        where = f"bucket {r['bucket']}" if "bucket" in r else str(r.get("file", ""))
        flag = "  (sentinel: no prediction, counted as grade 1)" if r.get("sentinel") else ""
        print(f"    {str(r.get('stem') or r.get('file')):<24} {r['label']} -> {r['pred']}   diff {r['diff']:+d}   {where}{flag}")


def oof_report(items: list, preds: dict, out_dir: str, tag: str) -> dict:
    """Out-of-fold predicted vs labelled grade; a fold's 'no prediction' sentinel counts as grade 1.
    Besides the metrics it lists the scores outside ±2 grades (printed, oof_<tag>_outside2.csv,
    'outside2' in the JSON) so they can be looked at one by one."""
    rows = [{"stem": it["stem"], "label": it["grade"], "pred": 1 if preds[it["stem"]] is None else preds[it["stem"]],
             "sentinel": preds[it["stem"]] is None, "bucket": it["bucket"]} for it in items if it["stem"] in preds]
    if not rows:
        return {}
    for r in rows:
        r["diff"] = r["pred"] - r["label"]
    y, p = [r["label"] for r in rows], [r["pred"] for r in rows]
    rep = agreement(y, p)
    rep["confusion"] = confusion(y, p)
    rep["n_sentinel"] = sum(r["sentinel"] for r in rows)
    per_grade = {}
    for g in range(1, N_CLASSES + 1):
        sub = [r["pred"] for r in rows if r["label"] == g]
        if sub:
            d = np.array(sub, float) - g
            per_grade[g] = {"n": len(sub), "mean_pred": round(float(np.mean(sub)), 2),
                            "mae": round(float(np.abs(d).mean()), 2), "within1_pct": round(100 * float(np.mean(np.abs(d) <= 1)), 1)}
    rep["per_grade"] = per_grade
    outside = outside_tolerance(rows)
    rep["n_outside2"] = len(outside)
    rep["outside2"] = [{k: r[k] for k in ("stem", "label", "pred", "diff", "bucket", "sentinel")} for r in outside]
    os.makedirs(out_dir, exist_ok=True)
    cols = ["stem", "label", "pred", "diff", "sentinel", "bucket"]
    with open(os.path.join(out_dir, f"oof_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, f"oof_{tag}_outside2.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(outside)
    json.dump(rep, open(os.path.join(out_dir, f"oof_{tag}.json"), "w"), indent=1)
    print(f"\n=== out-of-fold: {len(rows)} scores ({rep['n_sentinel']} sentinel -> grade 1)")
    print(f"  acc9={rep['acc9']:.3f} acc±1={rep['acc_pm1_macro']:.3f} acc3={rep['acc3']:.3f} "
          f"acc3_weighted={rep['acc3_weighted']:.3f} exact={rep['exact_pct']:.1f}% "
          f"within1={rep['within1_pct']:.1f}% within2={rep['within2_pct']:.1f}% MAE={rep['mae']:.2f} "
          f"RMSE={rep['rmse']:.2f} bias={rep['bias']:+.2f} spearman={rep['spearman']:.3f}")
    print("  per grade (label: n, mean predicted, MAE, within ±1):")
    for g, m in per_grade.items():
        print(f"    {g}: n={m['n']:<4} mean_pred={m['mean_pred']:.2f} MAE={m['mae']:.2f} within1={m['within1_pct']:.0f}%")
    print(f"  confusion (rows: label 1..{N_CLASSES}, cols: predicted 1..{N_CLASSES}):")
    for g, row in zip(range(1, N_CLASSES + 1), rep["confusion"]):
        print(f"    {g}: " + " ".join(f"{c:4d}" for c in row))
    print_outside(outside, len(rows))
    print(f"  -> {os.path.join(out_dir, f'oof_{tag}.csv')} / .json / _outside2.csv")
    return rep


# =========================================================================================
# 9. Prediction with the fold ensemble (predict.py)
# =========================================================================================
def find_scores(folder: str) -> list:
    """Every .musicxml / .xml / .mxl file below `folder`, recursively, sorted."""
    out = []
    for root, _dirs, files in os.walk(folder):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(SCORE_EXTS) and not f.startswith(".")]
    return sorted(out)


@torch.no_grad()
def grade_files(paths: list, ckpt_dir: str, bpm_by_stem: dict, args, out_csv: str = None,
                labels: dict = None, base_dir: str = None) -> list:
    """Grade MusicXML files with the fold ensemble. Start tempo per file: --start-bpm, else
    --file-tempo, else the workbook row of the file's stem, else the file's own first mark (noted).
    With out_csv the rows are written there (+ .json) and a table is printed instead of JSON."""
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "onset_ordinal-*.pt")))
    if not ckpts:
        sys.exit(f"no onset_ordinal-<fold>.pt checkpoints in {ckpt_dir}")
    device = pick_device(args.device)
    models = [load_checkpoint(p, device) for p in ckpts]
    for _, m in models:
        m["feature_set"] = resolve_feature_set(m)          # encoder-1 checkpoints -> the -norate sets
    enc = ensemble_encoder([m for _, m in models])
    partial = [str(m["fold"]) for _, m in models if m.get("partial")]
    arch = sorted({f"{'bi' if m.get('bidirectional', True) else 'uni'}-GRU {m.get('hidden', 32)}x{m.get('num_layers', 3)}"
                   for _, m in models})
    print(f"\n{len(models)} fold model(s) from {ckpt_dir}: features {sorted({m['features'] for _, m in models})}; "
          f"{', '.join(arch)}; encoder v{enc['version']}, rate from {enc['rate_from']}"
          + (f"; PARTIAL fold(s) {','.join(partial)} (best epoch of an interrupted run)" if partial else "")
          + f"; {len(paths)} file(s) to grade")
    labels = labels or {}
    results, t0 = [], time.time()
    for i, path in enumerate(paths, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        rec = {"file": os.path.relpath(path, base_dir) if base_dir else path, "stem": stem,
               "label": labels.get(stem), "error": ""}
        if args.start_bpm is not None:
            bpm, src = args.start_bpm, "argument"
        elif args.file_tempo:
            bpm, src = None, "file"
        elif stem in bpm_by_stem and bpm_by_stem[stem] is not None:
            bpm, src = bpm_by_stem[stem], "workbook"
        else:
            bpm, src = None, "file (no workbook tempo)"
        try:
            res = encode(path, None, start_bpm=bpm, file_tempo=bpm is None, rate_from=enc["rate_from"])
        except Exception as exc:  # noqa: BLE001
            res, rec["error"] = None, f"{type(exc).__name__}: {exc}"
        if res is None:
            rec["error"] = rec["error"] or "skipped: fewer than two staves with notes (see the warning)"
            results.append(rec)
            continue
        rows, _, marks, info = res
        rec.update({"n_events": len(rows), "start_bpm": info["start_bpm"], "start_bpm_source": src,
                    "tempo_marks": len(marks), "warnings": "; ".join(info["warnings"]), "folds": {}})
        grades = []
        for model, meta in models:
            x = matrix_from_rows(rows, meta["feature_set"])
            if meta.get("max_len") and x.shape[0] > meta["max_len"]:
                x = x[-meta["max_len"]:]
            lab = int(prediction2label(model(x.unsqueeze(0).to(device), torch.tensor([x.shape[0]])).cpu())[0])
            g = None if lab < 0 else lab + 1
            rec["folds"][str(meta["fold"])] = g
            if g is not None:
                grades.append(g)
        rec["average"] = round(sum(grades) / len(grades), 3) if grades else None
        rec["grade"] = int(round(rec["average"])) if grades else None
        rec["diff"] = rec["grade"] - rec["label"] if rec["grade"] is not None and rec["label"] is not None else None
        results.append(rec)
        if out_csv and (i % 25 == 0 or i == len(paths)):
            print(f"  graded [{i}/{len(paths)}] {time.time() - t0:.0f}s")
    if not out_csv:
        print(json.dumps(results[0] if len(results) == 1 else results, indent=2, ensure_ascii=False))
        return results

    fold_ids = sorted({f for r in results for f in r.get("folds", {})}, key=int)
    cols = ["file", "stem", "grade", "average", "label", "diff"] + [f"fold_{f}" for f in fold_ids] + \
           ["n_events", "start_bpm", "start_bpm_source", "tempo_marks", "warnings", "error"]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({**r, **{f"fold_{f}": g for f, g in r.get("folds", {}).items()}})
    json.dump(results, open(os.path.splitext(out_csv)[0] + ".json", "w"), indent=1, ensure_ascii=False)
    print(f"\n{'file':50} {'grade':>5} {'avg':>6} {'label':>5}  tempo")
    for r in results:
        print(f"{r['file'][:50]:50} {str(r.get('grade') or '-'):>5} "
              f"{('%.2f' % r['average']) if r.get('average') is not None else '-':>6} {str(r['label'] or '-'):>5}  "
              f"{r.get('start_bpm', '-')} ({r.get('start_bpm_source', '-')})" + (f"  ERROR {r['error']}" if r["error"] else ""))
    graded = [r for r in results if r.get("grade") is not None]
    hist = {g: sum(r["grade"] == g for r in graded) for g in range(1, N_CLASSES + 1)}
    print(f"\ngraded {len(graded)} of {len(results)} files ({sum(bool(r['error']) for r in results)} failed); "
          f"grade histogram: " + " ".join(f"{g}:{n}" for g, n in hist.items() if n))
    labelled = [r for r in graded if r["label"] is not None]
    if labelled:
        m = agreement([r["label"] for r in labelled], [r["grade"] for r in labelled])
        print(f"vs workbook labels ({len(labelled)} files): acc9={m['acc9']:.3f} acc±1={m['acc_pm1_macro']:.3f} acc3={m['acc3']:.3f} "
              f"exact={m['exact_pct']:.1f}% within1={m['within1_pct']:.1f}% within2={m['within2_pct']:.1f}% "
              f"MAE={m['mae']:.2f} bias={m['bias']:+.2f}")
        print_outside(outside_tolerance(labelled, "label", "grade"), len(labelled), "files")
    print(f"-> {out_csv} / .json")
    return results


# =========================================================================================
# 9b. --grid: one-fold hyper-parameter grid, resumable, ranked summary
# =========================================================================================
import copy as _copy
import itertools as _it

GRID_AXES_DEFAULT = ("features=core,core+timing;hidden=32,64,128;layers=1,2,3;bidirectional=0,1;"
                     "in_dropout=0,0.2,0.4;rnn_dropout=0,0.2;lr=0.0003,0.001,0.003;batch_size=32,64;"
                     "transpose=0,3;balanced=1,0;weight_decay=0,0.0001")
_INT = {"hidden", "layers", "batch_size", "transpose"}
_BOOL = {"bidirectional", "balanced"}
_FLOAT = {"in_dropout", "rnn_dropout", "lr", "weight_decay"}


def parse_axes(spec: str) -> dict:
    axes = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        name, _, vals = part.partition("=")
        name = name.strip().replace("-", "_")
        conv = int if name in _INT else (lambda v: bool(int(v))) if name in _BOOL else float if name in _FLOAT else str
        axes[name] = [conv(v.strip()) for v in vals.split(",") if v.strip()]
    return axes


def _apply(args, setting: dict):
    a = _copy.copy(args)
    for k, v in setting.items():
        setattr(a, k, v)
    a.quiet = True
    return a


def setting_name(s: dict) -> str:
    return (f"f-{s['features'].replace('+', '_')}_h{s['hidden']}_L{s['layers']}_{'bi' if s['bidirectional'] else 'uni'}"
            f"_in{s['in_dropout']:g}_rnn{s['rnn_dropout']:g}_lr{s['lr']:g}_b{s['batch_size']}_t{s['transpose']}"
            f"_{'bal' if s['balanced'] else 'nobal'}_wd{s['weight_decay']:g}")


def grid_settings(args, axes: dict) -> list:
    base = {k: getattr(args, k) for k in axes}
    settings = [dict(base)]
    mode = args.grid
    if mode in ("oat", "both"):
        for k, vals in axes.items():
            for v in vals:
                if v != base[k]:
                    s = dict(base)
                    s[k] = v
                    settings.append(s)
    if mode in ("random", "both"):
        rng = random.Random(args.grid_random_seed)
        keys = list(axes)
        tries = 0
        while sum(1 for s in settings if s.get("_random")) < args.grid_n and tries < 100 * args.grid_n:
            tries += 1
            s = {k: rng.choice(axes[k]) for k in keys}
            if any({kk: vv for kk, vv in t.items() if kk != "_random"} == s for t in settings):
                continue
            s["_random"] = True
            settings.append(s)
    if mode == "full":
        keys = list(axes)
        for combo in _it.product(*(axes[k] for k in keys)):
            s = dict(zip(keys, combo))
            if s not in settings:
                settings.append(s)
    return settings


def train_or_resume(fold: int, items: list, feats: dict, a, out_dir: str) -> dict:
    ckpt = os.path.join(out_dir, f"onset_ordinal-{fold}.pt")
    res_json = os.path.join(out_dir, f"result-{fold}.json")
    if os.path.exists(res_json) and os.path.exists(ckpt):
        try:
            meta = torch.load(ckpt, map_location="cpu", weights_only=False)
            if not meta.get("partial"):
                r = json.load(open(res_json))
                r["preds"] = {k: (None if v is None else int(v)) for k, v in r["preds"].items()}
                print(f"[fold {fold}] resume: cached result {res_json}")
                return r
        except Exception:  # noqa: BLE001
            pass
    r = train_one(fold, items, feats, a, out_dir)
    json.dump(r, open(res_json, "w"), default=float)
    return r


def merged_metrics(labels: dict, preds: dict) -> dict:
    """acc8m: balanced accuracy of the same predictions with grades 8+9 merged; per-grade recall."""
    stems = list(preds)
    y = [labels[s] for s in stems]
    p = [preds[s] or 1 for s in stems]
    rec = {}
    for g in sorted(set(y)):
        idx = [i for i, v in enumerate(y) if v == g]
        rec[g] = float(np.mean([p[i] == g for i in idx]))
    return {"acc8m": balanced_accuracy([min(v, 8) for v in y], [min(v, 8) for v in p]),
            "recall8": rec.get(8, float("nan")), "recall9": rec.get(9, float("nan")), "n9": y.count(9)}


def run_grid(items: list, feats_by_set: dict, args, grid_dir: str) -> list:
    os.makedirs(grid_dir, exist_ok=True)
    labels = {it["stem"]: int(it["grade"]) for it in items}
    axes = parse_axes(args.grid_axes)
    settings = grid_settings(args, axes)
    seeds = [int(s) for s in args.grid_seeds.split(",") if s.strip()]
    folds = [int(f) for f in args.grid_folds.split(",") if f.strip()]
    n_train = len(settings) * len(seeds) * len(folds)
    print(f"\n######## GRID ({args.grid}): {len(settings)} settings x {len(seeds)} seeds x {len(folds)} folds = {n_train} "
          f"trainings (resumable) | baseline {setting_name(settings[0])}")
    t_start = time.time()
    rows = []
    for i, s in enumerate(settings):
        clean = {k: v for k, v in s.items() if k != "_random"}
        name = setting_name(clean)
        a = _apply(args, clean)
        feats = feats_by_set[clean["features"]]
        res = {}
        for seed in seeds:
            a.seed = seed
            for f in folds:
                print(f"\n--- setting {i + 1}/{len(settings)}: {name} | seed {seed} fold {f} | "
                      f"{(time.time() - t_start) / 60:.0f} min elapsed")
                res[(seed, f)] = train_or_resume(f, items, feats, a, os.path.join(grid_dir, name, f"seed{seed}"))
        tests = [r["test"] for r in res.values()]
        mm = [merged_metrics(labels, r["preds"]) for r in res.values()]
        row = {"setting": name, **clean, "random": bool(s.get("_random")),
               "acc9": float(np.mean([t["acc9"] for t in tests])), "acc9_std": float(np.std([t["acc9"] for t in tests])),
               "acc8m": float(np.mean([m["acc8m"] for m in mm])),
               "acc_pm1": float(np.mean([t["acc_pm1"] for t in tests])), "acc3": float(np.mean([t["acc3"] for t in tests])),
               "macro_mse": float(np.mean([t["macro_mse"] for t in tests])),
               "recall8": float(np.nanmean([m["recall8"] for m in mm])), "recall9": float(np.nanmean([m["recall9"] for m in mm])),
               "n9": mm[0]["n9"], "best_epoch": float(np.mean([r["epoch"] for r in res.values()])),
               "per_run_acc9": [t["acc9"] for t in tests]}
        rows.append(row)
        json.dump(rows, open(os.path.join(grid_dir, "rows_partial.json"), "w"), indent=1, default=float)
    base = rows[0]
    ranked = sorted(rows, key=lambda r: (r["acc9"], r["acc_pm1"]), reverse=True)
    md = [f"# One-fold grid ({args.grid}; {(time.time() - t_start) / 60:.1f} min this run, {n_train} trainings)", "",
          f"folds {folds}, seeds {seeds}, {N_CLASSES} grades, dataset {DATASET_TAG}; baseline = {base['setting']}", "",
          f"baseline: acc9 {base['acc9']:.3f}, acc8m {base['acc8m']:.3f}, acc±1 {base['acc_pm1']:.3f}, "
          f"recall grade 8 {base['recall8']:.2f}, grade 9 {base['recall9']:.2f} ({base['n9']} grade-9 scores in the test fold)", "",
          "acc9 = balanced accuracy over 9 grades; acc8m = the same predictions with 8+9 merged (comparable with the "
          "earlier 8-grade runs); Δ = difference to the baseline on the same fold(s)/seed(s). "
          f"One fold: differences below ~0.05 are within the fold noise.", "",
          "| # | setting | acc9 | Δ | acc8m | acc±1 | acc3 | mMSE | rec8 | rec9 | epoch |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        tag = " **(baseline)**" if r is base else ""
        md.append(f"| {i} | {r['setting']}{tag} | {r['acc9']:.3f} | {r['acc9'] - base['acc9']:+.3f} | {r['acc8m']:.3f} | "
                  f"{r['acc_pm1']:.3f} | {r['acc3']:.3f} | {r['macro_mse']:.2f} | {r['recall8']:.2f} | {r['recall9']:.2f} | "
                  f"{r['best_epoch']:.0f} |")
    # per-axis view: mean acc9 of all settings sharing a value (random + oat together)
    md += ["", "## Per-axis means (acc9 over every setting with that value)", "", "| axis | value | n | mean acc9 | mean acc8m | mean acc±1 |", "|---|---|---|---|---|---|"]
    for k, vals in axes.items():
        for v in vals:
            sub = [r for r in rows if r[k] == v]
            if sub:
                md.append(f"| {k} | {v} | {len(sub)} | {np.mean([r['acc9'] for r in sub]):.3f} | "
                          f"{np.mean([r['acc8m'] for r in sub]):.3f} | {np.mean([r['acc_pm1'] for r in sub]):.3f} |")
    md += ["", "## Shortlist", ""]
    for r in ranked[:5]:
        md.append(f"- {r['setting']}: acc9 {r['acc9']:.3f} (Δ {r['acc9'] - base['acc9']:+.3f}), acc8m {r['acc8m']:.3f}, "
                  f"acc±1 {r['acc_pm1']:.3f}, rec9 {r['recall9']:.2f}")
    md.append("")
    md.append("Confirm the shortlist with all folds x 3 seeds before believing any of it.")
    text = "\n".join(md) + "\n"
    open(os.path.join(grid_dir, "summary.md"), "w").write(text)
    json.dump({"folds": folds, "seeds": seeds, "axes": axes, "rows": rows}, open(os.path.join(grid_dir, "summary.json"), "w"),
              indent=1, default=float)
    print("\n" + "#" * 100 + "\n" + text + "#" * 100)
    print(f"-> {os.path.join(grid_dir, 'summary.md')} (paste this for analysis)")
    return rows


# =========================================================================================
# 10. Main
# =========================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", default="/content/drive/MyDrive/train_data_tatiana",
                    help="folder with the score files (score_<id>.musicxml / .xml / .mxl, subfolders searched)")
    ap.add_argument("--bpm-xlsx",
                    default="/content/drive/MyDrive/806_prefinal_score_dataset_difficulty_bpm.xlsx",
                    help="the workbook with musicxml_path, the grade column and bpm_db / bpm")
    ap.add_argument("--out-dir", default="/content/onset_out", help="features/, checkpoints/ go here")
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="core")
    ap.add_argument("--rate-from", choices=list(RATE_FROM), default="ioi",
                    help="local attack-rate bits from the inter-onset interval (ioi) or the written duration (dur)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="parallel encoders")
    ap.add_argument("--force-features", action="store_true", help="re-encode every score")
    ap.add_argument("--exclude-non-piano", action="store_true", help="drop the 'не для ф-но' scores")
    ap.add_argument("--limit-scores", type=int, default=0, help="only the first N scores (quick test)")
    ap.add_argument("--n-folds", type=int, default=5, help="5 = embedded buckets; other N = re-dealt stratified folds")
    ap.add_argument("--val-frac", type=float, default=0.2, help="[--n-folds != 5] validation share of the non-test scores")
    ap.add_argument("--folds", default=None, help="comma-separated fold ids to train (default: all)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64, help="pieces per batch (memory = batch-size x longest piece)")
    ap.add_argument("--max-len", type=int, default=0, help="0 = full pieces; >0 = window of N events")
    ap.add_argument("--hidden", type=int, default=64, help="GRU width (per direction with --bidirectional)")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--bidirectional", action="store_true", help="BiGRU as in CIPI (default: unidirectional)")
    ap.add_argument("--num-head", type=int, default=1, help="attention heads (GRU output width must be divisible)")
    ap.add_argument("--in-dropout", type=float, default=0.2, help="dropout on the input matrix")
    ap.add_argument("--rnn-dropout", type=float, default=0.2, help="dropout between the GRU layers")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (0 = plain Adam)")
    ap.add_argument("--quiet", action="store_true", help="print only every 10th / improving epoch")
    ap.add_argument("--grid", nargs="?", const="both", choices=["oat", "random", "both", "full"], default=None,
                    help="one-fold hyper-parameter grid (section 9b) and exit; bare --grid = both")
    ap.add_argument("--grid-folds", default="0", help="[--grid] fold(s) to train on")
    ap.add_argument("--grid-seeds", default="42", help="[--grid] seed(s)")
    ap.add_argument("--grid-n", type=int, default=30, help="[--grid random/both] number of random settings")
    ap.add_argument("--grid-random-seed", type=int, default=7, help="[--grid] seed of the random settings")
    ap.add_argument("--grid-axes", default=GRID_AXES_DEFAULT, help="[--grid] axes 'name=v1,v2;name=...'")
    ap.add_argument("--transpose", type=int, default=0, help="training augmentation: random key shift ±N semitones")
    ap.add_argument("--no-balanced", dest="balanced", action="store_false", help="no per-grade class weights")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-train", action="store_true", help="only build the features (and --predict)")
    ap.add_argument("--smoke", action="store_true", help="pipeline test: 60 scores, 3 epochs, fold 0, max_len 300")
    ap.add_argument("--predict", nargs="*", default=None, help="MusicXML files to grade with the trained folds")
    ap.add_argument("--predict-dir", default=None, help="grade every .musicxml/.xml/.mxl under this folder (recursive)")
    ap.add_argument("--predict-out", default=None,
                    help="[--predict-dir] CSV for the results (default: <out-dir>/predictions/<folder name>.csv)")
    ap.add_argument("--ckpt-dir", default=None, help="[--predict] checkpoints (default: <out-dir>/checkpoints/<variant>)")
    ap.add_argument("--start-bpm", type=float, default=None, help="[--predict] start tempo of the given files")
    ap.add_argument("--file-tempo", action="store_true", help="[--predict] use the files' own first tempo mark")
    args = ap.parse_args(argv)

    variant = (args.features.replace("+", "_") + f"_v{ENCODER_VERSION}_{N_CLASSES}g_{DATASET_TAG}"
               + ("_dur" if args.rate_from == "dur" else "") + ("_bi" if args.bidirectional else "")
               + (f"_in{args.in_dropout:g}" if args.in_dropout != 0.2 else "")
               + (f"_rnn{args.rnn_dropout:g}" if args.rnn_dropout != 0.2 else "")
               + (f"_wd{args.weight_decay:g}" if args.weight_decay > 0 else "")
               + (f"_{args.n_folds}fold" if args.n_folds != 5 else "") + f"_s{args.seed}")
    if args.smoke:
        args.limit_scores, args.epochs, args.patience, args.folds, args.max_len = 60, 3, 3, "0", 300
        variant += "_smoke"
    ckpt_dir = args.ckpt_dir or os.path.join(args.out_dir, "checkpoints", variant)
    dev = pick_device(args.device)
    print(f"torch {torch.__version__} | device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    items = read_workbook(args.bpm_xlsx)
    bpm_by_stem = {it["stem"]: it["bpm"] for it in items}
    labels = {it["stem"]: it["grade"] for it in items}
    if args.exclude_non_piano:
        items = [it for it in items if not it["not_piano"]]
    if args.limit_scores:
        items = items[: args.limit_scores]

    # files to grade: --predict FILE ... and/or every score under --predict-dir
    predict_paths = list(args.predict or [])
    if args.predict_dir:
        found = find_scores(args.predict_dir)
        if not found:
            sys.exit(f"no {' / '.join(SCORE_EXTS)} files under {args.predict_dir}")
        predict_paths += found

    def do_predict():
        out_csv = None
        if args.predict_dir:
            out_csv = args.predict_out or os.path.join(
                args.out_dir, "predictions", os.path.basename(os.path.abspath(args.predict_dir).rstrip("/")) + ".csv")
        grade_files(predict_paths, ckpt_dir, bpm_by_stem, args, out_csv=out_csv, labels=labels,
                    base_dir=args.predict_dir)

    if predict_paths and args.no_train:
        do_predict()
        return

    features_dir = os.path.join(args.out_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    index = build_features(items, args.scores_dir, features_dir, args.workers, args.force_features,
                           rate_from=args.rate_from)
    if args.no_train:
        if predict_paths:
            do_predict()
        return

    feats = load_features(features_dir, index, args.features)
    items = [it for it in items if it["stem"] in feats]
    if args.n_folds < 2:
        ap.error("--n-folds must be at least 2")
    if args.n_folds != 5:
        buckets = stratified_buckets(items, args.n_folds, args.seed)
        for it in items:
            it["bucket"] = buckets[it["stem"]]
        print(f"{args.n_folds}-fold split (seed {args.seed}): bucket sizes "
              f"{[sum(b == k for b in buckets.values()) for k in range(args.n_folds)]}, val share {args.val_frac:g}")
    print(f"{len(items)} scores with features | {describe(items)}")
    if args.grid:
        axes = parse_axes(args.grid_axes)
        feats_by_set = {args.features: feats}
        for fs in axes.get("features", []):
            if fs not in feats_by_set:
                feats_by_set[fs] = load_features(features_dir, index, fs)
        run_grid(items, feats_by_set, args, os.path.join(args.out_dir, "grid",
                                                         f"{N_CLASSES}g_{DATASET_TAG}_v{ENCODER_VERSION}_f{args.grid_folds.replace(',', '-')}"))
        return
    folds = [int(f) for f in args.folds.split(",") if f.strip()] if args.folds else list(range(args.n_folds))
    bad = [f for f in folds if not 0 <= f < args.n_folds]
    if bad:
        ap.error(f"fold ids {bad} outside 0..{args.n_folds - 1}")

    results, preds = [], {}
    for fold in folds:
        r = train_one(fold, items, feats, args, ckpt_dir)
        preds.update(r.pop("preds"))
        results.append(r)
    print("\n=== per-fold test metrics")
    for r in results:
        print(f"  fold {r['fold']}: epoch {r['epoch']:>3} val {r['val']} test {r['test']}")
    if len(results) > 1:
        for k in ("acc9", "acc_pm1", "acc3", "acc3_weighted", "mse", "macro_mse"):
            vals = [r["test"][k] for r in results if k in r["test"]]
            print(f"  mean test {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    oof_report(items, preds, ckpt_dir, variant)
    if predict_paths:
        do_predict()


if __name__ == "__main__":
    main()
