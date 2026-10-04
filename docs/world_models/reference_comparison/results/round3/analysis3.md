# Round 3 analysis (PREREG3.md)

Root `<scratch>/refrun/full3`; seeds [0, 1, 2, 3, 4]; 60 checkpoints per run (400 to 24000 rows); LATE = 31 checkpoints in [12000, 24000].

### LATE (primary)

| seed | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| reference | 307.6 | 346.7 | 324.5 | 314.2 | 270.3 | 312.7 |
| Ajax | 310.6 | 321.1 | 198.3 | 232.6 | 264.6 | 265.4 |

* LEARNS = False (Ajax mean 265.4 >= reference min 270.3)
* WITHIN_RANGE = False (reference range [270.3, 346.7])
* Welch t = 1.795, two-sided p = 0.1216 (positive t: reference higher)

### S1 (16000, 18000, 20000)

| seed | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| reference | 201.5 | 298.8 | 358.1 | 273.6 | 278.6 | 282.1 |
| Ajax | 373.8 | 354.0 | 250.7 | 270.1 | 280.7 | 305.8 |

* LEARNS = True (Ajax mean 305.8 >= reference min 201.5)
* WITHIN_RANGE = True (reference range [201.5, 358.1])
* Welch t = -0.678, two-sided p = 0.5166 (positive t: reference higher)

### Concordance (eval >= 400)

* at 20000: reference 2/5, Ajax 2/5
* at 24000: reference 4/5, Ajax 3/5
* rate over [16000, 24000]: reference 0.457, Ajax 0.324

## Reading: H2 or noise (no reference concordance beyond its late rate, no significant LATE difference)

## Curves (eval mean return; every 2000 rows shown, all points used above)

| rows | ref s0 | ref s1 | ref s2 | ref s3 | ref s4 | ajax s0 | ajax s1 | ajax s2 | ajax s3 | ajax s4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2000 | 20 | 23 | 29 | 17 | 16 | 22 | 19 | 16 | 32 | 28 |
| 4000 | 94 | 190 | 208 | 126 | 148 | 129 | 153 | 49 | 125 | 65 |
| 6000 | 163 | 226 | 325 | 170 | 93 | 365 | 259 | 303 | 369 | 167 |
| 8000 | 323 | 100 | 151 | 56 | 27 | 60 | 77 | 117 | 156 | 31 |
| 10000 | 152 | 156 | 40 | 37 | 174 | 132 | 148 | 82 | 365 | 348 |
| 12000 | 500 | 500 | 138 | 251 | 150 | 305 | 325 | 97 | 177 | 479 |
| 14000 | 144 | 250 | 202 | 192 | 142 | 123 | 236 | 73 | 57 | 330 |
| 16000 | 166 | 162 | 367 | 246 | 138 | 227 | 184 | 121 | 483 | 275 |
| 18000 | 173 | 372 | 208 | 203 | 220 | 500 | 378 | 432 | 131 | 67 |
| 20000 | 265 | 362 | 500 | 371 | 478 | 394 | 500 | 199 | 196 | 500 |
| 22000 | 407 | 488 | 500 | 451 | 212 | 362 | 500 | 160 | 330 | 500 |
| 24000 | 500 | 479 | 500 | 500 | 212 | 500 | 500 | 500 | 191 | 128 |
