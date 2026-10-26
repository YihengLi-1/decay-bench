# Decoder decay leaderboard

R2 by days since training (protocol A: since the last held-in day; protocol B: between training and test day), and the decay slope of R2 against log2(1 + days) with a 95% bootstrap CI. Less negative slope = slower decay.

## H1 · protocol A

| model | 4-7d | 8-14d | 15-30d | slope [95% CI] | collapsed | pairs | source |
|---|---|---|---|---|---|---|---|
| ridge_zsess | 0.04 | -0.03 | -0.00 | -0.066 [-0.204, +0.078] | 0% | 14 | local |
| ridge_raw | 0.08 | -0.10 | -0.10 | -0.184 [-0.319, -0.066] | 0% | 14 | local |

## H1 · protocol B

| model | 1-3d | 4-7d | 8-14d | 15-30d | 31-400d | slope [95% CI] | collapsed | pairs | source |
|---|---|---|---|---|---|---|---|---|---|
| NDT2 (official) | 0.20 | 0.13 | 0.10 | 0.09 | 0.08 | -0.036 [-0.052, -0.020] | 0% | 169 | FALCON authors, private test |
| GRU (official) | 0.08 | 0.03 | -0.13 | -0.12 | -0.10 | -0.083 [-0.122, -0.043] | 2% | 169 | FALCON authors, private test |
| ridge_zsess | 0.12 | 0.09 | -0.03 | -0.11 | -0.15 | -0.101 [-0.121, -0.078] | 0% | 149 | local |
| ridge_raw | -0.04 | -0.35 | -0.34 | -0.74 | -0.80 | -0.143 [-0.233, -0.039] | 15% | 149 | local |

## M1 · protocol A

| model | 4-7d | 15-30d | slope [95% CI] | collapsed | pairs | source |
|---|---|---|---|---|---|---|
| spint_e10 | 0.72 | 0.52 | -0.108 [-0.126, -0.014] | 0% | 3 | local |
| ridge_zsess | 0.47 | 0.21 | -0.144 [-0.187, +0.074] | 0% | 3 | local |
| ridge_raw | -3.13 | -1.88 | n/a (collapsed) | 100% | 3 | local |

## M1 · protocol B

| model | 1-3d | 4-7d | 8-14d | 15-30d | slope [95% CI] | collapsed | pairs | source |
|---|---|---|---|---|---|---|---|---|
| ridge_zsess | 0.41 | 0.42 | 0.47 | 0.29 | -0.038 [-0.056, -0.020] | 0% | 24 | local |
| NDT2 (official) | 0.40 | 0.23 | 0.29 | 0.13 | -0.074 [-0.131, +0.005] | 0% | 49 | FALCON authors, private test |
| GRU (official) | -0.05 | 0.01 | -0.23 | -0.55 | -0.145 [-0.254, -0.021] | 16% | 49 | FALCON authors, private test |
| ridge_raw | -0.57 | -4.81 | -11.74 | -7.37 | n/a (collapsed) | 58% | 24 | local |

## M2 · protocol B

| model | 1-3d | 4-7d | 8-14d | 15-30d | 31-400d | slope [95% CI] | collapsed | pairs | source |
|---|---|---|---|---|---|---|---|---|---|
| NDT2 (official) | 0.15 | 0.10 | 0.17 | 0.04 | -0.07 | -0.033 [-0.063, +0.002] | 0% | 64 | FALCON authors, private test |
| GRU (official) | 0.07 | 0.07 | -0.12 | -0.09 | -0.11 | -0.069 [-0.111, -0.026] | 2% | 64 | FALCON authors, private test |
