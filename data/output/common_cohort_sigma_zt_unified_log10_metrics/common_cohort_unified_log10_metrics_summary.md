# Unified common-cohort metrics for the displayed parity figures

Cohort: same 383 rows / 109 samples / 83 DOIs; reported ZT within ±2 K.

## Electrical conductivity (sigma)

| Metric | exp005 DF | exp005 ML | exp006 |
|---|---:|---:|---:|
| RMSE(log10 pred/exp) | 0.119 | 0.803 | 0.568 |
| MAE(log10 pred/exp) | 0.089 | 0.537 | 0.430 |
| Factor-2 accuracy | 97.9% | 33.4% | 41.5% |
| Factor-10 accuracy | 100.0% | 94.3% | 94.0% |
| Valid rows | 383/383 | 383/383 | 383/383 |

All sigma metrics use finite, positive experimental/predicted pairs only.

## ZT

| Metric | exp005 DF | exp005 ML | exp006 |
|---|---:|---:|---:|
| RMSE(log10 pred/exp) | 0.447 | 0.859 | 0.563 |
| MAE(log10 pred/exp) | 0.182 | 0.585 | 0.435 |
| Factor-2 accuracy | 91.6% | 31.4% | 40.6% |
| Factor-10 accuracy | 95.3% | 90.0% | 92.9% |
| Valid rows | 379/383 | 379/383 | 379/383 |

All ZT metrics use finite, positive experimental/predicted pairs only.

## Important scope note

The exp006 values here reproduce the old/default exp006 predictions used in the two displayed parity figures. They are not the later broad_family predictions. DF is reference-only direct in-sample fitting.
