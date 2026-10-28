# Poster unified log10-ratio metrics

All metrics below use `log10(pred/exp)` and only finite, positive pairs.

## Electrical conductivity prediction

| Metric | exp006 |
|---|---:|
| RMSE(log10 σpred/σexp) | 1.251 |
| MAE(log10 σpred/σexp) | 0.724 |
| Factor-2 accuracy | 42.7% |
| Factor-10 accuracy | 78.4% |

Valid pairs: 18,968 / 19,008 rows (40 excluded).

## ZT prediction on common cohort

| Metric | exp005 ML | exp006 |
|---|---:|---:|
| RMSE(log10 ZTpred/ZTexp) | 0.859 | 0.563 |
| MAE(log10 ZTpred/ZTexp) | 0.585 | 0.435 |
| Factor-2 accuracy | 31.4% | 40.6% |
| Factor-10 accuracy | 90.0% | 92.9% |

Valid pairs per method: 379 / 383 rows (4 excluded).

The exp005 DF row in the CSV is included only as a reference and is not part of the poster's main ML-versus-exp006 comparison.
