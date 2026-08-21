# Native full-range exp005 / exp006 ZT comparison

## Scope

- exp005 DF and ML use the original Step22 primary DOI-group test rows.
- exp006 uses its original full sample-holdout reported-ZT rows.
- No common-sample or ZT<1 restriction is applied.
- Metrics are recomputed with the same definitions, but the evaluation cohorts are not paired.

## Coverage

- exp005: 6,381 rows, 1,115 samples, observed ZT max 2.29833.
- exp006: 2,279 rows, 619 samples, observed ZT max 1.9685.
- Observed ZT>=1 samples: exp005=268, exp006=10.

## Main results

- log RMSE: DF=0.811592, ML=1.28518, exp006=1.55895.
- within factor 2: DF=93.4%, ML=54.6%, exp006=37.1%.
- sample-max Spearman: DF=0.925615, ML=0.684135, exp006=0.439553.
- ZT>=1 F1: DF=0.807882, ML=0.569948, exp006=0.137931.

## Interpretation

- On these native full-range evaluations, exp006 does not outperform exp005 ML on log RMSE, factor-2 coverage, sample-max ranking, or ZT>=1 F1.
- This supports a descriptive statement that exp006 did not show a clear overall improvement over the exp005 ML result.
- It is not a causal or paired superiority test because exp005 is DOI-held-out and exp006 is sample-held-out.
- exp006 has only a small number of observed high-ZT samples, so its F1 estimate is unstable.
- exp005 DF is an in-sample fitting upper reference, not an equal unknown-material prediction baseline.
