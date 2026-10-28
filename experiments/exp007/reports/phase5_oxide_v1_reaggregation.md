# Phase 5: existing predictions reaggregated by oxide_v1

## Scope

- Baseline: `broad_family_100K` held-out predictions from Phase 1
- Taxonomy: frozen `oxide_v1` assignments from Phase 4
- Model fitting: none; this phase only changes the reporting groups
- Primary view: default-eligible samples
- Weighting: row-equal and sample-equal metrics are both reported

## Cohort

- All assigned: 3,437 rows / 610 samples
- Default eligible: 3,032 rows / 553 samples
- Review: 36 samples
- Excluded: 21 samples

## Primary pooled metrics

- Row-equal MAE: 1.030 log10
- Sample-equal MAE: 0.902 log10
- Within factor 2: 33.2%
- Within factor 10: 67.2%

## Highest sample-equal errors among supported groups

- `oxide_v1::Co_Mn_mixed_oxide`: 7 samples, sample-equal MAE 1.577, mean residual +1.576 log10
- `oxide_v1::Mn_containing_oxide`: 89 samples, sample-equal MAE 1.356, mean residual +1.268 log10
- `oxide_v1::Mo_W_V_Nb_containing_oxide`: 27 samples, sample-equal MAE 1.333, mean residual +1.080 log10
- `oxide_v1::other_inorganic_O_containing`: 17 samples, sample-equal MAE 1.203, mean residual +0.596 log10
- `oxide_v1::Cu_containing_oxide`: 23 samples, sample-equal MAE 1.175, mean residual +0.679 log10

## Interpretation

Row-equal metrics weight every temperature row equally. Sample-equal metrics first take the median signed residual within each sample and then weight samples equally. Differences between the two indicate unequal temperature-row density or within-sample error structure.

These results describe the existing broad-family predictor after regrouping. They do not estimate the performance of a future oxide_v1-specific retrained model. Small groups and review/exclusion subsets should not be compared as if they had equal statistical precision.
