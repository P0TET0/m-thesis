# exp007 Phase 1 residual dataset

## Scope

This phase converts the frozen Phase 0 primary baseline into analysis-ready electrical-conductivity residual tables. The exp006 source is read-only.

- Baseline variant: `broad_family_100K`
- Config: `sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median`
- Source: `experiments/exp006/data/processed/step6b_broad_family/step5b_test_predictions_default.csv`
- Test rows retained: 19,008
- Evaluable rows: 18,968
- Evaluable samples: 3,189
- Evaluable papers: 2,200
- Broad-oxide evaluable rows: 3,437

## Residual definition

`residual_log10_pred_over_observed = log10(sigma_predicted / sigma_observed)`

Positive values are overpredictions. Negative values are underpredictions. Absolute residuals of log10(2), log10(5), and 1 delimit factor-2, factor-5, and factor-10 error bands.

## Residual bands

| band | rows | fraction of evaluable rows |
| --- | ---: | ---: |
| within_factor_2 | 8,093 | 42.667% |
| factor_2_to_5 | 4,930 | 25.991% |
| factor_5_to_10 | 1,844 | 9.722% |
| beyond_factor_10 | 4,101 | 21.621% |

## Prediction status

- `ok`: 18,968
- `missing_reference_bin`: 27
- `unreliable_reference_bin`: 13

## Output roles

- `phase1_sigma_residual_rows.csv`: all test rows, including unavailable predictions.
- `phase1_sigma_sample_summary.csv`: one row per held-out sample.
- `phase1_sigma_material_group_summary.csv`: broad material-group aggregates.
- `phase1_sigma_temperature_bin_summary.csv`: 100 K temperature-bin aggregates.
- `phase1_sigma_paper_summary.csv`: held-out paper aggregates.
- `phase1_sigma_outlier_rows.csv`: rows with error greater than factor 10.
- `phase1_sigma_residual_analysis.xlsx`: reader-facing workbook built from these tables.

Phase 1 prepares residual evidence only. Interpretation of error causes belongs to the next phase.

## Validation

- Stored residual maximum absolute difference: 8.79e-14
- Phase 0 metrics reproduced: True
- exp006 source unchanged: True
