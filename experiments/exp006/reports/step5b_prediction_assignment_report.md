# Step5B Prediction Assignment Report

## Summary

- input_file: experiments\exp006\data\processed\step5a_validation_rows_with_splits.parquet
- input_rows: 97086
- validation rows used: 97086
- dropped rows: 0
- config_count: 32
- train reference curve bins: 1064
- reliable reference curve bins: 992
- test prediction rows: 604920
- prediction_status counts: {'ok': 604856, 'unreliable_reference_bin': 48, 'missing_reference_bin': 16}
- prediction_status == ok rows: 604856
- prediction_status != ok rows: 64
- coverage_fraction summary: min=0.999579, median=1, max=1
- default coverage_fraction: 0.9995791245791246
- global default coverage_fraction: 0.9995791245791246
- sample_holdout coverage median: 0.9997895622895623
- paper_holdout coverage median: 1.0
- global/material_family coverage median: {'global': 1.0, 'material_family': 1.0}
- reference subset coverage median: {'all_valid': 1.0, 'conservative_valid': 1.0}
- curve method coverage median: {'row_median': 1.0, 'sample_median': 1.0}
- default ok rows: 19000
- global default ok rows: 19000
- elapsed_seconds: 77.00

## Parquet Status

- step5b_train_reference_curve_bins.parquet: saved
- step5b_test_predictions.parquet: saved
- step5b_test_predictions_valid.parquet: saved
- step5b_test_predictions_default.parquet: saved
- step5b_test_predictions_global_default.parquet: saved

## Prediction Unavailable Reasons

| prediction_status | row_count |
| --- | --- |
| unreliable_reference_bin | 48 |
| missing_reference_bin | 16 |

## Coverage By Config

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | test_rows | prediction_ok_rows | prediction_unavailable_rows | coverage_fraction | p_test_rows | n_test_rows | p_prediction_ok_rows | n_prediction_ok_rows | reference_bins_total | reference_bins_reliable | prediction_status_counts | T_bin_count_test | material_family_count_test | sample_count_test | paper_count_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_all_valid__eval_all_valid__global__row_median | sample_holdout | all_valid | all_valid | global | row_median | 19008 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 33 | 31 | {'missing_reference_bin': 2, 'ok': 19000, 'unreliable_reference_bin': 6} | 19 | 1 | 3191 | 2202 |
| sample_holdout__ref_all_valid__eval_all_valid__global__sample_median | sample_holdout | all_valid | all_valid | global | sample_median | 19008 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 33 | 31 | {'missing_reference_bin': 2, 'ok': 19000, 'unreliable_reference_bin': 6} | 19 | 1 | 3191 | 2202 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__row_median | sample_holdout | all_valid | all_valid | material_family | row_median | 19008 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 33 | 31 | {'missing_reference_bin': 2, 'ok': 19000, 'unreliable_reference_bin': 6} | 19 | 1 | 3191 | 2202 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | sample_holdout | all_valid | all_valid | material_family | sample_median | 19008 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 33 | 31 | {'missing_reference_bin': 2, 'ok': 19000, 'unreliable_reference_bin': 6} | 19 | 1 | 3191 | 2202 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__row_median | sample_holdout | all_valid | conservative_valid | global | row_median | 17737 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 33 | 31 | {'ok': 17737} | 14 | 1 | 3073 | 2142 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | sample_holdout | all_valid | conservative_valid | global | sample_median | 17737 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 33 | 31 | {'ok': 17737} | 14 | 1 | 3073 | 2142 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | sample_holdout | all_valid | conservative_valid | material_family | row_median | 17737 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 33 | 31 | {'ok': 17737} | 14 | 1 | 3073 | 2142 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | sample_holdout | all_valid | conservative_valid | material_family | sample_median | 17737 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 33 | 31 | {'ok': 17737} | 14 | 1 | 3073 | 2142 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__row_median | sample_holdout | conservative_valid | all_valid | global | row_median | 19008 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 33 | 31 | {'missing_reference_bin': 2, 'ok': 19000, 'unreliable_reference_bin': 6} | 19 | 1 | 3191 | 2202 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | 19008 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 33 | 31 | {'missing_reference_bin': 2, 'ok': 19000, 'unreliable_reference_bin': 6} | 19 | 1 | 3191 | 2202 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | sample_holdout | conservative_valid | all_valid | material_family | row_median | 19008 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 33 | 31 | {'missing_reference_bin': 2, 'ok': 19000, 'unreliable_reference_bin': 6} | 19 | 1 | 3191 | 2202 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | 19008 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 33 | 31 | {'missing_reference_bin': 2, 'ok': 19000, 'unreliable_reference_bin': 6} | 19 | 1 | 3191 | 2202 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | sample_holdout | conservative_valid | conservative_valid | global | row_median | 17737 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 33 | 31 | {'ok': 17737} | 14 | 1 | 3073 | 2142 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | sample_holdout | conservative_valid | conservative_valid | global | sample_median | 17737 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 33 | 31 | {'ok': 17737} | 14 | 1 | 3073 | 2142 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | sample_holdout | conservative_valid | conservative_valid | material_family | row_median | 17737 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 33 | 31 | {'ok': 17737} | 14 | 1 | 3073 | 2142 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | sample_holdout | conservative_valid | conservative_valid | material_family | sample_median | 17737 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 33 | 31 | {'ok': 17737} | 14 | 1 | 3073 | 2142 |
| paper_holdout__ref_all_valid__eval_all_valid__global__row_median | paper_holdout | all_valid | all_valid | global | row_median | 20154 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 34 | 31 | {'ok': 20154} | 16 | 1 | 3179 | 866 |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | paper_holdout | all_valid | all_valid | global | sample_median | 20154 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 34 | 31 | {'ok': 20154} | 16 | 1 | 3179 | 866 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__row_median | paper_holdout | all_valid | all_valid | material_family | row_median | 20154 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 34 | 31 | {'ok': 20154} | 16 | 1 | 3179 | 866 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | paper_holdout | all_valid | all_valid | material_family | sample_median | 20154 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 34 | 31 | {'ok': 20154} | 16 | 1 | 3179 | 866 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__row_median | paper_holdout | all_valid | conservative_valid | global | row_median | 18716 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 34 | 31 | {'ok': 18716} | 16 | 1 | 3049 | 850 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | paper_holdout | all_valid | conservative_valid | global | sample_median | 18716 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 34 | 31 | {'ok': 18716} | 16 | 1 | 3049 | 850 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | paper_holdout | all_valid | conservative_valid | material_family | row_median | 18716 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 34 | 31 | {'ok': 18716} | 16 | 1 | 3049 | 850 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | all_valid | conservative_valid | material_family | sample_median | 18716 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 34 | 31 | {'ok': 18716} | 16 | 1 | 3049 | 850 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__row_median | paper_holdout | conservative_valid | all_valid | global | row_median | 20154 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 33 | 31 | {'ok': 20154} | 16 | 1 | 3179 | 866 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | 20154 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 33 | 31 | {'ok': 20154} | 16 | 1 | 3179 | 866 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | paper_holdout | conservative_valid | all_valid | material_family | row_median | 20154 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 33 | 31 | {'ok': 20154} | 16 | 1 | 3179 | 866 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | 20154 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 33 | 31 | {'ok': 20154} | 16 | 1 | 3179 | 866 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | paper_holdout | conservative_valid | conservative_valid | global | row_median | 18716 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 33 | 31 | {'ok': 18716} | 16 | 1 | 3049 | 850 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | paper_holdout | conservative_valid | conservative_valid | global | sample_median | 18716 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 33 | 31 | {'ok': 18716} | 16 | 1 | 3049 | 850 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | paper_holdout | conservative_valid | conservative_valid | material_family | row_median | 18716 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 33 | 31 | {'ok': 18716} | 16 | 1 | 3049 | 850 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | conservative_valid | conservative_valid | material_family | sample_median | 18716 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 33 | 31 | {'ok': 18716} | 16 | 1 | 3049 | 850 |

## Sanity Check

- input_rows_equal_used_plus_dropped: True
- config_count_32: True
- reference_config_id_exists: True
- prediction_config_id_exists: True
- sample_holdout_no_leakage: True
- paper_holdout_no_leakage: True
- prediction_status_allowed: True
- ok_sigma0_ref_positive: True
- ok_F0_positive: True
- ok_sigma_pred_positive: True
- ok_log10_sigma_pred_finite: True
- ok_ratio_positive: True
- ok_log10_ratio_finite: True
- not_ok_prediction_values_nan: True
- sigma_pred_formula: True
- log10_ratio_formula: True
- coverage_fraction_range: True
- coverage_config_id_unique: True
- default_file_exists_nonempty: True
- global_default_file_exists_nonempty: True
- valid_file_only_ok: True
- unavailable_file_only_not_ok: True
- full_prediction_ok_nonzero: True
- full_default_ok_nonzero: True
- test_predictions_from_test_rows_only: True

## Notes

- WARNING: none
- Step5B builds reference curves from train rows only.
- Step4 full-data reference curves are not read for independent validation.
- Test-row sigma0_S_per_m is retained for diagnostics, but not used to compute sigma_pred.
- This step creates point-level error columns; aggregate metrics such as MAE, RMSE, and factor accuracy belong to Step5C.
- Step5C should use step5b_test_predictions_valid.csv for accuracy summaries.
