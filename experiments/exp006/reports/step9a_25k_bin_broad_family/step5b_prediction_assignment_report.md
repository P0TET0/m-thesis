# Step5B Prediction Assignment Report

## Summary

- input_file: experiments\exp006\data\processed\step9a_25k_bin_broad_family\step9a_25k_validation_rows_with_splits.parquet
- input_rows: 97086
- validation rows used: 97086
- dropped rows: 0
- config_count: 32
- train reference curve bins: 19256
- reliable reference curve bins: 16244
- test prediction rows: 604920
- prediction_status counts: {'ok': 602862, 'unreliable_reference_bin': 1512, 'missing_reference_bin': 546}
- prediction_status == ok rows: 602862
- prediction_status != ok rows: 2058
- coverage_fraction summary: min=0.991565, median=0.997572, max=0.999947
- default coverage_fraction: 0.9934238215488216
- global default coverage_fraction: 0.9993160774410774
- sample_holdout coverage median: 0.9975720038781188
- paper_holdout coverage median: 0.9961835515388104
- global/material_family coverage median: {'global': 0.9999221923981195, 'material_family': 0.992945080255069}
- reference subset coverage median: {'all_valid': 0.9975720038781188, 'conservative_valid': 0.9974874348980208}
- curve method coverage median: {'row_median': 0.9975720038781188, 'sample_median': 0.9975720038781188}
- default ok rows: 18883
- global default ok rows: 18995
- elapsed_seconds: 328.23

## Parquet Status

- step5b_train_reference_curve_bins.parquet: saved
- step5b_test_predictions.parquet: saved
- step5b_test_predictions_valid.parquet: saved
- step5b_test_predictions_default.parquet: saved
- step5b_test_predictions_global_default.parquet: saved

## Prediction Unavailable Reasons

| prediction_status | row_count |
| --- | --- |
| unreliable_reference_bin | 1512 |
| missing_reference_bin | 546 |

## Coverage By Config

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | test_rows | prediction_ok_rows | prediction_unavailable_rows | coverage_fraction | p_test_rows | n_test_rows | p_prediction_ok_rows | n_prediction_ok_rows | reference_bins_total | reference_bins_reliable | prediction_status_counts | T_bin_count_test | material_family_count_test | sample_count_test | paper_count_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_all_valid__eval_all_valid__global__row_median | sample_holdout | all_valid | all_valid | global | row_median | 19008 | 18995 | 13 | 0.9993160774410774 | 10171 | 8837 | 10161 | 8834 | 122 | 111 | {'missing_reference_bin': 6, 'ok': 18995, 'unreliable_reference_bin': 7} | 63 | 1 | 3191 | 2202 |
| sample_holdout__ref_all_valid__eval_all_valid__global__sample_median | sample_holdout | all_valid | all_valid | global | sample_median | 19008 | 18995 | 13 | 0.9993160774410774 | 10171 | 8837 | 10161 | 8834 | 122 | 111 | {'missing_reference_bin': 6, 'ok': 18995, 'unreliable_reference_bin': 7} | 63 | 1 | 3191 | 2202 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__row_median | sample_holdout | all_valid | all_valid | material_family | row_median | 19008 | 18909 | 99 | 0.9947916666666666 | 10171 | 8837 | 10116 | 8793 | 1085 | 911 | {'missing_reference_bin': 34, 'ok': 18909, 'unreliable_reference_bin': 65} | 63 | 15 | 3191 | 2202 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | sample_holdout | all_valid | all_valid | material_family | sample_median | 19008 | 18909 | 99 | 0.9947916666666666 | 10171 | 8837 | 10116 | 8793 | 1085 | 911 | {'missing_reference_bin': 34, 'ok': 18909, 'unreliable_reference_bin': 65} | 63 | 15 | 3191 | 2202 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__row_median | sample_holdout | all_valid | conservative_valid | global | row_median | 17737 | 17736 | 1 | 0.9999436206799346 | 9474 | 8263 | 9474 | 8262 | 122 | 111 | {'ok': 17736, 'unreliable_reference_bin': 1} | 52 | 1 | 3073 | 2142 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | sample_holdout | all_valid | conservative_valid | global | sample_median | 17737 | 17736 | 1 | 0.9999436206799346 | 9474 | 8263 | 9474 | 8262 | 122 | 111 | {'ok': 17736, 'unreliable_reference_bin': 1} | 52 | 1 | 3073 | 2142 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | sample_holdout | all_valid | conservative_valid | material_family | row_median | 17737 | 17663 | 74 | 0.9958279303151604 | 9474 | 8263 | 9434 | 8229 | 1085 | 911 | {'missing_reference_bin': 20, 'ok': 17663, 'unreliable_reference_bin': 54} | 52 | 15 | 3073 | 2142 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | sample_holdout | all_valid | conservative_valid | material_family | sample_median | 17737 | 17663 | 74 | 0.9958279303151604 | 9474 | 8263 | 9434 | 8229 | 1085 | 911 | {'missing_reference_bin': 20, 'ok': 17663, 'unreliable_reference_bin': 54} | 52 | 15 | 3073 | 2142 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__row_median | sample_holdout | conservative_valid | all_valid | global | row_median | 19008 | 18995 | 13 | 0.9993160774410774 | 10171 | 8837 | 10161 | 8834 | 122 | 111 | {'missing_reference_bin': 6, 'ok': 18995, 'unreliable_reference_bin': 7} | 63 | 1 | 3191 | 2202 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | 19008 | 18995 | 13 | 0.9993160774410774 | 10171 | 8837 | 10161 | 8834 | 122 | 111 | {'missing_reference_bin': 6, 'ok': 18995, 'unreliable_reference_bin': 7} | 63 | 1 | 3191 | 2202 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | sample_holdout | conservative_valid | all_valid | material_family | row_median | 19008 | 18883 | 125 | 0.9934238215488216 | 10171 | 8837 | 10091 | 8792 | 1076 | 902 | {'missing_reference_bin': 41, 'ok': 18883, 'unreliable_reference_bin': 84} | 63 | 15 | 3191 | 2202 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | 19008 | 18883 | 125 | 0.9934238215488216 | 10171 | 8837 | 10091 | 8792 | 1076 | 902 | {'missing_reference_bin': 41, 'ok': 18883, 'unreliable_reference_bin': 84} | 63 | 15 | 3191 | 2202 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | sample_holdout | conservative_valid | conservative_valid | global | row_median | 17737 | 17736 | 1 | 0.9999436206799346 | 9474 | 8263 | 9474 | 8262 | 122 | 111 | {'ok': 17736, 'unreliable_reference_bin': 1} | 52 | 1 | 3073 | 2142 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | sample_holdout | conservative_valid | conservative_valid | global | sample_median | 17737 | 17736 | 1 | 0.9999436206799346 | 9474 | 8263 | 9474 | 8262 | 122 | 111 | {'ok': 17736, 'unreliable_reference_bin': 1} | 52 | 1 | 3073 | 2142 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | sample_holdout | conservative_valid | conservative_valid | material_family | row_median | 17737 | 17660 | 77 | 0.9956587923549642 | 9474 | 8263 | 9432 | 8228 | 1076 | 902 | {'missing_reference_bin': 20, 'ok': 17660, 'unreliable_reference_bin': 57} | 52 | 15 | 3073 | 2142 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | sample_holdout | conservative_valid | conservative_valid | material_family | sample_median | 17737 | 17660 | 77 | 0.9956587923549642 | 9474 | 8263 | 9432 | 8228 | 1076 | 902 | {'missing_reference_bin': 20, 'ok': 17660, 'unreliable_reference_bin': 57} | 52 | 15 | 3073 | 2142 |
| paper_holdout__ref_all_valid__eval_all_valid__global__row_median | paper_holdout | all_valid | all_valid | global | row_median | 20154 | 20152 | 2 | 0.9999007641163045 | 11273 | 8881 | 11271 | 8881 | 127 | 112 | {'missing_reference_bin': 1, 'ok': 20152, 'unreliable_reference_bin': 1} | 55 | 1 | 3179 | 866 |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | paper_holdout | all_valid | all_valid | global | sample_median | 20154 | 20152 | 2 | 0.9999007641163045 | 11273 | 8881 | 11271 | 8881 | 127 | 112 | {'missing_reference_bin': 1, 'ok': 20152, 'unreliable_reference_bin': 1} | 55 | 1 | 3179 | 866 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__row_median | paper_holdout | all_valid | all_valid | material_family | row_median | 20154 | 20000 | 154 | 0.9923588369554431 | 11273 | 8881 | 11189 | 8811 | 1089 | 907 | {'missing_reference_bin': 38, 'ok': 20000, 'unreliable_reference_bin': 116} | 55 | 15 | 3179 | 866 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | paper_holdout | all_valid | all_valid | material_family | sample_median | 20154 | 20000 | 154 | 0.9923588369554431 | 11273 | 8881 | 11189 | 8811 | 1089 | 907 | {'missing_reference_bin': 38, 'ok': 20000, 'unreliable_reference_bin': 116} | 55 | 15 | 3179 | 866 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__row_median | paper_holdout | all_valid | conservative_valid | global | row_median | 18716 | 18715 | 1 | 0.9999465697798675 | 10516 | 8200 | 10515 | 8200 | 127 | 112 | {'ok': 18715, 'unreliable_reference_bin': 1} | 53 | 1 | 3049 | 850 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | paper_holdout | all_valid | conservative_valid | global | sample_median | 18716 | 18715 | 1 | 0.9999465697798675 | 10516 | 8200 | 10515 | 8200 | 127 | 112 | {'ok': 18715, 'unreliable_reference_bin': 1} | 53 | 1 | 3049 | 850 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | paper_holdout | all_valid | conservative_valid | material_family | row_median | 18716 | 18575 | 141 | 0.9924663389613165 | 10516 | 8200 | 10443 | 8132 | 1089 | 907 | {'missing_reference_bin': 34, 'ok': 18575, 'unreliable_reference_bin': 107} | 53 | 15 | 3049 | 850 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | all_valid | conservative_valid | material_family | sample_median | 18716 | 18575 | 141 | 0.9924663389613165 | 10516 | 8200 | 10443 | 8132 | 1089 | 907 | {'missing_reference_bin': 34, 'ok': 18575, 'unreliable_reference_bin': 107} | 53 | 15 | 3049 | 850 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__row_median | paper_holdout | conservative_valid | all_valid | global | row_median | 20154 | 20152 | 2 | 0.9999007641163045 | 11273 | 8881 | 11271 | 8881 | 122 | 111 | {'missing_reference_bin': 1, 'ok': 20152, 'unreliable_reference_bin': 1} | 55 | 1 | 3179 | 866 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | 20154 | 20152 | 2 | 0.9999007641163045 | 11273 | 8881 | 11271 | 8881 | 122 | 111 | {'missing_reference_bin': 1, 'ok': 20152, 'unreliable_reference_bin': 1} | 55 | 1 | 3179 | 866 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | paper_holdout | conservative_valid | all_valid | material_family | row_median | 20154 | 19984 | 170 | 0.9915649498858787 | 11273 | 8881 | 11178 | 8806 | 1071 | 896 | {'missing_reference_bin': 38, 'ok': 19984, 'unreliable_reference_bin': 132} | 55 | 15 | 3179 | 866 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | 20154 | 19984 | 170 | 0.9915649498858787 | 11273 | 8881 | 11178 | 8806 | 1071 | 896 | {'missing_reference_bin': 38, 'ok': 19984, 'unreliable_reference_bin': 132} | 55 | 15 | 3179 | 866 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | paper_holdout | conservative_valid | conservative_valid | global | row_median | 18716 | 18715 | 1 | 0.9999465697798675 | 10516 | 8200 | 10515 | 8200 | 122 | 111 | {'ok': 18715, 'unreliable_reference_bin': 1} | 53 | 1 | 3049 | 850 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | paper_holdout | conservative_valid | conservative_valid | global | sample_median | 18716 | 18715 | 1 | 0.9999465697798675 | 10516 | 8200 | 10515 | 8200 | 122 | 111 | {'ok': 18715, 'unreliable_reference_bin': 1} | 53 | 1 | 3049 | 850 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | paper_holdout | conservative_valid | conservative_valid | material_family | row_median | 18716 | 18561 | 155 | 0.9917183158794615 | 10516 | 8200 | 10433 | 8128 | 1071 | 896 | {'missing_reference_bin': 34, 'ok': 18561, 'unreliable_reference_bin': 121} | 53 | 15 | 3049 | 850 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | conservative_valid | conservative_valid | material_family | sample_median | 18716 | 18561 | 155 | 0.9917183158794615 | 10516 | 8200 | 10433 | 8128 | 1071 | 896 | {'missing_reference_bin': 34, 'ok': 18561, 'unreliable_reference_bin': 121} | 53 | 15 | 3049 | 850 |

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
