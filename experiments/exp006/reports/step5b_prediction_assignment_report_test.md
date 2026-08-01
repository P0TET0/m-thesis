# Step5B Prediction Assignment Report

## Summary

- input_file: experiments\exp006\data\processed\step5a_validation_rows_with_splits.parquet
- input_rows: 5000
- validation rows used: 5000
- dropped rows: 0
- config_count: 32
- train reference curve bins: 752
- reliable reference curve bins: 704
- test prediction rows: 33616
- prediction_status counts: {'ok': 33600, 'missing_reference_bin': 16}
- prediction_status == ok rows: 33600
- prediction_status != ok rows: 16
- coverage_fraction summary: min=0.998857, median=0.999462, max=1
- default coverage_fraction: 0.9989235737351991
- global default coverage_fraction: 0.9989235737351991
- sample_holdout coverage median: 0.998890358296171
- paper_holdout coverage median: 1.0
- global/material_family coverage median: {'global': 0.9994617868675996, 'material_family': 0.9994617868675996}
- reference subset coverage median: {'all_valid': 0.9994617868675996, 'conservative_valid': 0.9994617868675996}
- curve method coverage median: {'row_median': 0.9994617868675996, 'sample_median': 0.9994617868675996}
- default ok rows: 928
- global default ok rows: 928
- elapsed_seconds: 6.23

## Parquet Status

- step5b_train_reference_curve_bins_test.parquet: saved
- step5b_test_predictions_test.parquet: saved
- step5b_test_predictions_valid_test.parquet: saved
- step5b_test_predictions_default_test.parquet: saved
- step5b_test_predictions_global_default_test.parquet: saved

## Prediction Unavailable Reasons

| prediction_status | row_count |
| --- | --- |
| missing_reference_bin | 16 |

## Coverage By Config

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | test_rows | prediction_ok_rows | prediction_unavailable_rows | coverage_fraction | p_test_rows | n_test_rows | p_prediction_ok_rows | n_prediction_ok_rows | reference_bins_total | reference_bins_reliable | prediction_status_counts | T_bin_count_test | material_family_count_test | sample_count_test | paper_count_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_all_valid__eval_all_valid__global__row_median | sample_holdout | all_valid | all_valid | global | row_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_all_valid__global__sample_median | sample_holdout | all_valid | all_valid | global | sample_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__row_median | sample_holdout | all_valid | all_valid | material_family | row_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | sample_holdout | all_valid | all_valid | material_family | sample_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__row_median | sample_holdout | all_valid | conservative_valid | global | row_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | sample_holdout | all_valid | conservative_valid | global | sample_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | sample_holdout | all_valid | conservative_valid | material_family | row_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | sample_holdout | all_valid | conservative_valid | material_family | sample_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__row_median | sample_holdout | conservative_valid | all_valid | global | row_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | sample_holdout | conservative_valid | all_valid | material_family | row_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | sample_holdout | conservative_valid | conservative_valid | global | row_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | sample_holdout | conservative_valid | conservative_valid | global | sample_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | sample_holdout | conservative_valid | conservative_valid | material_family | row_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | sample_holdout | conservative_valid | conservative_valid | material_family | sample_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| paper_holdout__ref_all_valid__eval_all_valid__global__row_median | paper_holdout | all_valid | all_valid | global | row_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | paper_holdout | all_valid | all_valid | global | sample_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__row_median | paper_holdout | all_valid | all_valid | material_family | row_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | paper_holdout | all_valid | all_valid | material_family | sample_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__row_median | paper_holdout | all_valid | conservative_valid | global | row_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | paper_holdout | all_valid | conservative_valid | global | sample_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | paper_holdout | all_valid | conservative_valid | material_family | row_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | all_valid | conservative_valid | material_family | sample_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__row_median | paper_holdout | conservative_valid | all_valid | global | row_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | paper_holdout | conservative_valid | all_valid | material_family | row_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | paper_holdout | conservative_valid | conservative_valid | global | row_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | paper_holdout | conservative_valid | conservative_valid | global | sample_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | paper_holdout | conservative_valid | conservative_valid | material_family | row_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | conservative_valid | conservative_valid | material_family | sample_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |

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
- test_predictions_from_test_rows_only: True

## Notes

- WARNING: none
- Step5B builds reference curves from train rows only.
- Step4 full-data reference curves are not read for independent validation.
- Test-row sigma0_S_per_m is retained for diagnostics, but not used to compute sigma_pred.
- This step creates point-level error columns; aggregate metrics such as MAE, RMSE, and factor accuracy belong to Step5C.
- Step5C should use step5b_test_predictions_valid.csv for accuracy summaries.
