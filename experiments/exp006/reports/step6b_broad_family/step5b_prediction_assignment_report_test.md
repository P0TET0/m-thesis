# Step5B Prediction Assignment Report

## Summary

- input_file: experiments\exp006\data\processed\step6a_validation_rows_with_splits_key_broad_family.parquet
- input_rows: 5000
- validation rows used: 5000
- dropped rows: 0
- config_count: 32
- train reference curve bins: 3180
- reliable reference curve bins: 2428
- test prediction rows: 33616
- prediction_status counts: {'ok': 32040, 'missing_reference_bin': 1084, 'unreliable_reference_bin': 492}
- prediction_status == ok rows: 32040
- prediction_status != ok rows: 1576
- coverage_fraction summary: min=0.880134, median=0.970903, max=1
- default coverage_fraction: 0.9429494079655544
- global default coverage_fraction: 0.9989235737351991
- sample_holdout coverage median: 0.9709032754113487
- paper_holdout coverage median: 0.9406639004149377
- global/material_family coverage median: {'global': 0.9994617868675996, 'material_family': 0.9103781861292235}
- reference subset coverage median: {'all_valid': 0.9709032754113487, 'conservative_valid': 0.9709032754113487}
- curve method coverage median: {'row_median': 0.9709032754113487, 'sample_median': 0.9709032754113487}
- default ok rows: 876
- global default ok rows: 928
- elapsed_seconds: 11.72

## Parquet Status

- step5b_train_reference_curve_bins_test.parquet: saved
- step5b_test_predictions_test.parquet: saved
- step5b_test_predictions_valid_test.parquet: saved
- step5b_test_predictions_default_test.parquet: saved
- step5b_test_predictions_global_default_test.parquet: saved

## Prediction Unavailable Reasons

| prediction_status | row_count |
| --- | --- |
| missing_reference_bin | 1084 |
| unreliable_reference_bin | 492 |

## Coverage By Config

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | test_rows | prediction_ok_rows | prediction_unavailable_rows | coverage_fraction | p_test_rows | n_test_rows | p_prediction_ok_rows | n_prediction_ok_rows | reference_bins_total | reference_bins_reliable | prediction_status_counts | T_bin_count_test | material_family_count_test | sample_count_test | paper_count_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_all_valid__eval_all_valid__global__row_median | sample_holdout | all_valid | all_valid | global | row_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_all_valid__global__sample_median | sample_holdout | all_valid | all_valid | global | sample_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__row_median | sample_holdout | all_valid | all_valid | material_family | row_median | 929 | 876 | 53 | 0.9429494079655544 | 565 | 364 | 547 | 329 | 176 | 132 | {'missing_reference_bin': 16, 'ok': 876, 'unreliable_reference_bin': 37} | 12 | 15 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | sample_holdout | all_valid | all_valid | material_family | sample_median | 929 | 876 | 53 | 0.9429494079655544 | 565 | 364 | 547 | 329 | 176 | 132 | {'missing_reference_bin': 16, 'ok': 876, 'unreliable_reference_bin': 37} | 12 | 15 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__row_median | sample_holdout | all_valid | conservative_valid | global | row_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | sample_holdout | all_valid | conservative_valid | global | sample_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | sample_holdout | all_valid | conservative_valid | material_family | row_median | 875 | 822 | 53 | 0.9394285714285714 | 523 | 352 | 505 | 317 | 176 | 132 | {'missing_reference_bin': 16, 'ok': 822, 'unreliable_reference_bin': 37} | 12 | 15 | 152 | 114 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | sample_holdout | all_valid | conservative_valid | material_family | sample_median | 875 | 822 | 53 | 0.9394285714285714 | 523 | 352 | 505 | 317 | 176 | 132 | {'missing_reference_bin': 16, 'ok': 822, 'unreliable_reference_bin': 37} | 12 | 15 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__row_median | sample_holdout | conservative_valid | all_valid | global | row_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | 929 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 928} | 12 | 1 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | sample_holdout | conservative_valid | all_valid | material_family | row_median | 929 | 876 | 53 | 0.9429494079655544 | 565 | 364 | 547 | 329 | 173 | 130 | {'missing_reference_bin': 17, 'ok': 876, 'unreliable_reference_bin': 36} | 12 | 15 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | 929 | 876 | 53 | 0.9429494079655544 | 565 | 364 | 547 | 329 | 173 | 130 | {'missing_reference_bin': 17, 'ok': 876, 'unreliable_reference_bin': 36} | 12 | 15 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | sample_holdout | conservative_valid | conservative_valid | global | row_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | sample_holdout | conservative_valid | conservative_valid | global | sample_median | 875 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 23 | 22 | {'missing_reference_bin': 1, 'ok': 874} | 12 | 1 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | sample_holdout | conservative_valid | conservative_valid | material_family | row_median | 875 | 822 | 53 | 0.9394285714285714 | 523 | 352 | 505 | 317 | 173 | 130 | {'missing_reference_bin': 17, 'ok': 822, 'unreliable_reference_bin': 36} | 12 | 15 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | sample_holdout | conservative_valid | conservative_valid | material_family | sample_median | 875 | 822 | 53 | 0.9394285714285714 | 523 | 352 | 505 | 317 | 173 | 130 | {'missing_reference_bin': 17, 'ok': 822, 'unreliable_reference_bin': 36} | 12 | 15 | 152 | 114 |
| paper_holdout__ref_all_valid__eval_all_valid__global__row_median | paper_holdout | all_valid | all_valid | global | row_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | paper_holdout | all_valid | all_valid | global | sample_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__row_median | paper_holdout | all_valid | all_valid | material_family | row_median | 1205 | 1062 | 143 | 0.8813278008298755 | 616 | 589 | 583 | 479 | 177 | 130 | {'missing_reference_bin': 118, 'ok': 1062, 'unreliable_reference_bin': 25} | 11 | 12 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | paper_holdout | all_valid | all_valid | material_family | sample_median | 1205 | 1062 | 143 | 0.8813278008298755 | 616 | 589 | 583 | 479 | 177 | 130 | {'missing_reference_bin': 118, 'ok': 1062, 'unreliable_reference_bin': 25} | 11 | 12 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__row_median | paper_holdout | all_valid | conservative_valid | global | row_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | paper_holdout | all_valid | conservative_valid | global | sample_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | paper_holdout | all_valid | conservative_valid | material_family | row_median | 1193 | 1050 | 143 | 0.8801341156747695 | 611 | 582 | 578 | 472 | 177 | 130 | {'missing_reference_bin': 118, 'ok': 1050, 'unreliable_reference_bin': 25} | 11 | 12 | 165 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | all_valid | conservative_valid | material_family | sample_median | 1193 | 1050 | 143 | 0.8801341156747695 | 611 | 582 | 578 | 472 | 177 | 130 | {'missing_reference_bin': 118, 'ok': 1050, 'unreliable_reference_bin': 25} | 11 | 12 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__row_median | paper_holdout | conservative_valid | all_valid | global | row_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 24 | 22 | {'ok': 1205} | 11 | 1 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | paper_holdout | conservative_valid | all_valid | material_family | row_median | 1205 | 1062 | 143 | 0.8813278008298755 | 616 | 589 | 583 | 479 | 175 | 127 | {'missing_reference_bin': 118, 'ok': 1062, 'unreliable_reference_bin': 25} | 11 | 12 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | 1205 | 1062 | 143 | 0.8813278008298755 | 616 | 589 | 583 | 479 | 175 | 127 | {'missing_reference_bin': 118, 'ok': 1062, 'unreliable_reference_bin': 25} | 11 | 12 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | paper_holdout | conservative_valid | conservative_valid | global | row_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | paper_holdout | conservative_valid | conservative_valid | global | sample_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 24 | 22 | {'ok': 1193} | 11 | 1 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | paper_holdout | conservative_valid | conservative_valid | material_family | row_median | 1193 | 1050 | 143 | 0.8801341156747695 | 611 | 582 | 578 | 472 | 175 | 127 | {'missing_reference_bin': 118, 'ok': 1050, 'unreliable_reference_bin': 25} | 11 | 12 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | conservative_valid | conservative_valid | material_family | sample_median | 1193 | 1050 | 143 | 0.8801341156747695 | 611 | 582 | 578 | 472 | 175 | 127 | {'missing_reference_bin': 118, 'ok': 1050, 'unreliable_reference_bin': 25} | 11 | 12 | 165 | 43 |

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
