# Step5B Prediction Assignment Report

## Summary

- input_file: experiments\exp006\data\processed\step9a_25k_bin_broad_family\step9a_25k_validation_rows_with_splits.parquet
- input_rows: 5000
- validation rows used: 5000
- dropped rows: 0
- config_count: 32
- train reference curve bins: 9640
- reliable reference curve bins: 5940
- test prediction rows: 33616
- prediction_status counts: {'ok': 29804, 'unreliable_reference_bin': 2060, 'missing_reference_bin': 1752}
- prediction_status == ok rows: 29804
- prediction_status != ok rows: 3812
- coverage_fraction summary: min=0.623638, median=0.93471, max=1
- default coverage_fraction: 0.8546824542518837
- global default coverage_fraction: 0.9946178686759957
- sample_holdout coverage median: 0.9347101337844073
- paper_holdout coverage median: 0.8981559094719196
- global/material_family coverage median: {'global': 0.9973089343379978, 'material_family': 0.8238701951862053}
- reference subset coverage median: {'all_valid': 0.9347101337844073, 'conservative_valid': 0.924484084268799}
- curve method coverage median: {'row_median': 0.9347101337844073, 'sample_median': 0.9347101337844073}
- default ok rows: 794
- global default ok rows: 924
- elapsed_seconds: 41.49

## Parquet Status

- step5b_train_reference_curve_bins_test.parquet: saved
- step5b_test_predictions_test.parquet: saved
- step5b_test_predictions_valid_test.parquet: saved
- step5b_test_predictions_default_test.parquet: saved
- step5b_test_predictions_global_default_test.parquet: saved

## Prediction Unavailable Reasons

| prediction_status | row_count |
| --- | --- |
| unreliable_reference_bin | 2060 |
| missing_reference_bin | 1752 |

## Coverage By Config

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | test_rows | prediction_ok_rows | prediction_unavailable_rows | coverage_fraction | p_test_rows | n_test_rows | p_prediction_ok_rows | n_prediction_ok_rows | reference_bins_total | reference_bins_reliable | prediction_status_counts | T_bin_count_test | material_family_count_test | sample_count_test | paper_count_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_all_valid__eval_all_valid__global__row_median | sample_holdout | all_valid | all_valid | global | row_median | 929 | 924 | 5 | 0.9946178686759957 | 565 | 364 | 561 | 363 | 81 | 77 | {'missing_reference_bin': 1, 'ok': 924, 'unreliable_reference_bin': 4} | 41 | 1 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_all_valid__global__sample_median | sample_holdout | all_valid | all_valid | global | sample_median | 929 | 924 | 5 | 0.9946178686759957 | 565 | 364 | 561 | 363 | 81 | 77 | {'missing_reference_bin': 1, 'ok': 924, 'unreliable_reference_bin': 4} | 41 | 1 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__row_median | sample_holdout | all_valid | all_valid | material_family | row_median | 929 | 813 | 116 | 0.8751345532831001 | 565 | 364 | 523 | 290 | 530 | 307 | {'missing_reference_bin': 49, 'ok': 813, 'unreliable_reference_bin': 67} | 41 | 15 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | sample_holdout | all_valid | all_valid | material_family | sample_median | 929 | 813 | 116 | 0.8751345532831001 | 565 | 364 | 523 | 290 | 530 | 307 | {'missing_reference_bin': 49, 'ok': 813, 'unreliable_reference_bin': 67} | 41 | 15 | 158 | 118 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__row_median | sample_holdout | all_valid | conservative_valid | global | row_median | 875 | 870 | 5 | 0.9942857142857143 | 523 | 352 | 519 | 351 | 81 | 77 | {'missing_reference_bin': 1, 'ok': 870, 'unreliable_reference_bin': 4} | 41 | 1 | 152 | 114 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | sample_holdout | all_valid | conservative_valid | global | sample_median | 875 | 870 | 5 | 0.9942857142857143 | 523 | 352 | 519 | 351 | 81 | 77 | {'missing_reference_bin': 1, 'ok': 870, 'unreliable_reference_bin': 4} | 41 | 1 | 152 | 114 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | sample_holdout | all_valid | conservative_valid | material_family | row_median | 875 | 761 | 114 | 0.8697142857142857 | 523 | 352 | 482 | 279 | 530 | 307 | {'missing_reference_bin': 49, 'ok': 761, 'unreliable_reference_bin': 65} | 41 | 15 | 152 | 114 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | sample_holdout | all_valid | conservative_valid | material_family | sample_median | 875 | 761 | 114 | 0.8697142857142857 | 523 | 352 | 482 | 279 | 530 | 307 | {'missing_reference_bin': 49, 'ok': 761, 'unreliable_reference_bin': 65} | 41 | 15 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__row_median | sample_holdout | conservative_valid | all_valid | global | row_median | 929 | 924 | 5 | 0.9946178686759957 | 565 | 364 | 561 | 363 | 81 | 77 | {'missing_reference_bin': 1, 'ok': 924, 'unreliable_reference_bin': 4} | 41 | 1 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | 929 | 924 | 5 | 0.9946178686759957 | 565 | 364 | 561 | 363 | 81 | 77 | {'missing_reference_bin': 1, 'ok': 924, 'unreliable_reference_bin': 4} | 41 | 1 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | sample_holdout | conservative_valid | all_valid | material_family | row_median | 929 | 794 | 135 | 0.8546824542518837 | 565 | 364 | 506 | 288 | 520 | 297 | {'missing_reference_bin': 51, 'ok': 794, 'unreliable_reference_bin': 84} | 41 | 15 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | 929 | 794 | 135 | 0.8546824542518837 | 565 | 364 | 506 | 288 | 520 | 297 | {'missing_reference_bin': 51, 'ok': 794, 'unreliable_reference_bin': 84} | 41 | 15 | 158 | 118 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | sample_holdout | conservative_valid | conservative_valid | global | row_median | 875 | 870 | 5 | 0.9942857142857143 | 523 | 352 | 519 | 351 | 81 | 77 | {'missing_reference_bin': 1, 'ok': 870, 'unreliable_reference_bin': 4} | 41 | 1 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | sample_holdout | conservative_valid | conservative_valid | global | sample_median | 875 | 870 | 5 | 0.9942857142857143 | 523 | 352 | 519 | 351 | 81 | 77 | {'missing_reference_bin': 1, 'ok': 870, 'unreliable_reference_bin': 4} | 41 | 1 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | sample_holdout | conservative_valid | conservative_valid | material_family | row_median | 875 | 745 | 130 | 0.8514285714285714 | 523 | 352 | 468 | 277 | 520 | 297 | {'missing_reference_bin': 51, 'ok': 745, 'unreliable_reference_bin': 79} | 41 | 15 | 152 | 114 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | sample_holdout | conservative_valid | conservative_valid | material_family | sample_median | 875 | 745 | 130 | 0.8514285714285714 | 523 | 352 | 468 | 277 | 520 | 297 | {'missing_reference_bin': 51, 'ok': 745, 'unreliable_reference_bin': 79} | 41 | 15 | 152 | 114 |
| paper_holdout__ref_all_valid__eval_all_valid__global__row_median | paper_holdout | all_valid | all_valid | global | row_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 82 | 79 | {'ok': 1205} | 35 | 1 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | paper_holdout | all_valid | all_valid | global | sample_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 82 | 79 | {'ok': 1205} | 35 | 1 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__row_median | paper_holdout | all_valid | all_valid | material_family | row_median | 1205 | 959 | 246 | 0.795850622406639 | 616 | 589 | 518 | 441 | 521 | 289 | {'missing_reference_bin': 169, 'ok': 959, 'unreliable_reference_bin': 77} | 35 | 12 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | paper_holdout | all_valid | all_valid | material_family | sample_median | 1205 | 959 | 246 | 0.795850622406639 | 616 | 589 | 518 | 441 | 521 | 289 | {'missing_reference_bin': 169, 'ok': 959, 'unreliable_reference_bin': 77} | 35 | 12 | 167 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__row_median | paper_holdout | all_valid | conservative_valid | global | row_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 82 | 79 | {'ok': 1193} | 35 | 1 | 165 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | paper_holdout | all_valid | conservative_valid | global | sample_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 82 | 79 | {'ok': 1193} | 35 | 1 | 165 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | paper_holdout | all_valid | conservative_valid | material_family | row_median | 1193 | 950 | 243 | 0.7963118189438391 | 611 | 582 | 513 | 437 | 521 | 289 | {'missing_reference_bin': 167, 'ok': 950, 'unreliable_reference_bin': 76} | 35 | 12 | 165 | 43 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | all_valid | conservative_valid | material_family | sample_median | 1193 | 950 | 243 | 0.7963118189438391 | 611 | 582 | 513 | 437 | 521 | 289 | {'missing_reference_bin': 167, 'ok': 950, 'unreliable_reference_bin': 76} | 35 | 12 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__row_median | paper_holdout | conservative_valid | all_valid | global | row_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 82 | 79 | {'ok': 1205} | 35 | 1 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | 1205 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 82 | 79 | {'ok': 1205} | 35 | 1 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | paper_holdout | conservative_valid | all_valid | material_family | row_median | 1205 | 752 | 453 | 0.6240663900414938 | 616 | 589 | 511 | 241 | 513 | 280 | {'missing_reference_bin': 169, 'ok': 752, 'unreliable_reference_bin': 284} | 35 | 12 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | 1205 | 752 | 453 | 0.6240663900414938 | 616 | 589 | 511 | 241 | 513 | 280 | {'missing_reference_bin': 169, 'ok': 752, 'unreliable_reference_bin': 284} | 35 | 12 | 167 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | paper_holdout | conservative_valid | conservative_valid | global | row_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 82 | 79 | {'ok': 1193} | 35 | 1 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | paper_holdout | conservative_valid | conservative_valid | global | sample_median | 1193 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 82 | 79 | {'ok': 1193} | 35 | 1 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | paper_holdout | conservative_valid | conservative_valid | material_family | row_median | 1193 | 744 | 449 | 0.6236378876781223 | 611 | 582 | 506 | 238 | 513 | 280 | {'missing_reference_bin': 167, 'ok': 744, 'unreliable_reference_bin': 282} | 35 | 12 | 165 | 43 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | conservative_valid | conservative_valid | material_family | sample_median | 1193 | 744 | 449 | 0.6236378876781223 | 611 | 582 | 506 | 238 | 513 | 280 | {'missing_reference_bin': 167, 'ok': 744, 'unreliable_reference_bin': 282} | 35 | 12 | 165 | 43 |

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
