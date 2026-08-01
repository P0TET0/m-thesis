# Step5D-1 Visual Diagnostics Report

## Inputs

- predictions_valid: experiments\exp006\data\processed\step5b_test_predictions_valid.parquet
- predictions_all: experiments\exp006\data\processed\step5b_test_predictions.csv
- reference_bins: experiments\exp006\data\processed\step5b_train_reference_curve_bins.csv
- metrics_config: experiments\exp006\data\processed\step5c_metrics_by_config.csv
- metrics_carrier: experiments\exp006\data\processed\step5c_metrics_by_carrier_type.csv
- metrics_material: experiments\exp006\data\processed\step5c_metrics_by_material_family.csv
- metrics_temperature: experiments\exp006\data\processed\step5c_metrics_by_temperature_bin.csv
- metrics_eta: experiments\exp006\data\processed\step5c_metrics_by_eta_bin.csv
- default_comparison: experiments\exp006\data\processed\step5c_default_comparison.csv
- ranking: experiments\exp006\data\processed\step5c_config_ranking.csv
- largest_errors: experiments\exp006\data\processed\step5c_largest_abs_error_rows.csv

## Figures

| figure_id | figure_path_png | figure_path_pdf | config_id | n_points_plotted |
| --- | --- | --- | --- | --- |
| step5d_scatter_pred_vs_exp_material_family_default | experiments\exp006\figures\step5d\step5d_scatter_pred_vs_exp_material_family_default.png | experiments\exp006\figures\step5d\step5d_scatter_pred_vs_exp_material_family_default.pdf | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 19000 |
| step5d_error_hist_material_family_default | experiments\exp006\figures\step5d\step5d_error_hist_material_family_default.png | experiments\exp006\figures\step5d\step5d_error_hist_material_family_default.pdf | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 19000 |
| step5d_scatter_pred_vs_exp_global_default | experiments\exp006\figures\step5d\step5d_scatter_pred_vs_exp_global_default.png | experiments\exp006\figures\step5d\step5d_scatter_pred_vs_exp_global_default.pdf | sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 19000 |
| step5d_error_hist_global_default | experiments\exp006\figures\step5d\step5d_error_hist_global_default.png | experiments\exp006\figures\step5d\step5d_error_hist_global_default.pdf | sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 19000 |
| step5d_scatter_pred_vs_exp_paper_material_family_default | experiments\exp006\figures\step5d\step5d_scatter_pred_vs_exp_paper_material_family_default.png | experiments\exp006\figures\step5d\step5d_scatter_pred_vs_exp_paper_material_family_default.pdf | paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 20000 |
| step5d_error_hist_paper_material_family_default | experiments\exp006\figures\step5d\step5d_error_hist_paper_material_family_default.png | experiments\exp006\figures\step5d\step5d_error_hist_paper_material_family_default.pdf | paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 20154 |
| step5d_scatter_pred_vs_exp_paper_global_default | experiments\exp006\figures\step5d\step5d_scatter_pred_vs_exp_paper_global_default.png | experiments\exp006\figures\step5d\step5d_scatter_pred_vs_exp_paper_global_default.pdf | paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 20000 |
| step5d_error_hist_paper_global_default | experiments\exp006\figures\step5d\step5d_error_hist_paper_global_default.png | experiments\exp006\figures\step5d\step5d_error_hist_paper_global_default.pdf | paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 20154 |
| step5d_config_mae_top16 | experiments\exp006\figures\step5d\step5d_config_mae_top16.png | experiments\exp006\figures\step5d\step5d_config_mae_top16.pdf | multiple | 16 |
| step5d_config_factor2_top16 | experiments\exp006\figures\step5d\step5d_config_factor2_top16.png | experiments\exp006\figures\step5d\step5d_config_factor2_top16.pdf | multiple | 16 |
| step5d_eta_bin_mae_default_comparison | experiments\exp006\figures\step5d\step5d_eta_bin_mae_default_comparison.png | experiments\exp006\figures\step5d\step5d_eta_bin_mae_default_comparison.pdf | default_vs_global | 12 |
| step5d_temperature_bin_mae_default_comparison | experiments\exp006\figures\step5d\step5d_temperature_bin_mae_default_comparison.png | experiments\exp006\figures\step5d\step5d_temperature_bin_mae_default_comparison.pdf | default_vs_global | 32 |
| step5d_carrier_type_mae_default_comparison | experiments\exp006\figures\step5d\step5d_carrier_type_mae_default_comparison.png | experiments\exp006\figures\step5d\step5d_carrier_type_mae_default_comparison.pdf | default_vs_global | 4 |
| step5d_material_family_mae_worst20_default | experiments\exp006\figures\step5d\step5d_material_family_mae_worst20_default.png | experiments\exp006\figures\step5d\step5d_material_family_mae_worst20_default.pdf | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 1 |
| step5d_abs_error_vs_eta_default | experiments\exp006\figures\step5d\step5d_abs_error_vs_eta_default.png | experiments\exp006\figures\step5d\step5d_abs_error_vs_eta_default.pdf | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 19000 |
| step5d_abs_error_vs_eta_default_clipped_y0_5 | experiments\exp006\figures\step5d\step5d_abs_error_vs_eta_default_clipped_y0_5.png | experiments\exp006\figures\step5d\step5d_abs_error_vs_eta_default_clipped_y0_5.pdf | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 19000 |
| step5d_abs_error_vs_temperature_default | experiments\exp006\figures\step5d\step5d_abs_error_vs_temperature_default.png | experiments\exp006\figures\step5d\step5d_abs_error_vs_temperature_default.pdf | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 19000 |
| step5d_abs_error_vs_temperature_default_clipped_y0_5 | experiments\exp006\figures\step5d\step5d_abs_error_vs_temperature_default_clipped_y0_5.png | experiments\exp006\figures\step5d\step5d_abs_error_vs_temperature_default_clipped_y0_5.pdf | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 19000 |

## Diagnostic Tables

- step5d_global_vs_material_family_prediction_diff.csv
- step5d_global_vs_material_family_prediction_diff_summary.csv
- step5d_reference_group_diagnostics.csv
- step5d_reference_group_counts.csv
- step5d_largest_error_diagnostics_top100.csv
- step5d_default_metrics_for_figures.csv
- step5d_visual_diagnostics_summary.csv

## Default Metrics

| config_label | config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction | n_rows | n_samples | n_papers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_default | sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.852081750001124 | 1.4064629875967487 | -0.0128278172165359 | 0.3263157894736842 | 0.6264736842105263 | 0.7508421052631579 | 0.9995791245791246 | 19000 | 3191 | 2202 |
| material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.852081750001124 | 1.4064629875967487 | -0.0128278172165359 | 0.3263157894736842 | 0.6264736842105263 | 0.7508421052631579 | 0.9995791245791246 | 19000 | 3191 | 2202 |
| paper_global_default | paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.8696361454419639 | 1.4699765395399786 | -0.0635495182310127 | 0.3128907412920512 | 0.6159571300982435 | 0.7470973504019053 | 1.0 | 20154 | 3179 | 866 |
| paper_material_family_default | paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.8696361454419639 | 1.4699765395399786 | -0.0635495182310127 | 0.3128907412920512 | 0.6159571300982435 | 0.7470973504019053 | 1.0 | 20154 | 3179 | 866 |
| global_default | sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.7372337311582936 | 1.2343448639232923 | 0.001209547017221 | 0.350987151363209 | 0.6950799122532122 | 0.7972422438107176 | 0.9995791245791246 | 3191 | 3191 | 2202 |
| material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.7372337311582936 | 1.2343448639232923 | 0.001209547017221 | 0.350987151363209 | 0.6950799122532122 | 0.7972422438107176 | 0.9995791245791246 | 3191 | 3191 | 2202 |
| paper_global_default | paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.733850844545466 | 1.2177864499192326 | -0.0310619312563688 | 0.3425605536332179 | 0.6860648002516515 | 0.8052846807172067 | 1.0 | 3179 | 3179 | 866 |
| paper_material_family_default | paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.733850844545466 | 1.2177864499192326 | -0.0310619312563688 | 0.3425605536332179 | 0.6860648002516515 | 0.8052846807172067 | 1.0 | 3179 | 3179 | 866 |

## How To Read Figures

- predicted vs experimental: points near y=x are accurate; vertical distance is multiplicative error.
- error distribution: zero is perfect, +/-1 corresponds to factor-10 error.
- eta/temperature/carrier plots show where MAE changes by subset.
- material_family worst20 highlights reliable material groups with largest MAE.

## Material Family vs Global

- material_family default identical to global default: True
- paper material_family default identical to paper global default: True
- inferred reason: material_group_key appears effectively single-valued and material/global reference bins have identical values

| comparison_label | left_config_id | right_config_id | joined_row_count | max_abs_delta_log10_sigma_pred | median_abs_delta_log10_sigma_pred | max_abs_delta_log10_sigma0_ref | median_abs_delta_log10_sigma0_ref | exact_equal_prediction_count | approximately_equal_prediction_count | different_prediction_count | unique_material_group_key_count | unique_material_group_key_examples | unique_material_group_key_for_prediction_count_material_family | unique_material_group_key_for_prediction_count_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout_material_family_vs_global | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 19000 | 0.0 | 0.0 | 0.0 | 0.0 | 19000 | 19000 | 0 | 1 | unknown_material_family | 1 | 1 |
| paper_holdout_material_family_vs_global | paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 20154 | 0.0 | 0.0 | 0.0 | 0.0 | 20154 | 20154 | 0 | 1 | unknown_material_family | 1 | 1 |

## Reference Group Diagnostics

| comparison_label | material_family_config_id | global_config_id | material_family_reference_bins | global_reference_bins | material_family_material_group_key_count | material_family_material_group_key_examples | joined_carrier_T_bins | max_abs_delta_log10_sigma0_ref | median_abs_delta_log10_sigma0_ref | same_reference_value_count | different_reference_value_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout_reference_material_family_vs_global | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 33 | 33 | 1 | unknown_material_family | 33 | 0.0 | 0.0 | 33 | 0 |
| paper_holdout_reference_material_family_vs_global | paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 33 | 33 | 1 | unknown_material_family | 33 | 0.0 | 0.0 | 33 | 0 |

## Largest Outliers

| config_id | row_id | abs_log10_sigma_pred_over_exp | log10_sigma_pred_over_exp | likely_error_origin_hint |
| --- | --- | --- | --- | --- |
| sample_holdout__ref_all_valid__eval_all_valid__global__sample_median | step0_00130231 | 14.924287322210992 | -14.924287322210992 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | step0_00130231 | 14.924287322210992 | -14.924287322210992 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | step0_00130231 | 14.924287322210992 | -14.924287322210992 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | step0_00130231 | 14.924287322210992 | -14.924287322210992 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__row_median | step0_00130231 | 14.915247569589235 | -14.915247569589235 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_all_valid__eval_all_valid__global__row_median | step0_00130231 | 14.915247569589235 | -14.915247569589235 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__row_median | step0_00130231 | 14.915247569589235 | -14.915247569589235 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | step0_00130231 | 14.915247569589235 | -14.915247569589235 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | step0_00130231 | 14.89226223168901 | -14.89226223168901 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | step0_00130231 | 14.89226223168901 | -14.89226223168901 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | step0_00130231 | 14.89226223168901 | -14.89226223168901 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | step0_00130231 | 14.89226223168901 | -14.89226223168901 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | step0_00130231 | 14.886311342591172 | -14.886311342591172 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | step0_00130231 | 14.886311342591172 | -14.886311342591172 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | step0_00130231 | 14.886311342591172 | -14.886311342591172 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__row_median | step0_00130231 | 14.886311342591172 | -14.886311342591172 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | step0_00130230 | 14.677015855105491 | -14.677015855105491 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | step0_00130230 | 14.677015855105491 | -14.677015855105491 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | step0_00130230 | 14.677015855105491 | -14.677015855105491 | sigma0_ref_much_smaller_than_row_sigma0 |
| sample_holdout__ref_all_valid__eval_all_valid__global__sample_median | step0_00130230 | 14.677015855105491 | -14.677015855105491 | sigma0_ref_much_smaller_than_row_sigma0 |

- Max abs error is explained by log10(sigma0_ref / row_sigma0): True

## Visual Diagnostics Summary

| diagnostic_item | status | value | comment |
| --- | --- | --- | --- |
| material_family_vs_global_default_identical_or_not | warning | True | True means both default predictions are identical within 1e-12. |
| paper_material_family_vs_global_default_identical_or_not | warning | True | True means both paper-holdout predictions are identical within 1e-12. |
| material_family_default_unique_material_group_key_count | warning | 1 | Low count suggests material grouping is effectively collapsed. |
| global_default_unique_material_group_key_for_prediction_count | ok | 1 | Global prediction key should be ALL. |
| reference_bins_material_group_key_count | warning | 1 | Count of material groups inside material_family reference configs. |
| max_abs_log10_error | warning | 14.924287322210992 | Largest absolute log10 error among top outliers. |
| max_abs_log10_error_row_id | warning | step0_00130231 | Row id of largest error. |
| number_of_extreme_ge_10_decade_errors | warning | 100 | Count in top100 outlier diagnostics. |
| number_of_severe_ge_5_decade_errors | warning | 100 | Count in top100 outlier diagnostics. |
| default_mae_log10 | ok | 0.852081750001124 | Material-family default row_equal MAE. |
| default_factor_2_accuracy | ok | 0.3263157894736842 | Material-family default row_equal factor-2 accuracy. |
| default_factor_10_accuracy | ok | 0.7508421052631579 | Material-family default row_equal factor-10 accuracy. |

## Sanity Check

- prediction_status_ok: True
- sigma_exp_positive: True
- sigma_pred_positive: True
- sigma_pred_over_exp_consistent: True
- log10_ratio_consistent: True
- sigma0_ratio_matches_prediction_error: True
- default_4_configs_exist: True
- at_least_one_default_has_rows: True
- diff_summary_created: True
- reference_diagnostics_created: True
- largest_error_diagnostics_created: True
- figure_index_created: True
- figure_files_exist_nonzero: True
- default_metrics_created: True
- visual_summary_created: True
- report_created: True
- did_not_read_step4_full_data_reference_curve: True

## Notes

- This Step5D-1 only visualizes and diagnoses existing predictions.
- Step5B prediction results are visualized; predictions are not recomputed.
- Step4 full-data reference curves are not used.
- If material_family and global results are identical, confirm material grouping before drawing research conclusions.
- Next: inspect material_group_key generation, review top outliers by paper/sample, and choose final figures or add supplemental sample_equal/paper_holdout plots.
- elapsed_seconds: 11.18
