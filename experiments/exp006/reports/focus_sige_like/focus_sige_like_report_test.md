# Focus SiGe_like Report

## Inputs
- Prediction input: `experiments\exp006\data\processed\step6b_broad_family\step5b_test_predictions_valid.parquet`
- Target config: `sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median`
- Target material group: `broad::SiGe_like`
- Material group filter column: `material_group_key_for_prediction`

## Optional Inputs
- `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step6c_broad_family\step6c_broad_family_group_performance_summary.csv`: loaded rows=15 columns=14
- `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step6b_broad_family\step5c_metrics_by_material_family.csv`: loaded rows=960 columns=43
- `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step6b_broad_family\step5c_metrics_by_carrier_type.csv`: loaded rows=128 columns=42

## Extracted Rows
- Total rows: 197
- p-type rows: 62
- n-type rows: 135

## Metrics
- all: MAE=0.734, RMSE=0.928, factor2=29.9%, factor10=71.1%
- p: MAE=0.585, RMSE=0.769, factor2=35.5%, factor10=80.6%
- n: MAE=0.802, RMSE=0.992, factor2=27.4%, factor10=66.7%

## Largest Outlier
- row_id: `step0_00171032`
- carrier_type: `p`
- abs_log10_sigma_pred_over_exp: 2.673
- log10_sigma_pred_over_exp: 2.673
- sigma_exp_S_per_m: 197.879
- sigma_pred_S_per_m: 93127.612
- sample_key: `10.3938_jkps.65.691__sample_9210`
- paper_id: `3588`

## Figures
- `scatter_all`: `experiments\exp006\figures\focus_sige_like\focus_sige_like_scatter_pred_vs_exp_all_test.png` / `experiments\exp006\figures\focus_sige_like\focus_sige_like_scatter_pred_vs_exp_all_test.pdf`
- `scatter_p`: `experiments\exp006\figures\focus_sige_like\focus_sige_like_scatter_pred_vs_exp_p_test.png` / `experiments\exp006\figures\focus_sige_like\focus_sige_like_scatter_pred_vs_exp_p_test.pdf`
- `scatter_n`: `experiments\exp006\figures\focus_sige_like\focus_sige_like_scatter_pred_vs_exp_n_test.png` / `experiments\exp006\figures\focus_sige_like\focus_sige_like_scatter_pred_vs_exp_n_test.pdf`
- `error_hist_all`: `experiments\exp006\figures\focus_sige_like\focus_sige_like_error_hist_all_test.png` / `experiments\exp006\figures\focus_sige_like\focus_sige_like_error_hist_all_test.pdf`

## How To Read The Scatter Plots
- Points closer to y=x are better.
- Points above y=x are overpredictions.
- Points below y=x are underpredictions.

## Notes
- This is a focus analysis that filters existing prediction results by material group.
- No new sigma_pred values are calculated.
- Step4 full-data reference curves are not used.
- Starrydata2 raw data is not read.
- SiGe_like results are provisional material-group checks based on broad_family classification.
