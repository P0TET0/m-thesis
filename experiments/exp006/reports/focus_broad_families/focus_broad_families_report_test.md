# Focus Broad Families Report

## Inputs
- Prediction input: `experiments\exp006\data\processed\step6b_broad_family\step5b_test_predictions_valid.parquet`
- Performance summary input: `experiments\exp006\data\processed\step6c_broad_family\step6c_broad_family_group_performance_summary.csv`
- Target config: `sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median`
- selection_mode: `reliable_from_summary`
- min_rows: 30
- min_samples: 3
- min_papers: 1
- exclude_groups: `broad::SiGe_like`

## Selected Material Groups
- `broad::sulfide`
- `unknown_material_group`
- `broad::oxide`
- `broad::GeTe_like`
- `broad::other_formula_system`

## Skipped Material Groups
- None

## Per-Group Metrics
- `broad::GeTe_like`: rows=183, p=183, n=0, MAE=0.817, RMSE=2.043, factor2=86.3%, factor10=87.4%
- `broad::other_formula_system`: rows=2000, p=0, n=2000, MAE=0.823, RMSE=1.258, factor2=24.3%, factor10=72.4%
- `broad::oxide`: rows=2000, p=727, n=1273, MAE=0.896, RMSE=1.421, factor2=36.3%, factor10=70.9%
- `unknown_material_group`: rows=110, p=110, n=0, MAE=1.157, RMSE=1.605, factor2=40.9%, factor10=53.6%
- `broad::sulfide`: rows=892, p=512, n=380, MAE=1.443, RMSE=2.498, factor2=39.8%, factor10=66.5%

## Best By MAE
- `broad::GeTe_like`: MAE=0.817
- `broad::other_formula_system`: MAE=0.823
- `broad::oxide`: MAE=0.896
- `unknown_material_group`: MAE=1.157
- `broad::sulfide`: MAE=1.443

## Best By Factor2
- `broad::GeTe_like`: factor2=86.3%
- `unknown_material_group`: factor2=40.9%
- `broad::sulfide`: factor2=39.8%
- `broad::oxide`: factor2=36.3%
- `broad::other_formula_system`: factor2=24.3%

## Best By Factor10
- `broad::GeTe_like`: factor10=87.4%
- `broad::other_formula_system`: factor10=72.4%
- `broad::oxide`: factor10=70.9%
- `broad::sulfide`: factor10=66.5%
- `unknown_material_group`: factor10=53.6%

## Largest Outlier Groups
- `broad::other_formula_system`: max abs log10 error=11.212
- `broad::sulfide`: max abs log10 error=8.412
- `broad::oxide`: max abs log10 error=7.221
- `broad::GeTe_like`: max abs log10 error=6.543
- `unknown_material_group`: max abs log10 error=3.730

## Figures
- `broad_sulfide_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_all_test.png` / `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_all_test.pdf`
- `broad_sulfide_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_p_test.png` / `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_p_test.pdf`
- `broad_sulfide_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_n_test.png` / `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_n_test.pdf`
- `broad_sulfide_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_error_hist_all_test.png` / `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_error_hist_all_test.pdf`
- `broad_sulfide_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_error_hist_by_carrier_test.png` / `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_error_hist_by_carrier_test.pdf`
- `unknown_material_group_scatter_all`: `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_scatter_pred_vs_exp_all_test.png` / `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_scatter_pred_vs_exp_all_test.pdf`
- `unknown_material_group_scatter_p`: `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_scatter_pred_vs_exp_p_test.png` / `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_scatter_pred_vs_exp_p_test.pdf`
- `unknown_material_group_error_hist_all`: `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_error_hist_all_test.png` / `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_error_hist_all_test.pdf`
- `broad_oxide_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_all_test.png` / `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_all_test.pdf`
- `broad_oxide_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_p_test.png` / `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_p_test.pdf`
- `broad_oxide_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_n_test.png` / `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_n_test.pdf`
- `broad_oxide_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_error_hist_all_test.png` / `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_error_hist_all_test.pdf`
- `broad_oxide_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_error_hist_by_carrier_test.png` / `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_error_hist_by_carrier_test.pdf`
- `broad_GeTe_like_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_scatter_pred_vs_exp_all_test.png` / `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_scatter_pred_vs_exp_all_test.pdf`
- `broad_GeTe_like_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_scatter_pred_vs_exp_p_test.png` / `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_scatter_pred_vs_exp_p_test.pdf`
- `broad_GeTe_like_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_error_hist_all_test.png` / `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_error_hist_all_test.pdf`
- `broad_other_formula_system_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_scatter_pred_vs_exp_all_test.png` / `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_scatter_pred_vs_exp_all_test.pdf`
- `broad_other_formula_system_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_scatter_pred_vs_exp_n_test.png` / `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_scatter_pred_vs_exp_n_test.pdf`
- `broad_other_formula_system_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_error_hist_all_test.png` / `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_error_hist_all_test.pdf`

## How To Read The Scatter Plots
- Points closer to y=x are better.
- Points above y=x are overpredictions.
- Points below y=x are underpredictions.
- Points within the factor2 guide lines are within a factor of 2.
- Points within the factor10 guide lines are within a factor of 10.

## Notes
- This is a focus analysis that filters existing prediction results by material group.
- No new sigma_pred values are calculated.
- Step4 full-data reference curves are not used.
- Starrydata2 raw data is not read.
- broad_family classification is heuristic and is not a strict material taxonomy.

## Next Checks
- Inspect groups with small MAE and sufficient n_rows/n_samples.
- Compare p-type and n-type performance within each group.
- Treat groups with scatter close to y=x as candidates where the model works well.
- For poor scatter groups, consider whether scattering mechanisms or band assumptions differ.
