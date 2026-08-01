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
- `broad::Mg2SiSn_like`
- `broad::CoSb_skutterudite_like`
- `broad::BiSbTe_tetradymite_like`
- `broad::selenide`
- `broad::telluride`
- `broad::SbTe_like`
- `broad::BiTe_like`
- `broad::PbTe_like`
- `broad::SnTe_like`

## Skipped Material Groups
- None

## Per-Group Metrics
- `broad::SnTe_like`: rows=370, p=370, n=0, MAE=0.165, RMSE=0.371, factor2=89.5%, factor10=96.8%
- `broad::PbTe_like`: rows=472, p=269, n=203, MAE=0.268, RMSE=0.460, factor2=76.5%, factor10=96.0%
- `broad::BiTe_like`: rows=1776, p=498, n=1278, MAE=0.372, RMSE=0.566, factor2=59.6%, factor10=92.5%
- `broad::SbTe_like`: rows=974, p=671, n=303, MAE=0.391, RMSE=0.634, factor2=62.5%, factor10=90.5%
- `broad::telluride`: rows=404, p=228, n=176, MAE=0.443, RMSE=0.585, factor2=45.5%, factor10=90.6%
- `broad::selenide`: rows=1783, p=941, n=842, MAE=0.503, RMSE=0.847, factor2=53.8%, factor10=85.2%
- `broad::BiSbTe_tetradymite_like`: rows=643, p=514, n=129, MAE=0.598, RMSE=1.053, factor2=45.9%, factor10=86.9%
- `broad::CoSb_skutterudite_like`: rows=1328, p=692, n=636, MAE=0.652, RMSE=1.203, factor2=41.6%, factor10=80.9%
- `broad::Mg2SiSn_like`: rows=267, p=78, n=189, MAE=0.690, RMSE=1.062, factor2=52.8%, factor10=72.7%
- `broad::other_formula_system`: rows=6132, p=2856, n=3276, MAE=0.781, RMSE=1.196, factor2=29.7%, factor10=74.2%
- `broad::GeTe_like`: rows=183, p=183, n=0, MAE=0.817, RMSE=2.043, factor2=86.3%, factor10=87.4%
- `broad::oxide`: rows=3437, p=2164, n=1273, MAE=1.016, RMSE=1.587, factor2=33.8%, factor10=67.3%
- `unknown_material_group`: rows=110, p=110, n=0, MAE=1.157, RMSE=1.605, factor2=40.9%, factor10=53.6%
- `broad::sulfide`: rows=892, p=512, n=380, MAE=1.443, RMSE=2.498, factor2=39.8%, factor10=66.5%

## Best By MAE
- `broad::SnTe_like`: MAE=0.165
- `broad::PbTe_like`: MAE=0.268
- `broad::BiTe_like`: MAE=0.372
- `broad::SbTe_like`: MAE=0.391
- `broad::telluride`: MAE=0.443
- `broad::selenide`: MAE=0.503
- `broad::BiSbTe_tetradymite_like`: MAE=0.598
- `broad::CoSb_skutterudite_like`: MAE=0.652
- `broad::Mg2SiSn_like`: MAE=0.690
- `broad::other_formula_system`: MAE=0.781

## Best By Factor2
- `broad::SnTe_like`: factor2=89.5%
- `broad::GeTe_like`: factor2=86.3%
- `broad::PbTe_like`: factor2=76.5%
- `broad::SbTe_like`: factor2=62.5%
- `broad::BiTe_like`: factor2=59.6%
- `broad::selenide`: factor2=53.8%
- `broad::Mg2SiSn_like`: factor2=52.8%
- `broad::BiSbTe_tetradymite_like`: factor2=45.9%
- `broad::telluride`: factor2=45.5%
- `broad::CoSb_skutterudite_like`: factor2=41.6%

## Best By Factor10
- `broad::SnTe_like`: factor10=96.8%
- `broad::PbTe_like`: factor10=96.0%
- `broad::BiTe_like`: factor10=92.5%
- `broad::telluride`: factor10=90.6%
- `broad::SbTe_like`: factor10=90.5%
- `broad::GeTe_like`: factor10=87.4%
- `broad::BiSbTe_tetradymite_like`: factor10=86.9%
- `broad::selenide`: factor10=85.2%
- `broad::CoSb_skutterudite_like`: factor10=80.9%
- `broad::other_formula_system`: factor10=74.2%

## Largest Outlier Groups
- `broad::CoSb_skutterudite_like`: max abs log10 error=14.571
- `broad::other_formula_system`: max abs log10 error=11.212
- `broad::sulfide`: max abs log10 error=8.412
- `broad::oxide`: max abs log10 error=7.909
- `broad::GeTe_like`: max abs log10 error=6.543
- `broad::selenide`: max abs log10 error=6.265
- `broad::BiSbTe_tetradymite_like`: max abs log10 error=4.883
- `broad::BiTe_like`: max abs log10 error=3.876
- `unknown_material_group`: max abs log10 error=3.730
- `broad::Mg2SiSn_like`: max abs log10 error=3.613

## Figures
- `broad_sulfide_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_all.pdf`
- `broad_sulfide_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_p.pdf`
- `broad_sulfide_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_n.png` / `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_scatter_pred_vs_exp_n.pdf`
- `broad_sulfide_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_error_hist_all.pdf`
- `broad_sulfide_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_error_hist_by_carrier.png` / `experiments\exp006\figures\focus_broad_families\broad_sulfide\broad_sulfide_error_hist_by_carrier.pdf`
- `unknown_material_group_scatter_all`: `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_scatter_pred_vs_exp_all.pdf`
- `unknown_material_group_scatter_p`: `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_scatter_pred_vs_exp_p.pdf`
- `unknown_material_group_error_hist_all`: `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\unknown_material_group\unknown_material_group_error_hist_all.pdf`
- `broad_oxide_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_all.pdf`
- `broad_oxide_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_p.pdf`
- `broad_oxide_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_n.png` / `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_scatter_pred_vs_exp_n.pdf`
- `broad_oxide_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_error_hist_all.pdf`
- `broad_oxide_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_error_hist_by_carrier.png` / `experiments\exp006\figures\focus_broad_families\broad_oxide\broad_oxide_error_hist_by_carrier.pdf`
- `broad_GeTe_like_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_scatter_pred_vs_exp_all.pdf`
- `broad_GeTe_like_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_scatter_pred_vs_exp_p.pdf`
- `broad_GeTe_like_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_GeTe_like\broad_GeTe_like_error_hist_all.pdf`
- `broad_other_formula_system_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_scatter_pred_vs_exp_all.pdf`
- `broad_other_formula_system_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_scatter_pred_vs_exp_p.pdf`
- `broad_other_formula_system_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_scatter_pred_vs_exp_n.png` / `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_scatter_pred_vs_exp_n.pdf`
- `broad_other_formula_system_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_error_hist_all.pdf`
- `broad_other_formula_system_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_error_hist_by_carrier.png` / `experiments\exp006\figures\focus_broad_families\broad_other_formula_system\broad_other_formula_system_error_hist_by_carrier.pdf`
- `broad_Mg2SiSn_like_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_Mg2SiSn_like\broad_Mg2SiSn_like_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_Mg2SiSn_like\broad_Mg2SiSn_like_scatter_pred_vs_exp_all.pdf`
- `broad_Mg2SiSn_like_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_Mg2SiSn_like\broad_Mg2SiSn_like_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_Mg2SiSn_like\broad_Mg2SiSn_like_scatter_pred_vs_exp_p.pdf`
- `broad_Mg2SiSn_like_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_Mg2SiSn_like\broad_Mg2SiSn_like_scatter_pred_vs_exp_n.png` / `experiments\exp006\figures\focus_broad_families\broad_Mg2SiSn_like\broad_Mg2SiSn_like_scatter_pred_vs_exp_n.pdf`
- `broad_Mg2SiSn_like_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_Mg2SiSn_like\broad_Mg2SiSn_like_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_Mg2SiSn_like\broad_Mg2SiSn_like_error_hist_all.pdf`
- `broad_Mg2SiSn_like_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_Mg2SiSn_like\broad_Mg2SiSn_like_error_hist_by_carrier.png` / `experiments\exp006\figures\focus_broad_families\broad_Mg2SiSn_like\broad_Mg2SiSn_like_error_hist_by_carrier.pdf`
- `broad_CoSb_skutterudite_like_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_CoSb_skutterudite_like\broad_CoSb_skutterudite_like_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_CoSb_skutterudite_like\broad_CoSb_skutterudite_like_scatter_pred_vs_exp_all.pdf`
- `broad_CoSb_skutterudite_like_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_CoSb_skutterudite_like\broad_CoSb_skutterudite_like_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_CoSb_skutterudite_like\broad_CoSb_skutterudite_like_scatter_pred_vs_exp_p.pdf`
- `broad_CoSb_skutterudite_like_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_CoSb_skutterudite_like\broad_CoSb_skutterudite_like_scatter_pred_vs_exp_n.png` / `experiments\exp006\figures\focus_broad_families\broad_CoSb_skutterudite_like\broad_CoSb_skutterudite_like_scatter_pred_vs_exp_n.pdf`
- `broad_CoSb_skutterudite_like_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_CoSb_skutterudite_like\broad_CoSb_skutterudite_like_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_CoSb_skutterudite_like\broad_CoSb_skutterudite_like_error_hist_all.pdf`
- `broad_CoSb_skutterudite_like_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_CoSb_skutterudite_like\broad_CoSb_skutterudite_like_error_hist_by_carrier.png` / `experiments\exp006\figures\focus_broad_families\broad_CoSb_skutterudite_like\broad_CoSb_skutterudite_like_error_hist_by_carrier.pdf`
- `broad_BiSbTe_tetradymite_like_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_BiSbTe_tetradymite_like\broad_BiSbTe_tetradymite_like_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_BiSbTe_tetradymite_like\broad_BiSbTe_tetradymite_like_scatter_pred_vs_exp_all.pdf`
- `broad_BiSbTe_tetradymite_like_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_BiSbTe_tetradymite_like\broad_BiSbTe_tetradymite_like_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_BiSbTe_tetradymite_like\broad_BiSbTe_tetradymite_like_scatter_pred_vs_exp_p.pdf`
- `broad_BiSbTe_tetradymite_like_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_BiSbTe_tetradymite_like\broad_BiSbTe_tetradymite_like_scatter_pred_vs_exp_n.png` / `experiments\exp006\figures\focus_broad_families\broad_BiSbTe_tetradymite_like\broad_BiSbTe_tetradymite_like_scatter_pred_vs_exp_n.pdf`
- `broad_BiSbTe_tetradymite_like_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_BiSbTe_tetradymite_like\broad_BiSbTe_tetradymite_like_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_BiSbTe_tetradymite_like\broad_BiSbTe_tetradymite_like_error_hist_all.pdf`
- `broad_BiSbTe_tetradymite_like_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_BiSbTe_tetradymite_like\broad_BiSbTe_tetradymite_like_error_hist_by_carrier.png` / `experiments\exp006\figures\focus_broad_families\broad_BiSbTe_tetradymite_like\broad_BiSbTe_tetradymite_like_error_hist_by_carrier.pdf`
- `broad_selenide_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_selenide\broad_selenide_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_selenide\broad_selenide_scatter_pred_vs_exp_all.pdf`
- `broad_selenide_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_selenide\broad_selenide_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_selenide\broad_selenide_scatter_pred_vs_exp_p.pdf`
- `broad_selenide_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_selenide\broad_selenide_scatter_pred_vs_exp_n.png` / `experiments\exp006\figures\focus_broad_families\broad_selenide\broad_selenide_scatter_pred_vs_exp_n.pdf`
- `broad_selenide_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_selenide\broad_selenide_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_selenide\broad_selenide_error_hist_all.pdf`
- `broad_selenide_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_selenide\broad_selenide_error_hist_by_carrier.png` / `experiments\exp006\figures\focus_broad_families\broad_selenide\broad_selenide_error_hist_by_carrier.pdf`
- `broad_telluride_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_telluride\broad_telluride_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_telluride\broad_telluride_scatter_pred_vs_exp_all.pdf`
- `broad_telluride_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_telluride\broad_telluride_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_telluride\broad_telluride_scatter_pred_vs_exp_p.pdf`
- `broad_telluride_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_telluride\broad_telluride_scatter_pred_vs_exp_n.png` / `experiments\exp006\figures\focus_broad_families\broad_telluride\broad_telluride_scatter_pred_vs_exp_n.pdf`
- `broad_telluride_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_telluride\broad_telluride_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_telluride\broad_telluride_error_hist_all.pdf`
- `broad_telluride_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_telluride\broad_telluride_error_hist_by_carrier.png` / `experiments\exp006\figures\focus_broad_families\broad_telluride\broad_telluride_error_hist_by_carrier.pdf`
- `broad_SbTe_like_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_SbTe_like\broad_SbTe_like_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_SbTe_like\broad_SbTe_like_scatter_pred_vs_exp_all.pdf`
- `broad_SbTe_like_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_SbTe_like\broad_SbTe_like_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_SbTe_like\broad_SbTe_like_scatter_pred_vs_exp_p.pdf`
- `broad_SbTe_like_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_SbTe_like\broad_SbTe_like_scatter_pred_vs_exp_n.png` / `experiments\exp006\figures\focus_broad_families\broad_SbTe_like\broad_SbTe_like_scatter_pred_vs_exp_n.pdf`
- `broad_SbTe_like_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_SbTe_like\broad_SbTe_like_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_SbTe_like\broad_SbTe_like_error_hist_all.pdf`
- `broad_SbTe_like_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_SbTe_like\broad_SbTe_like_error_hist_by_carrier.png` / `experiments\exp006\figures\focus_broad_families\broad_SbTe_like\broad_SbTe_like_error_hist_by_carrier.pdf`
- `broad_BiTe_like_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_BiTe_like\broad_BiTe_like_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_BiTe_like\broad_BiTe_like_scatter_pred_vs_exp_all.pdf`
- `broad_BiTe_like_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_BiTe_like\broad_BiTe_like_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_BiTe_like\broad_BiTe_like_scatter_pred_vs_exp_p.pdf`
- `broad_BiTe_like_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_BiTe_like\broad_BiTe_like_scatter_pred_vs_exp_n.png` / `experiments\exp006\figures\focus_broad_families\broad_BiTe_like\broad_BiTe_like_scatter_pred_vs_exp_n.pdf`
- `broad_BiTe_like_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_BiTe_like\broad_BiTe_like_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_BiTe_like\broad_BiTe_like_error_hist_all.pdf`
- `broad_BiTe_like_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_BiTe_like\broad_BiTe_like_error_hist_by_carrier.png` / `experiments\exp006\figures\focus_broad_families\broad_BiTe_like\broad_BiTe_like_error_hist_by_carrier.pdf`
- `broad_PbTe_like_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_PbTe_like\broad_PbTe_like_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_PbTe_like\broad_PbTe_like_scatter_pred_vs_exp_all.pdf`
- `broad_PbTe_like_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_PbTe_like\broad_PbTe_like_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_PbTe_like\broad_PbTe_like_scatter_pred_vs_exp_p.pdf`
- `broad_PbTe_like_scatter_n`: `experiments\exp006\figures\focus_broad_families\broad_PbTe_like\broad_PbTe_like_scatter_pred_vs_exp_n.png` / `experiments\exp006\figures\focus_broad_families\broad_PbTe_like\broad_PbTe_like_scatter_pred_vs_exp_n.pdf`
- `broad_PbTe_like_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_PbTe_like\broad_PbTe_like_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_PbTe_like\broad_PbTe_like_error_hist_all.pdf`
- `broad_PbTe_like_error_hist_by_carrier`: `experiments\exp006\figures\focus_broad_families\broad_PbTe_like\broad_PbTe_like_error_hist_by_carrier.png` / `experiments\exp006\figures\focus_broad_families\broad_PbTe_like\broad_PbTe_like_error_hist_by_carrier.pdf`
- `broad_SnTe_like_scatter_all`: `experiments\exp006\figures\focus_broad_families\broad_SnTe_like\broad_SnTe_like_scatter_pred_vs_exp_all.png` / `experiments\exp006\figures\focus_broad_families\broad_SnTe_like\broad_SnTe_like_scatter_pred_vs_exp_all.pdf`
- `broad_SnTe_like_scatter_p`: `experiments\exp006\figures\focus_broad_families\broad_SnTe_like\broad_SnTe_like_scatter_pred_vs_exp_p.png` / `experiments\exp006\figures\focus_broad_families\broad_SnTe_like\broad_SnTe_like_scatter_pred_vs_exp_p.pdf`
- `broad_SnTe_like_error_hist_all`: `experiments\exp006\figures\focus_broad_families\broad_SnTe_like\broad_SnTe_like_error_hist_all.png` / `experiments\exp006\figures\focus_broad_families\broad_SnTe_like\broad_SnTe_like_error_hist_all.pdf`

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
