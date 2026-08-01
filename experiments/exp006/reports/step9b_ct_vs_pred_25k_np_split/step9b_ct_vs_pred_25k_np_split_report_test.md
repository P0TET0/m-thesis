# Step9B: Step9A 25 K sigma_pred vs SS2026 old C(T)

## Purpose and inputs

- Purpose: compare the Step9A 25 K predicted electrical conductivity with the old SS2026 C(T) baseline.
- Prediction file: `experiments\exp006\data\processed\step9a_25k_bin_broad_family\step5b_test_predictions_valid.parquet`
- Config ID: `sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median`
- Old C(T) source script (read statically, not executed): `experiments\exp005\fit_tau_eff_step12.py`
- Old C(T) output CSV: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\sigma_predictions_step12.csv`
- Old C(T) column: `prefactor_C_S_per_m_step12`
- Old temperature column: `temperature_bin_K_step12`
- Old material column: `material_system`
- Old carrier column present but excluded from aggregation: `n_or_p`

## Old C(T) material mapping and aggregation

- The old C(T) curve was aggregated by `material_group_key_mapped` and `T_K` only.
- The `n_or_p` column was deliberately excluded, so one p/n-unsplit median C(T) curve is used for both figures in each material group.
- The Step12 `material_system` column is `unknown` in the available file. The `composition` column from the same Step12 output was therefore used as a fallback label.
- Rows mapped through the composition fallback: 326264
- Unmapped effective labels were retained in `step9b_unmatched_old_material_labels.csv`.

## Target material groups

- broad::SnTe_like
- broad::PbTe_like
- broad::BiTe_like

## Prediction and old C(T) counts

| material_group_key | carrier_type | prediction_points | old_ct_points | T_pred_min_K | T_pred_max_K | T_old_ct_min_K | T_old_ct_max_K | sigma_pred_median_S_per_m | old_C_T_median_S_per_m | median_log10_pred_over_oldCT_nearest | warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad::SnTe_like | p | 370 | 43 | 27.11864 | 880.814 | 0.0 | 1050.0 | 214848.17111881642 | 31399.16 | 0.756846942507277 |  |
| broad::SnTe_like | n | 0 | 43 |  |  | 0.0 | 1050.0 |  | 31399.16 |  | no_prediction_points;no_nearest_comparison |
| broad::PbTe_like | p | 267 | 42 | 101.3021 | 871.1859 | 0.0 | 1075.0 | 109346.2068798643 | 39355.84 | 0.3858290233049777 |  |
| broad::PbTe_like | n | 200 | 42 | 98.36292 | 776.7355 | 0.0 | 1075.0 | 148031.64471256657 | 39355.84 | 0.5238838543875648 |  |
| broad::BiTe_like | p | 497 | 50 | 6.021278 | 822.8013 | 0.0 | 1225.0 | 81000.87137925084 | 40416.74135123066 | 0.19216470635264482 |  |
| broad::BiTe_like | n | 1278 | 50 | 1.530612 | 755.518 | 0.0 | 1225.0 | 69545.28249304477 | 40416.74135123066 | 0.10586808161880734 |  |

## Median log10(sigma_pred / old C(T))

- Overall matched rows: 2612
- Overall median: 0.252604
- Q25 / Q75: 0.038302 / 0.515250
- Minimum / maximum: -0.876790 / 2.151275

| material_group_key | carrier_type | matched_rows | median_log10_pred_over_oldCT | q25 | q75 |
| --- | --- | --- | --- | --- | --- |
| broad::BiTe_like | n | 1278 | 0.10586808161880734 | -0.011945772960632262 | 0.27257657193011986 |
| broad::BiTe_like | p | 497 | 0.19216470635264482 | -0.048503240002935044 | 0.4099018935949381 |
| broad::PbTe_like | n | 200 | 0.5238838543875648 | 0.3321082969487379 | 0.7234997002841179 |
| broad::PbTe_like | p | 267 | 0.3858290233049777 | 0.2660821231937831 | 0.5250326369486085 |
| broad::SnTe_like | p | 370 | 0.756846942507277 | 0.6086936033429373 | 0.9411938094743835 |

## Figures

| figure_id | material_group_key | carrier_type | figure_path_png | figure_path_pdf | n_prediction_points | n_old_ct_points |
| --- | --- | --- | --- | --- | --- | --- |
| FIG_001 | broad::SnTe_like | p | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SnTe_like_p_sigma_pred_vs_oldCT_25k_test.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SnTe_like_p_sigma_pred_vs_oldCT_25k_test.pdf | 370 | 43 |
| FIG_002 | broad::SnTe_like | n | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SnTe_like_n_sigma_pred_vs_oldCT_25k_test.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SnTe_like_n_sigma_pred_vs_oldCT_25k_test.pdf | 0 | 43 |
| FIG_003 | broad::PbTe_like | p | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_PbTe_like_p_sigma_pred_vs_oldCT_25k_test.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_PbTe_like_p_sigma_pred_vs_oldCT_25k_test.pdf | 267 | 42 |
| FIG_004 | broad::PbTe_like | n | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_PbTe_like_n_sigma_pred_vs_oldCT_25k_test.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_PbTe_like_n_sigma_pred_vs_oldCT_25k_test.pdf | 200 | 42 |
| FIG_005 | broad::BiTe_like | p | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_BiTe_like_p_sigma_pred_vs_oldCT_25k_test.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_BiTe_like_p_sigma_pred_vs_oldCT_25k_test.pdf | 497 | 50 |
| FIG_006 | broad::BiTe_like | n | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_BiTe_like_n_sigma_pred_vs_oldCT_25k_test.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_BiTe_like_n_sigma_pred_vs_oldCT_25k_test.pdf | 1278 | 50 |

## Missing data

- Material groups without old C(T): none
- Material-group/carrier combinations without prediction points:
  - broad::SnTe_like / n

## How to read the figures

- Points are the Step9A 25 K predicted electrical conductivity.
- The line is the old SS2026 C(T).
- The line has no p/n split and is identical in the p and n figures for a material group.
- Only the point cloud is separated into p-type and n-type figures.

## Notes

- Measured sigma is not included in the main figures.
- sigma0_ref is not included in the figures.
- No new sigma_pred was calculated.
- This step only visualizes the existing Step9A predictions.
- Step4 full-data reference curves were not used.
- Starrydata2 raw data was not read.
- elapsed_seconds: 14.94
