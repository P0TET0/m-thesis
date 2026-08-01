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
- broad::SbTe_like
- broad::SiGe_like
- broad::oxide
- broad::sulfide

## Prediction and old C(T) counts

| material_group_key | carrier_type | prediction_points | old_ct_points | T_pred_min_K | T_pred_max_K | T_old_ct_min_K | T_old_ct_max_K | sigma_pred_median_S_per_m | old_C_T_median_S_per_m | median_log10_pred_over_oldCT_nearest | warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad::SnTe_like | p | 370 | 43 | 27.11864 | 880.814 | 0.0 | 1050.0 | 214848.17111881642 | 31399.16 | 0.756846942507277 |  |
| broad::SnTe_like | n | 0 | 43 |  |  | 0.0 | 1050.0 |  | 31399.16 |  | no_prediction_points;no_nearest_comparison |
| broad::PbTe_like | p | 267 | 42 | 101.3021 | 871.1859 | 0.0 | 1075.0 | 109346.2068798643 | 39355.84 | 0.3858290233049777 |  |
| broad::PbTe_like | n | 200 | 42 | 98.36292 | 776.7355 | 0.0 | 1075.0 | 148031.64471256657 | 39355.84 | 0.5238838543875648 |  |
| broad::BiTe_like | p | 497 | 50 | 6.021278 | 822.8013 | 0.0 | 1225.0 | 81000.87137925084 | 40416.74135123066 | 0.19216470635264482 |  |
| broad::BiTe_like | n | 1278 | 50 | 1.530612 | 755.518 | 0.0 | 1225.0 | 69545.28249304477 | 40416.74135123066 | 0.10586808161880734 |  |
| broad::SbTe_like | p | 669 | 52 | 1.83863 | 871.43 | 0.0 | 1275.0 | 97066.258058782 | 35861.100000000006 | 0.34871492743730137 |  |
| broad::SbTe_like | n | 302 | 52 | 2.085701 | 747.5085 | 0.0 | 1275.0 | 87353.79118952042 | 35861.100000000006 | 0.28834750931821906 |  |
| broad::SiGe_like | p | 57 | 53 | 21.96653 | 473.7352 | 0.0 | 1300.0 | 146494.04198422277 | 45592.27688577714 | 0.3949758165224422 |  |
| broad::SiGe_like | n | 131 | 53 | 78.32849 | 776.0492 | 0.0 | 1300.0 | 146446.99198667638 | 45592.27688577714 | 0.45477967198706093 |  |
| broad::oxide | p | 2161 | 52 | 4.38749 | 1201.907 | 0.0 | 1275.0 | 6460.378752116374 | 30355.91425733133 | -0.7698373279119641 |  |
| broad::oxide | n | 1272 | 52 | 2.008032 | 1284.759 | 0.0 | 1275.0 | 19284.533416592436 | 30355.91425733133 | -0.28739456460821566 |  |
| broad::sulfide | p | 511 | 54 | 4.08054 | 948.4342 | 0.0 | 1400.0 | 34658.048014693006 | 38248.58855733524 | -0.14981831965103876 |  |
| broad::sulfide | n | 374 | 54 | 3.174603 | 926.7281 | 0.0 | 1400.0 | 42706.0937063199 | 38248.58855733524 | 0.021956876319103813 |  |

## Median log10(sigma_pred / old C(T))

- Overall matched rows: 8089
- Overall median: 0.019992
- Q25 / Q75: -0.554858 / 0.354119
- Minimum / maximum: -2.252093 / 2.151275

| material_group_key | carrier_type | matched_rows | median_log10_pred_over_oldCT | q25 | q75 |
| --- | --- | --- | --- | --- | --- |
| broad::BiTe_like | n | 1278 | 0.10586808161880734 | -0.011945772960632262 | 0.27257657193011986 |
| broad::BiTe_like | p | 497 | 0.19216470635264482 | -0.048503240002935044 | 0.4099018935949381 |
| broad::PbTe_like | n | 200 | 0.5238838543875648 | 0.3321082969487379 | 0.7234997002841179 |
| broad::PbTe_like | p | 267 | 0.3858290233049777 | 0.2660821231937831 | 0.5250326369486085 |
| broad::SbTe_like | n | 302 | 0.28834750931821906 | 0.11605495823909062 | 0.48426911332382083 |
| broad::SbTe_like | p | 669 | 0.34871492743730137 | 0.19530090730311223 | 0.5421190654492163 |
| broad::SiGe_like | n | 131 | 0.45477967198706093 | 0.19430271840032187 | 0.9363987593879375 |
| broad::SiGe_like | p | 57 | 0.3949758165224422 | 0.04987978012384353 | 0.5364083564475326 |
| broad::SnTe_like | p | 370 | 0.756846942507277 | 0.6086936033429373 | 0.9411938094743835 |
| broad::oxide | n | 1272 | -0.28739456460821566 | -0.5529524347336867 | -0.01789323601217019 |
| broad::oxide | p | 2161 | -0.7698373279119641 | -0.9302590182497661 | -0.39806042189132107 |
| broad::sulfide | n | 374 | 0.021956876319103813 | -0.2471116382616354 | 0.2525741449777368 |
| broad::sulfide | p | 511 | -0.14981831965103876 | -0.573751546617697 | 0.1126878414268262 |

## Figures

| figure_id | material_group_key | carrier_type | figure_path_png | figure_path_pdf | n_prediction_points | n_old_ct_points |
| --- | --- | --- | --- | --- | --- | --- |
| FIG_001 | broad::SnTe_like | p | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SnTe_like_p_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SnTe_like_p_sigma_pred_vs_oldCT_25k.pdf | 370 | 43 |
| FIG_002 | broad::SnTe_like | n | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SnTe_like_n_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SnTe_like_n_sigma_pred_vs_oldCT_25k.pdf | 0 | 43 |
| FIG_003 | broad::PbTe_like | p | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_PbTe_like_p_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_PbTe_like_p_sigma_pred_vs_oldCT_25k.pdf | 267 | 42 |
| FIG_004 | broad::PbTe_like | n | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_PbTe_like_n_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_PbTe_like_n_sigma_pred_vs_oldCT_25k.pdf | 200 | 42 |
| FIG_005 | broad::BiTe_like | p | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_BiTe_like_p_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_BiTe_like_p_sigma_pred_vs_oldCT_25k.pdf | 497 | 50 |
| FIG_006 | broad::BiTe_like | n | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_BiTe_like_n_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_BiTe_like_n_sigma_pred_vs_oldCT_25k.pdf | 1278 | 50 |
| FIG_007 | broad::SbTe_like | p | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SbTe_like_p_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SbTe_like_p_sigma_pred_vs_oldCT_25k.pdf | 669 | 52 |
| FIG_008 | broad::SbTe_like | n | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SbTe_like_n_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SbTe_like_n_sigma_pred_vs_oldCT_25k.pdf | 302 | 52 |
| FIG_009 | broad::SiGe_like | p | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SiGe_like_p_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SiGe_like_p_sigma_pred_vs_oldCT_25k.pdf | 57 | 53 |
| FIG_010 | broad::SiGe_like | n | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SiGe_like_n_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_SiGe_like_n_sigma_pred_vs_oldCT_25k.pdf | 131 | 53 |
| FIG_011 | broad::oxide | p | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_oxide_p_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_oxide_p_sigma_pred_vs_oldCT_25k.pdf | 2161 | 52 |
| FIG_012 | broad::oxide | n | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_oxide_n_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_oxide_n_sigma_pred_vs_oldCT_25k.pdf | 1272 | 52 |
| FIG_013 | broad::sulfide | p | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_sulfide_p_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_sulfide_p_sigma_pred_vs_oldCT_25k.pdf | 511 | 54 |
| FIG_014 | broad::sulfide | n | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_sulfide_n_sigma_pred_vs_oldCT_25k.png | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split\broad_sulfide_n_sigma_pred_vs_oldCT_25k.pdf | 374 | 54 |

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
- elapsed_seconds: 20.63
