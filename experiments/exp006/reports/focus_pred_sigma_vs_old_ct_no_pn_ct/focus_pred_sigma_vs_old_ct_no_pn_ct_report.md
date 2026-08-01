# Predicted sigma vs no-p/n old C(T)

This is a separate focus analysis outside the existing Step0-Step7C pipeline.

## Inputs
- Prediction file: `experiments\exp006\data\processed\step6b_broad_family\step5b_test_predictions_valid.parquet`
- Old C(T) source script: `experiments\exp005\fit_tau_eff_step12.py`
- Detected old C(T) CSV: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\sigma_predictions_step12.csv`
- Old C(T) column: `prefactor_C_S_per_m_step12`
- Temperature column: `temperature_bin_K_step12`
- Material column: `material_system`
- Ignored p/n column from old C(T), if present: `n_or_p`

## Old C(T) Aggregation
- Old C(T) is aggregated without p/n splitting.
- Aggregation method: median over material group x temperature.
- The n_or_p/carrier column is not used for the old C(T) curve.
- Each material group has at most one old C(T) line.

## Target Material Groups
- broad::SnTe_like
- broad::PbTe_like
- broad::BiTe_like
- broad::SbTe_like
- broad::SiGe_like
- broad::oxide
- broad::sulfide

## Group Summary
- broad::SnTe_like: predictions=370 (p=370, n=0), old_ct=37, median_log10_pred_over_oldCT=0.7633839585620497, warning=none
- broad::PbTe_like: predictions=472 (p=269, n=203), old_ct=40, median_log10_pred_over_oldCT=0.39282230742921787, warning=none
- broad::BiTe_like: predictions=1776 (p=498, n=1278), old_ct=48, median_log10_pred_over_oldCT=0.1634064465437126, warning=none
- broad::SbTe_like: predictions=974 (p=671, n=303), old_ct=38, median_log10_pred_over_oldCT=0.34562934207764184, warning=none
- broad::SiGe_like: predictions=197 (p=62, n=135), old_ct=13, median_log10_pred_over_oldCT=0.3847342169248093, warning=none
- broad::oxide: predictions=3437 (p=2164, n=1273), old_ct=52, median_log10_pred_over_oldCT=-0.5647743981737064, warning=none
- broad::sulfide: predictions=892 (p=512, n=380), old_ct=51, median_log10_pred_over_oldCT=0.01264238379299092, warning=none

## Missing Data
- Material groups without old C(T): none
- Material groups without prediction points: none

## Ratio Overview
- count: 8118
- min log10(sigma_pred / old C(T)): -2.99752
- median log10(sigma_pred / old C(T)): 0.0582081
- max log10(sigma_pred / old C(T)): 2.0162

## Figures
- broad::SnTe_like / sigma_pred_vs_old_ct_no_pn: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct_no_pn_ct\broad_SnTe_like_sigma_pred_points_vs_oldCT_no_pn_line.png`
- broad::PbTe_like / sigma_pred_vs_old_ct_no_pn: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct_no_pn_ct\broad_PbTe_like_sigma_pred_points_vs_oldCT_no_pn_line.png`
- broad::BiTe_like / sigma_pred_vs_old_ct_no_pn: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct_no_pn_ct\broad_BiTe_like_sigma_pred_points_vs_oldCT_no_pn_line.png`
- broad::SbTe_like / sigma_pred_vs_old_ct_no_pn: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct_no_pn_ct\broad_SbTe_like_sigma_pred_points_vs_oldCT_no_pn_line.png`
- broad::SiGe_like / sigma_pred_vs_old_ct_no_pn: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct_no_pn_ct\broad_SiGe_like_sigma_pred_points_vs_oldCT_no_pn_line.png`
- broad::oxide / sigma_pred_vs_old_ct_no_pn: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct_no_pn_ct\broad_oxide_sigma_pred_points_vs_oldCT_no_pn_line.png`
- broad::sulfide / sigma_pred_vs_old_ct_no_pn: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct_no_pn_ct\broad_sulfide_sigma_pred_points_vs_oldCT_no_pn_line.png`

## How To Read The Figures
- Points are current predicted sigma values from the broad_family prediction result.
- The line is the SS2026 old C(T) curve aggregated without p/n splitting.
- If the points lie near the old C(T) line, the prediction has a similar temperature-dependent scale to the old C(T) baseline.
- If the points are far from the line, the S-input prediction differs from the old C(T) baseline.
- Because old C(T) is not split by p/n, compare where p-type and n-type point clouds sit relative to the same line.

## Notes
- Points are current predicted sigma.
- The line is the no-p/n SS2026 old C(T).
- Experimental sigma points are not included in the main figures.
- sigma0_ref is not included in the figures.
- No new sigma_pred is calculated.
- Step4 full-data reference curves are not used.
- Starrydata2 raw data are not read.
