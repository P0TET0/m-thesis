# Predicted Sigma vs Old C(T)

## Inputs
- prediction file: `experiments\exp006\data\processed\step6b_broad_family\step5b_test_predictions_valid.parquet`
- old C(T) source script: `experiments\exp005\fit_tau_eff_step12.py`
- detected old C(T) CSV: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\sigma_predictions_step12.csv`
- adopted old C(T) column: `prefactor_C_S_per_m_step12`
- temperature column: `temperature_bin_K_step12`
- material column: `material_system`
- carrier column: `n_or_p`

## Targets
- material groups: broad::SnTe_like, broad::PbTe_like, broad::BiTe_like
- carrier types: p, n

## How To Read The Figures
- Points are current broad_family predicted electrical conductivity, sigma_pred.
- Lines are SS2026 old C(T).
- If points lie near the old C(T) line, the current prediction has a similar scale and temperature dependence to the old C(T) baseline.
- If points are far from the line, the S-input prediction gives values different from the old C(T) baseline.
- Exact agreement is not required: old C(T) is based on measured sigma, while sigma_pred is predicted by the S-input method.

## Summary
- broad::SnTe_like / p: predictions=370, old_ct=37, median_log10_pred_over_oldCT=0.783519410539723, warning=
- broad::SnTe_like / n: predictions=0, old_ct=10, median_log10_pred_over_oldCT=nan, warning=no_prediction_points;no_nearest_comparison
- broad::PbTe_like / p: predictions=269, old_ct=28, median_log10_pred_over_oldCT=0.37147765185446174, warning=
- broad::PbTe_like / n: predictions=203, old_ct=38, median_log10_pred_over_oldCT=0.3899038080175999, warning=
- broad::BiTe_like / p: predictions=498, old_ct=48, median_log10_pred_over_oldCT=0.4260142551629266, warning=
- broad::BiTe_like / n: predictions=1278, old_ct=32, median_log10_pred_over_oldCT=0.012164741734832816, warning=

## Overall Ratio Summary
- count: 2618
- median log10(sigma_pred / old C(T)): 0.248361
- min/max: -0.998064 / 2.01512

## Figures
- broad::SnTe_like / p / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_SnTe_like_p_sigma_pred_points_vs_oldCT_line_test.png`
- broad::PbTe_like / p / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_PbTe_like_p_sigma_pred_points_vs_oldCT_line_test.png`
- broad::PbTe_like / n / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_PbTe_like_n_sigma_pred_points_vs_oldCT_line_test.png`
- broad::BiTe_like / p / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_BiTe_like_p_sigma_pred_points_vs_oldCT_line_test.png`
- broad::BiTe_like / n / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_BiTe_like_n_sigma_pred_points_vs_oldCT_line_test.png`

## Missing Combinations
- broad::SnTe_like / n: no_prediction_points;no_nearest_comparison

## Notes
- Points are current predicted sigma.
- Lines are SS2026 old C(T).
- Experimental sigma points are not included in the main figures.
- sigma0_ref is not included in these figures.
- No new sigma_pred is calculated.
- Step4 full-data reference curves are not used.
- Starrydata2 raw data are not read.
