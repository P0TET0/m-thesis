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
- material groups: broad::SnTe_like, broad::PbTe_like, broad::BiTe_like, broad::SbTe_like, broad::SiGe_like, broad::oxide, broad::sulfide
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
- broad::SbTe_like / p: predictions=671, old_ct=36, median_log10_pred_over_oldCT=0.3751680522620358, warning=
- broad::SbTe_like / n: predictions=303, old_ct=35, median_log10_pred_over_oldCT=0.2197776142530574, warning=
- broad::SiGe_like / p: predictions=62, old_ct=0, median_log10_pred_over_oldCT=nan, warning=no_old_ct;no_nearest_comparison
- broad::SiGe_like / n: predictions=135, old_ct=13, median_log10_pred_over_oldCT=0.3652769906438341, warning=
- broad::oxide / p: predictions=2164, old_ct=51, median_log10_pred_over_oldCT=-0.6837759334226763, warning=
- broad::oxide / n: predictions=1273, old_ct=52, median_log10_pred_over_oldCT=-0.4414465671336525, warning=
- broad::sulfide / p: predictions=512, old_ct=45, median_log10_pred_over_oldCT=-0.07418466486131749, warning=
- broad::sulfide / n: predictions=380, old_ct=51, median_log10_pred_over_oldCT=-0.0928790431917759, warning=

## Overall Ratio Summary
- count: 8056
- median log10(sigma_pred / old C(T)): -0.0115885
- min/max: -2.57202 / 2.01512

## Figures
- broad::SnTe_like / p / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_SnTe_like_p_sigma_pred_points_vs_oldCT_line.png`
- broad::PbTe_like / p / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_PbTe_like_p_sigma_pred_points_vs_oldCT_line.png`
- broad::PbTe_like / n / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_PbTe_like_n_sigma_pred_points_vs_oldCT_line.png`
- broad::BiTe_like / p / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_BiTe_like_p_sigma_pred_points_vs_oldCT_line.png`
- broad::BiTe_like / n / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_BiTe_like_n_sigma_pred_points_vs_oldCT_line.png`
- broad::SbTe_like / p / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_SbTe_like_p_sigma_pred_points_vs_oldCT_line.png`
- broad::SbTe_like / n / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_SbTe_like_n_sigma_pred_points_vs_oldCT_line.png`
- broad::SiGe_like / n / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_SiGe_like_n_sigma_pred_points_vs_oldCT_line.png`
- broad::oxide / p / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_oxide_p_sigma_pred_points_vs_oldCT_line.png`
- broad::oxide / n / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_oxide_n_sigma_pred_points_vs_oldCT_line.png`
- broad::sulfide / p / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_sulfide_p_sigma_pred_points_vs_oldCT_line.png`
- broad::sulfide / n / sigma_pred_vs_old_ct: `experiments\exp006\figures\focus_pred_sigma_vs_old_ct\broad_sulfide_n_sigma_pred_points_vs_oldCT_line.png`

## Missing Combinations
- broad::SnTe_like / n: no_prediction_points;no_nearest_comparison
- broad::SiGe_like / p: no_old_ct;no_nearest_comparison

## Notes
- Points are current predicted sigma.
- Lines are SS2026 old C(T).
- Experimental sigma points are not included in the main figures.
- sigma0_ref is not included in these figures.
- No new sigma_pred is calculated.
- Step4 full-data reference curves are not used.
- Starrydata2 raw data are not read.
