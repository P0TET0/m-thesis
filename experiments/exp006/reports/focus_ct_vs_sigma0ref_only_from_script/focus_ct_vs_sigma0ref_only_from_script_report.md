# Old C(T) vs Sigma0_ref(T) Only From Script

## Old C(T) Source
- old C(T) source script: `experiments\exp005\fit_tau_eff_step12.py`
- detected old C(T) CSV: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\sigma_predictions_step12.csv`
- adopted old C(T) column: `prefactor_C_S_per_m_step12`
- temperature column: `temperature_bin_K_step12`
- material column: `material_system`
- carrier_type column: `n_or_p`
- detection note: selected: primary Step12 prediction output contains prefactor_C_S_per_m_step12 required by fit_tau_eff_step12.py

## Current Sigma0_ref(T)
- sigma0_ref file: `experiments\exp006\data\processed\step6b_broad_family\step5b_train_reference_curve_bins.parquet`
- filter: sample_holdout / conservative_valid / all_valid / material_family / sample_median; reference-bin candidates preferred

## Targets
- material groups: broad::SnTe_like, broad::PbTe_like, broad::BiTe_like, broad::SbTe_like, broad::SiGe_like, broad::oxide, broad::sulfide
- carrier types: p, n

## Physical Interpretation
- Old C(T) is the SS2026 empirical baseline against measured electrical conductivity.
- sigma0_ref(T) is the Seebeck-derived coefficient after Fermi-level correction.
- They both have units of S/m, but they are not the same physical quantity.
- This overlay is for comparing temperature-dependence shape and scale, not for treating the curves as identical observables.

## Summary By Group And Carrier
- broad::SnTe_like / p: old_ct=1121, sigma0_ref=10, comparison=1121, median_log10_ratio=0.2132079962140708, warning=
- broad::SnTe_like / n: old_ct=18, sigma0_ref=0, comparison=0, median_log10_ratio=nan, warning=no_sigma0_ref;no_comparison
- broad::PbTe_like / p: old_ct=874, sigma0_ref=9, comparison=874, median_log10_ratio=0.028733600370436534, warning=
- broad::PbTe_like / n: old_ct=1666, sigma0_ref=8, comparison=1666, median_log10_ratio=0.05079349636037006, warning=
- broad::BiTe_like / p: old_ct=747, sigma0_ref=9, comparison=747, median_log10_ratio=-0.012459776471492452, warning=
- broad::BiTe_like / n: old_ct=1061, sigma0_ref=9, comparison=1061, median_log10_ratio=-0.25796701760696134, warning=
- broad::SbTe_like / p: old_ct=1974, sigma0_ref=10, comparison=1974, median_log10_ratio=0.06450975309747416, warning=
- broad::SbTe_like / n: old_ct=775, sigma0_ref=10, comparison=775, median_log10_ratio=-0.06626550260874607, warning=
- broad::SiGe_like / p: old_ct=0, sigma0_ref=6, comparison=0, median_log10_ratio=nan, warning=no_old_ct;no_comparison
- broad::SiGe_like / n: old_ct=18, sigma0_ref=10, comparison=18, median_log10_ratio=-0.43482984860791507, warning=
- broad::oxide / p: old_ct=8718, sigma0_ref=13, comparison=8718, median_log10_ratio=-0.9815056725370367, warning=
- broad::oxide / n: old_ct=7652, sigma0_ref=14, comparison=7652, median_log10_ratio=-0.8279033332793823, warning=
- broad::sulfide / p: old_ct=6432, sigma0_ref=11, comparison=6432, median_log10_ratio=-0.2163803406149253, warning=
- broad::sulfide / n: old_ct=4972, sigma0_ref=12, comparison=4972, median_log10_ratio=-0.4333556831006569, warning=

## Overall Ratio Summary
- count: 36010
- median log10(sigma0_ref / old_C_T): -0.585507
- min/max: -3.70226 / 0.939

## Figures
- broad::SnTe_like / p / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_SnTe_like_p_oldCT_vs_sigma0ref_overlay.png`
- broad::SnTe_like / p / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_SnTe_like_p_log_ratio_sigma0ref_over_oldCT.png`
- broad::PbTe_like / p / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_PbTe_like_p_oldCT_vs_sigma0ref_overlay.png`
- broad::PbTe_like / p / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_PbTe_like_p_log_ratio_sigma0ref_over_oldCT.png`
- broad::PbTe_like / n / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_PbTe_like_n_oldCT_vs_sigma0ref_overlay.png`
- broad::PbTe_like / n / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_PbTe_like_n_log_ratio_sigma0ref_over_oldCT.png`
- broad::BiTe_like / p / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_BiTe_like_p_oldCT_vs_sigma0ref_overlay.png`
- broad::BiTe_like / p / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_BiTe_like_p_log_ratio_sigma0ref_over_oldCT.png`
- broad::BiTe_like / n / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_BiTe_like_n_oldCT_vs_sigma0ref_overlay.png`
- broad::BiTe_like / n / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_BiTe_like_n_log_ratio_sigma0ref_over_oldCT.png`
- broad::SbTe_like / p / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_SbTe_like_p_oldCT_vs_sigma0ref_overlay.png`
- broad::SbTe_like / p / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_SbTe_like_p_log_ratio_sigma0ref_over_oldCT.png`
- broad::SbTe_like / n / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_SbTe_like_n_oldCT_vs_sigma0ref_overlay.png`
- broad::SbTe_like / n / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_SbTe_like_n_log_ratio_sigma0ref_over_oldCT.png`
- broad::SiGe_like / n / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_SiGe_like_n_oldCT_vs_sigma0ref_overlay.png`
- broad::SiGe_like / n / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_SiGe_like_n_log_ratio_sigma0ref_over_oldCT.png`
- broad::oxide / p / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_oxide_p_oldCT_vs_sigma0ref_overlay.png`
- broad::oxide / p / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_oxide_p_log_ratio_sigma0ref_over_oldCT.png`
- broad::oxide / n / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_oxide_n_oldCT_vs_sigma0ref_overlay.png`
- broad::oxide / n / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_oxide_n_log_ratio_sigma0ref_over_oldCT.png`
- broad::sulfide / p / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_sulfide_p_oldCT_vs_sigma0ref_overlay.png`
- broad::sulfide / p / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_sulfide_p_log_ratio_sigma0ref_over_oldCT.png`
- broad::sulfide / n / overlay: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_sulfide_n_oldCT_vs_sigma0ref_overlay.png`
- broad::sulfide / n / log_ratio: `experiments\exp006\figures\focus_ct_vs_sigma0ref_only_from_script\broad_sulfide_n_log_ratio_sigma0ref_over_oldCT.png`

## Warnings
- broad::SnTe_like / n: no_sigma0_ref;no_comparison
- broad::SiGe_like / p: no_old_ct;no_comparison

## Unmatched Old Material Labels
- unmatched unique labels: 5527
- ((C6H7N)94.94(C3N4)5.06)2.39(C)97.61
- (Al0.995Mg0.005Sb)0.9(Zn4Sb3)0.1
- (AlSb)0.7(Zn4Sb3)0.3
- (AlSb)0.8(Zn4Sb3)0.2
- (AlSb)0.9(Zn4Sb3)0.1
- (B4C)81.26(TiO2)18.74
- (B4C)85.25(TiO2)14.75
- (B4C)89.12(TiO2)10.88
- (B4C)92.86(TiO2)7.14
- (B4C)96.49(TiO2)3.51
- (Bi85Sb15)0.8Sn0.2
- (Bi85Sb15)0.95Sn0.05
- (Bi85Sb15)0.975Sn0.025
- (BiCuSeO)0.66(CuSe2)0.34
- (BiCuSeO)0.7(CuSe2)0.3
- (BiCuSeO)0.73(CuSe2)0.27
- (BiCuSeO)0.76(CuSe2)0.24
- (BiS)1.2(TiS2)2
- (BiSe)1.09TaSe2/TaSe2
- (BrC6H4NH2)2CuBr2
- (C6H7N)23.13(C)76.87
- (C6H7N)34.03(C)65.97
- (C6H7N)53.72(C)46.28
- (C6H7N)94.94(C3N4)5.06
- (CH3NH3I)100(PbI2)95(BiI3)5
- (CH3NH3I)100(PbI2)97(BiI3)3
- (CH3NH3I)100(PbI2)99(BiI3)1
- (Ca0.25Ba0.75)0.995Na0.005Mg2Bi1.98
- (Ca0.2Sr0.2Ba0.2Pb0.2La0.2)TiO3
- (Ca0.5Ba0.5)0.09(Ce0.508La0.281Nd0.161Pr0.05)0.09Co4Sb12
- (Ca0.5Ba0.5)0.995Na0.005Mg2Bi1.98
- (Ca0.5Sr0.5)0.09(Ce0.508La0.281Nd0.161Pr0.05)0.09Co4Sb12
- (Ca0.75Ba0.25)0.995Na0.005Mg2Bi1.98
- (Ca1.8Ba0.2CoO3)0.62(CoO2)
- (Ca1.8Pb0.2CoO3)0.62(CoO2)
- (Ca1.9Ba0.1CoO3)0.62(CoO2)
- (Ca1.9Pb0.1CoO3)0.62(CoO2)
- (Ca2CoO2.85)0.62(CoO2)
- (Ca2CoO2.94)0.62(CoO2)
- (Ca2CoO3)0.62(CoO2)
- (Ca2CoO3)0.7CoO2
- (Ce0.5092La0.2841Nd0.1568Pr0.0498)0.33(Ba0.33Yb0.33)0.35Co4Sb12.3
- (Ce0.5092La0.2841Nd0.1568Pr0.0498)0.86Fe4Sb12
- (Ce0.5092La0.2841Nd0.1568Pr0.0498)1.1Fe4Sb12
- (Ce0.5092La0.2841Nd0.1568Pr0.0498)1.4Fe4Sb12
- (Ce0.5092La0.2841Nd0.1568Pr0.0498)Fe4Sb12
- (Ce0.5278La0.2365Nd0.1686Pr0.0595Si0.005Fe0.0025)0.55Fe2.44Co1.56Sb11.96
- (Ce0.5278La0.2365Nd0.1686Pr0.0595Si0.005Fe0.0025)0.65Fe2.92Co1.08Sb11.98
- (Ce0.5278La0.2365Nd0.1686Pr0.0595Si0.005Fe0.0025)0.72Fe3.43Co0.57Sb11.97
- (Ce0.5278La0.2365Nd0.1686Pr0.0595Si0.005Fe0.0025)0.82Fe4.00Sb11.96

## Notes
- Old C(T) and sigma0_ref(T) have different meanings.
- No new sigma_pred is calculated.
- Step4 full-data reference curves are not used.
- Starrydata2 raw data are not read.
