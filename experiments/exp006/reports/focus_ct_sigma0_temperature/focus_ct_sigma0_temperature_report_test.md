# Focus C(T) vs Sigma0 Temperature Comparison

## Inputs
- current_rows: `experiments\exp006\data\processed\step6a_validation_rows_with_splits_key_broad_family.parquet`
- current_sigma0_ref: `experiments\exp006\data\processed\step6b_broad_family\step5b_train_reference_curve_bins.parquet`
- old_ct: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\sigma_predictions_step12.csv`

## Old C(T) File Selection
- Candidate Step12 C(T) files are scanned for temperature and C(T)-like columns.
- old C(T) source script: `experiments\exp005\fit_tau_eff_step12.py`
- source mode: `script`
- script output directory: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit`
- selected old C(T) file: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\sigma_predictions_step12.csv`
- selected old C(T) column: `prefactor_C_S_per_m_step12`
- temperature column: `temperature_bin_K_step12`
- material columns: `material_system;composition;prefactor_group_key_step12`
- carrier type column: `n_or_p`
- adoption reason: fit_tau_eff_step12.py writes sigma_predictions_step12.csv with temperature_K, temperature_bin_K_step12, material_system/composition, n_or_p, and prefactor_C_S_per_m_step12; assert_acceptance also requires prefactor_C_S_per_m_step12 in sigma_predictions_step12.

### Old C(T) Candidates
- selected: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\sigma_predictions_step12.csv`; ct_candidates=`prefactor_C_S_per_m_step12`; temperature=`temperature_bin_K_step12`; material=`material_system`; carrier=`n_or_p`; reason=fit_tau_eff_step12.py writes sigma_predictions_step12.csv with temperature_K, temperature_bin_K_step12, material_system/composition, n_or_p, and prefactor_C_S_per_m_step12; assert_acceptance also requires prefactor_C_S_per_m_step12 in sigma_predictions_step12.
- not selected: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\initial_tau_fit_predictions_step12.csv`; ct_candidates=`prefactor_C_S_per_m_step12`; temperature=`temperature_bin_K_step12`; material=`material_system`; carrier=`n_or_p`; reason=not adopted: filtered fit-ok subset; sigma_predictions_step12.csv is the primary full Step12 prediction output
- not selected: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\prefactor_baseline_audit_step12.csv`; ct_candidates=`median_prefactor_C_S_per_m_step12`; temperature=`temperature_bin_K_step12`; material=`prefactor_group_key_step12`; carrier=``; reason=not adopted: audit summary of prefactors, not row-level Step12 C(T) data
- not selected: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\tau_fit_results_step12.csv`; ct_candidates=``; temperature=``; material=`material_system`; carrier=`n_or_p`; reason=not adopted: no C(T)-like column found
- not selected: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\tau_fit_ready_samples_step12.csv`; ct_candidates=``; temperature=``; material=`material_system`; carrier=`n_or_p`; reason=not adopted: no C(T)-like column found
- not selected: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\tau_fit_problem_samples_step12.csv`; ct_candidates=``; temperature=``; material=`material_system`; carrier=`n_or_p`; reason=not adopted: no C(T)-like column found
- not selected: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\tau_fit_material_summary_step12.csv`; ct_candidates=``; temperature=``; material=`material_system`; carrier=`n_or_p`; reason=not adopted: no C(T)-like column found
- not selected: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\tau_fit_holdout_eval_step12.csv`; ct_candidates=``; temperature=``; material=`material_system`; carrier=`n_or_p`; reason=not adopted: no C(T)-like column found

### Rejected Script-Derived Candidates
- C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\tau_fit_results_step12.csv: missing temperature or old C(T) column
- C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\tau_fit_ready_samples_step12.csv: missing temperature or old C(T) column
- C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\tau_fit_problem_samples_step12.csv: missing temperature or old C(T) column
- C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\tau_fit_material_summary_step12.csv: missing temperature or old C(T) column
- C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step12_tau_fit\tau_fit_holdout_eval_step12.csv: missing temperature or old C(T) column

## Targets
- material groups: broad::SnTe_like, broad::PbTe_like, broad::BiTe_like
- carrier types: p, n

## Physical Difference
- Old C(T) is the empirical electrical-conductivity baseline from Step12 tau_eff fitting.
- The current sigma0_ref is a Seebeck-derived coefficient corrected using Fermi-level information.
- They share units of S/m, but they are not the same physical quantity.
- The two-panel figure is the main figure so measured sigma and Seebeck-derived coefficients are not visually conflated.

## Figure List
- broad::SnTe_like / p / two_panel: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_SnTe_like_p_sigma_C_and_sigma0_two_panel_test.png`
- broad::SnTe_like / p / overlay: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_SnTe_like_p_overlay_sigma_C_sigma0_test.png`
- broad::SnTe_like / p / log_ratio: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_SnTe_like_p_log_ratio_sigma0ref_over_oldCT_test.png`
- broad::SnTe_like / n / two_panel: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_SnTe_like_n_sigma_C_and_sigma0_two_panel_test.png`
- broad::SnTe_like / n / overlay: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_SnTe_like_n_overlay_sigma_C_sigma0_test.png`
- broad::PbTe_like / p / two_panel: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_PbTe_like_p_sigma_C_and_sigma0_two_panel_test.png`
- broad::PbTe_like / p / overlay: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_PbTe_like_p_overlay_sigma_C_sigma0_test.png`
- broad::PbTe_like / p / log_ratio: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_PbTe_like_p_log_ratio_sigma0ref_over_oldCT_test.png`
- broad::PbTe_like / n / two_panel: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_PbTe_like_n_sigma_C_and_sigma0_two_panel_test.png`
- broad::PbTe_like / n / overlay: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_PbTe_like_n_overlay_sigma_C_sigma0_test.png`
- broad::PbTe_like / n / log_ratio: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_PbTe_like_n_log_ratio_sigma0ref_over_oldCT_test.png`
- broad::BiTe_like / p / two_panel: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_BiTe_like_p_sigma_C_and_sigma0_two_panel_test.png`
- broad::BiTe_like / p / overlay: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_BiTe_like_p_overlay_sigma_C_sigma0_test.png`
- broad::BiTe_like / p / log_ratio: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_BiTe_like_p_log_ratio_sigma0ref_over_oldCT_test.png`
- broad::BiTe_like / n / two_panel: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_BiTe_like_n_sigma_C_and_sigma0_two_panel_test.png`
- broad::BiTe_like / n / overlay: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_BiTe_like_n_overlay_sigma_C_sigma0_test.png`
- broad::BiTe_like / n / log_ratio: `experiments\exp006\figures\focus_ct_sigma0_temperature\broad_BiTe_like_n_log_ratio_sigma0ref_over_oldCT_test.png`

## Data Availability
- broad::SnTe_like / p: sigma_rows=2050, old_ct=1121, sigma0_ref=10, warning=
- broad::SnTe_like / n: sigma_rows=19, old_ct=18, sigma0_ref=0, warning=no_current_sigma0_ref;no_curve_comparison
- broad::PbTe_like / p: sigma_rows=1364, old_ct=874, sigma0_ref=9, warning=
- broad::PbTe_like / n: sigma_rows=986, old_ct=1666, sigma0_ref=8, warning=
- broad::BiTe_like / p: sigma_rows=2801, old_ct=747, sigma0_ref=9, warning=
- broad::BiTe_like / n: sigma_rows=6006, old_ct=1061, sigma0_ref=9, warning=

## Missing Old C(T)
- none

## Missing Current Sigma0 Ref
- broad::SnTe_like / n

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

## Ratio Summary
- count: 5469
- median log10(sigma0_ref / old_C_T): 0.0203186
- min/max log10 ratio: -1.3719 / 0.939

## Notes
- Old C(T) is an observed-sigma empirical baseline.
- The current coefficient is corrected using Seebeck-derived Fermi-level information.
- Both have units of S/m but different meanings.
- No new sigma_pred is calculated.
- Step4 full-data reference curves are not used.
- Starrydata2 raw data are not read.

## Next Checks
- Compare temperature trends for SnTe_like, PbTe_like, BiTe_like, and SiGe_like.
- Check whether p/n carrier type changes the offset or slope.
- Identify material groups where sigma0_ref deviates strongly from old C(T).
