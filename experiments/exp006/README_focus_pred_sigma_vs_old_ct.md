# Predicted Sigma vs Old C(T)

This is a separate confirmation analysis, not a numbered exp006 Step.

This analysis compares current broad-family predicted electrical conductivity points with the SS2026 old C(T) curve by material group and carrier type:

- Points: current `sigma_pred_S_per_m` from `step5b_test_predictions_valid`
- Line: old `C(T)` detected from `experiments/exp005/fit_tau_eff_step12.py` outputs

The main figures do not include experimental sigma points and do not include `sigma0_ref(T)`.

Example test run:

```powershell
python experiments/exp006/build_focus_pred_sigma_vs_old_ct.py `
  --predictions experiments/exp006/data/processed/step6b_broad_family/step5b_test_predictions_valid.parquet `
  --old-ct-script experiments/exp005/fit_tau_eff_step12.py `
  --config-id sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median `
  --target-groups broad::SnTe_like broad::PbTe_like broad::BiTe_like broad::SbTe_like broad::SiGe_like broad::oxide broad::sulfide `
  --output experiments/exp006/data/processed/focus_pred_sigma_vs_old_ct `
  --figures experiments/exp006/figures/focus_pred_sigma_vs_old_ct `
  --report experiments/exp006/reports/focus_pred_sigma_vs_old_ct/focus_pred_sigma_vs_old_ct_report_test.md `
  --max-groups 3 `
  --max-rows-per-group 2000 `
  --output-suffix _test
```

Then check outputs:

```powershell
python experiments/exp006/check_focus_pred_sigma_vs_old_ct_outputs.py `
  --summary experiments/exp006/data/processed/focus_pred_sigma_vs_old_ct/focus_pred_sigma_vs_old_ct_summary_by_group_carrier_test.csv `
  --figure-index experiments/exp006/data/processed/focus_pred_sigma_vs_old_ct/focus_pred_sigma_vs_old_ct_figure_index_test.csv `
  --report experiments/exp006/reports/focus_pred_sigma_vs_old_ct/focus_pred_sigma_vs_old_ct_report_test.md
```

The script reads existing processed outputs only. It does not rerun Step0 to Step7C, does not read Starrydata2 raw data, does not use Step4 full-data reference curves, and does not calculate new `sigma_pred`.
