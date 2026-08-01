# Old C(T) vs Sigma0_ref(T) Only From Script

This is a separate confirmation analysis, not a numbered exp006 Step.

This analysis reads the old C(T) source script, `experiments/exp005/fit_tau_eff_step12.py`, detects the CSV output that stores the old C(T) curve, then overlays only:

- old SS2026 `C(T)`, detected from the Step12 output CSV
- current broad-family `sigma0_ref(T)`, from train-only Step5B reference curve bins

It does not plot measured sigma scatter points.

The two curves share units of S/m but have different meanings:

- Old `C(T)` is an empirical baseline against measured electrical conductivity.
- `sigma0_ref(T)` is a Seebeck-derived coefficient after Fermi-level correction.

The overlay is for checking temperature-trend shape and scale differences, not for treating the two as identical physical quantities.

Example test run:

```powershell
python experiments/exp006/build_focus_ct_vs_sigma0ref_only_from_script.py `
  --old-ct-script experiments/exp005/fit_tau_eff_step12.py `
  --current-sigma0-ref experiments/exp006/data/processed/step6b_broad_family/step5b_train_reference_curve_bins.parquet `
  --target-groups broad::SnTe_like broad::PbTe_like broad::BiTe_like broad::SbTe_like broad::SiGe_like broad::oxide broad::sulfide `
  --output experiments/exp006/data/processed/focus_ct_vs_sigma0ref_only_from_script `
  --figures experiments/exp006/figures/focus_ct_vs_sigma0ref_only_from_script `
  --report experiments/exp006/reports/focus_ct_vs_sigma0ref_only_from_script/focus_ct_vs_sigma0ref_only_from_script_report_test.md `
  --max-groups 3 `
  --output-suffix _test
```

Then check outputs:

```powershell
python experiments/exp006/check_focus_ct_vs_sigma0ref_only_from_script_outputs.py `
  --summary experiments/exp006/data/processed/focus_ct_vs_sigma0ref_only_from_script/focus_ct_vs_sigma0ref_summary_by_group_carrier_test.csv `
  --figure-index experiments/exp006/data/processed/focus_ct_vs_sigma0ref_only_from_script/focus_ct_vs_sigma0ref_figure_index_test.csv `
  --script-parse-summary experiments/exp006/data/processed/focus_ct_vs_sigma0ref_only_from_script/focus_ct_vs_sigma0ref_fit_tau_script_parse_summary_test.csv `
  --report experiments/exp006/reports/focus_ct_vs_sigma0ref_only_from_script/focus_ct_vs_sigma0ref_only_from_script_report_test.md
```

The script does not rerun Step0 to Step7C, does not rerun `fit_tau_eff_step12.py`, does not read Starrydata2 raw data, does not use Step4 full-data reference curves, and does not calculate new `sigma_pred`.
