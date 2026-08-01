# Focus C(T) vs Sigma0 Temperature Comparison

This is a separate confirmation analysis, not a new numbered Step in the existing exp006 pipeline.

The script compares two same-unit but physically different quantities:

- Old `C(T)` from `experiments/exp005/fit_tau_eff_step12.py`, stored as Step12 `prefactor_C_S_per_m_step12`. This is an empirical baseline made from observed electrical conductivity for tau_eff fitting.
- Current `sigma0_ref_S_per_m` from the broad-family train-only reference curves. This coefficient is derived after correcting measured conductivity with Fermi-level information inferred from measured Seebeck coefficient.

Because these quantities have different meanings, the main figure is a two-panel plot:

- Top: measured `sigma_exp(T)` and old empirical `C(T)`.
- Bottom: Seebeck-derived `sigma0_S_per_m` points and train-only `sigma0_ref(T)`.

The overlay plot is provided only as a same-unit visual comparison. It should not be interpreted as showing equivalent physical quantities.

Example test run:

```powershell
python experiments/exp006/build_focus_ct_sigma0_temperature_plots.py `
  --current-rows experiments/exp006/data/processed/step6a_validation_rows_with_splits_key_broad_family.parquet `
  --current-sigma0-ref experiments/exp006/data/processed/step6b_broad_family/step5b_train_reference_curve_bins.parquet `
  --old-ct-script experiments/exp005/fit_tau_eff_step12.py `
  --target-groups broad::SnTe_like broad::PbTe_like broad::BiTe_like broad::SbTe_like broad::SiGe_like broad::oxide broad::sulfide `
  --output experiments/exp006/data/processed/focus_ct_sigma0_temperature `
  --figures experiments/exp006/figures/focus_ct_sigma0_temperature `
  --report experiments/exp006/reports/focus_ct_sigma0_temperature/focus_ct_sigma0_temperature_report_test.md `
  --max-groups 3 `
  --output-suffix _test
```

Then check outputs:

```powershell
python experiments/exp006/check_focus_ct_sigma0_temperature_outputs.py `
  --summary experiments/exp006/data/processed/focus_ct_sigma0_temperature/focus_ct_sigma0_summary_by_group_carrier_test.csv `
  --figure-index experiments/exp006/data/processed/focus_ct_sigma0_temperature/focus_ct_sigma0_figure_index_test.csv `
  --report experiments/exp006/reports/focus_ct_sigma0_temperature/focus_ct_sigma0_temperature_report_test.md
```

The script reads existing processed outputs only. It does not rerun Step0 to Step7C, does not read Starrydata2 raw data, does not load Step4 full-data reference curves, and does not calculate new `sigma_pred`.
