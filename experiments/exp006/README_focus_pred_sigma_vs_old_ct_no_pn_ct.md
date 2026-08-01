# Predicted sigma vs no-p/n old C(T)

This analysis is a separate focus analysis outside the existing Step0-Step7C
pipeline. It compares the current broad_family predicted electrical
conductivity `sigma_pred` point cloud with the SS2026 old C(T) line aggregated
without p/n splitting, one figure per material group.

The old C(T) source script is:

```text
experiments/exp005/fit_tau_eff_step12.py
```

The script is read statically to identify the existing old C(T) output table.
It is not rerun. The expected old C(T) table and column are:

```text
data/output/starrydata2_step12_tau_fit/sigma_predictions_step12.csv
prefactor_C_S_per_m_step12
```

For this no-p/n analysis, the old C(T) curve is aggregated by
`material_group_key_mapped` and `T_K` only:

```text
old_C_T_S_per_m = median(prefactor_C_S_per_m_step12)
```

The old C(T) `n_or_p` column is ignored even when it exists. The plotted
prediction points may distinguish p and n by marker/color, but the old C(T)
line is one line per material group.

Main command:

```bash
python experiments/exp006/build_focus_pred_sigma_vs_old_ct_no_pn_ct.py \
  --predictions experiments/exp006/data/processed/step6b_broad_family/step5b_test_predictions_valid.parquet \
  --old-ct-script experiments/exp005/fit_tau_eff_step12.py \
  --config-id sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median \
  --target-groups broad::SnTe_like broad::PbTe_like broad::BiTe_like broad::SbTe_like broad::SiGe_like broad::oxide broad::sulfide \
  --output experiments/exp006/data/processed/focus_pred_sigma_vs_old_ct_no_pn_ct \
  --figures experiments/exp006/figures/focus_pred_sigma_vs_old_ct_no_pn_ct \
  --report experiments/exp006/reports/focus_pred_sigma_vs_old_ct_no_pn_ct/focus_pred_sigma_vs_old_ct_no_pn_ct_report.md
```

Check command:

```bash
python experiments/exp006/check_focus_pred_sigma_vs_old_ct_no_pn_ct_outputs.py \
  --summary experiments/exp006/data/processed/focus_pred_sigma_vs_old_ct_no_pn_ct/focus_pred_sigma_vs_old_ct_no_pn_summary_by_group.csv \
  --figure-index experiments/exp006/data/processed/focus_pred_sigma_vs_old_ct_no_pn_ct/focus_pred_sigma_vs_old_ct_no_pn_figure_index.csv \
  --report experiments/exp006/reports/focus_pred_sigma_vs_old_ct_no_pn_ct/focus_pred_sigma_vs_old_ct_no_pn_ct_report.md
```

This analysis does not calculate new `sigma_pred`, does not plot `sigma0_ref`,
does not include experimental sigma points in the main figures, does not use
Step4 full-data reference curves, and does not read Starrydata2 raw data.
