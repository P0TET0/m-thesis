# Step9B saved sigma_pred vs Step31 C(T)

This replaces the former sigma0_ref points in the ct-vs-pred-style supplementary figures with already-saved Step9A holdout/test sigma_pred rows.
Before replacement, the colored points were the train-derived 25 K sigma0_ref reference coefficient (before multiplying by row-specific F0_eta), compared with Step31's carrier-specific median measured conductivity C(T).

- prediction input: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9a_25k_bin_broad_family\step5b_test_predictions_valid.parquet`
- config_id: `sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median`
- Step31 C(T) input: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_step31_ct_curves_normalized.csv`
- target material groups: 3
- figures created: 6 PNG + 6 PDF
- saved sigma_pred rows plotted/matched: 1473
- line-only combinations: 1
- median log10(sigma_pred / Step31 C(T)): 0.437637
- x-axis policy: every figure includes 0-1300 K; data above 1300 K remain visible.
- figure index: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_figure_index_ct_vs_pred_style_test.csv`
- nearest comparison: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_pred_vs_step31_ct_comparison_test.csv`
- summary: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_pred_vs_step31_ct_summary_test.csv`
- figure directory: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\ct_vs_pred_style`

No sigma_pred was recalculated, and no Step9A or Step31 input was modified.

## Line-only combinations

- broad::SnTe_like, n: no saved sigma_pred rows
