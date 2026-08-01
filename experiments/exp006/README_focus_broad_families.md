# Focus Broad Families Check

This analysis is a separate confirmation pass outside the existing Step0-Step7C
sequence. It reads already generated broad_family prediction results, selects
material groups, and creates measured-versus-predicted electrical conductivity
plots for each group.

The script does not rerun Step0-Step7C, does not read Starrydata2 raw data, does
not use Step4 full-data reference curves, and does not calculate new
`sigma_pred` values.

Default config:

`sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median`

## Build

```bash
python experiments/exp006/build_focus_broad_families_scatter.py \
    --predictions experiments/exp006/data/processed/step6b_broad_family/step5b_test_predictions_valid.parquet \
    --performance-summary experiments/exp006/data/processed/step6c_broad_family/step6c_broad_family_group_performance_summary.csv \
    --config-id sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median \
    --selection-mode reliable_from_summary \
    --min-rows 30 \
    --min-samples 3 \
    --min-papers 1 \
    --exclude-groups broad::SiGe_like \
    --output experiments/exp006/data/processed/focus_broad_families \
    --figures experiments/exp006/figures/focus_broad_families \
    --report experiments/exp006/reports/focus_broad_families/focus_broad_families_report.md
```

## Check

```bash
python experiments/exp006/check_focus_broad_families_outputs.py \
    --selected-groups experiments/exp006/data/processed/focus_broad_families/focus_broad_families_selected_groups.csv \
    --rows experiments/exp006/data/processed/focus_broad_families/focus_broad_families_prediction_rows.csv \
    --metrics experiments/exp006/data/processed/focus_broad_families/focus_broad_families_metrics_summary.csv \
    --ranking experiments/exp006/data/processed/focus_broad_families/focus_broad_families_group_ranking.csv \
    --figure-index experiments/exp006/data/processed/focus_broad_families/focus_broad_families_figure_index.csv \
    --report experiments/exp006/reports/focus_broad_families/focus_broad_families_report.md
```

## Outputs

Tables are written to `experiments/exp006/data/processed/focus_broad_families/`.
Figures are written under
`experiments/exp006/figures/focus_broad_families/<safe_group_name>/`.
The report is written to
`experiments/exp006/reports/focus_broad_families/`.

The scatter plots are log-log plots. Points near `y=x` indicate close agreement,
points above `y=x` are overpredictions, and points below `y=x` are
underpredictions. Factor2 and factor10 guide lines show two-fold and ten-fold
error bands.
