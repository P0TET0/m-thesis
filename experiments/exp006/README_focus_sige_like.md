# Focus SiGe_like Check

This analysis is a separate confirmation pass outside the existing Step0-Step7C sequence.
It reads the already generated broad_family prediction results, filters them to a target
material group, and plots measured electrical conductivity against predicted electrical
conductivity.

Default target:

- config: `sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median`
- material group: `broad::SiGe_like`

The script does not rerun Step0-Step7C, does not read Starrydata2 raw data, does not use
Step4 full-data reference curves, and does not calculate new `sigma_pred` values.

## Build

```bash
python experiments/exp006/build_focus_sige_like_scatter.py \
    --predictions experiments/exp006/data/processed/step6b_broad_family/step5b_test_predictions_valid.parquet \
    --target-material-group broad::SiGe_like \
    --config-id sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median \
    --output experiments/exp006/data/processed/focus_sige_like \
    --figures experiments/exp006/figures/focus_sige_like \
    --report experiments/exp006/reports/focus_sige_like/focus_sige_like_report.md
```

## Check

```bash
python experiments/exp006/check_focus_sige_like_outputs.py \
    --rows experiments/exp006/data/processed/focus_sige_like/focus_sige_like_prediction_rows.csv \
    --metrics experiments/exp006/data/processed/focus_sige_like/focus_sige_like_metrics_summary.csv \
    --figure-index experiments/exp006/data/processed/focus_sige_like/focus_sige_like_figure_index.csv \
    --report experiments/exp006/reports/focus_sige_like/focus_sige_like_report.md
```

## Outputs

Tables are written to `experiments/exp006/data/processed/focus_sige_like/`.
Figures are written to `experiments/exp006/figures/focus_sige_like/`.
The report is written to `experiments/exp006/reports/focus_sige_like/`.

The scatter plots are log-log plots. Points near `y=x` indicate close agreement,
points above `y=x` are overpredictions, and points below `y=x` are underpredictions.
