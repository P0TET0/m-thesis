# Step5 Validation

Step5 uses the Step4/Step5A processed tables under `experiments/exp006`.

Do not read Starrydata2 raw data in Step5. Do not rerun Step0-Step4 from these scripts.

## Step5A: Validation Splits

Step5A creates stable sample-holdout and paper-holdout train/test splits, CV fold IDs, and a coverage preflight.

Small test:

```powershell
python experiments/exp006/build_step5a_validation_splits.py `
  --input experiments/exp006/data/processed/step4_sigma0_binned_input_rows.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step5a_validation_split_report_test.md `
  --test-size 0.2 `
  --n-folds 5 `
  --seed 20260618 `
  --min-rows-per-bin 3 `
  --min-samples-per-bin 3 `
  --min-papers-per-bin 1 `
  --max-rows 5000 `
  --output-suffix _test
```

```powershell
python experiments/exp006/check_step5a_validation_splits.py `
  --rows experiments/exp006/data/processed/step5a_validation_rows_with_splits_test.csv `
  --sample-assignments experiments/exp006/data/processed/step5a_sample_group_split_assignments_test.csv `
  --paper-assignments experiments/exp006/data/processed/step5a_paper_group_split_assignments_test.csv `
  --summary experiments/exp006/data/processed/step5a_split_summary_test.csv `
  --coverage experiments/exp006/data/processed/step5a_holdout_coverage_preflight_test.csv `
  --dropped experiments/exp006/data/processed/step5a_dropped_rows_test.csv
```

Full run:

```powershell
python experiments/exp006/build_step5a_validation_splits.py `
  --input experiments/exp006/data/processed/step4_sigma0_binned_input_rows.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step5a_validation_split_report.md `
  --test-size 0.2 `
  --n-folds 5 `
  --seed 20260618 `
  --min-rows-per-bin 3 `
  --min-samples-per-bin 3 `
  --min-papers-per-bin 1
```

```powershell
python experiments/exp006/check_step5a_validation_splits.py `
  --rows experiments/exp006/data/processed/step5a_validation_rows_with_splits.csv `
  --sample-assignments experiments/exp006/data/processed/step5a_sample_group_split_assignments.csv `
  --paper-assignments experiments/exp006/data/processed/step5a_paper_group_split_assignments.csv `
  --summary experiments/exp006/data/processed/step5a_split_summary.csv `
  --coverage experiments/exp006/data/processed/step5a_holdout_coverage_preflight.csv `
  --dropped experiments/exp006/data/processed/step5a_dropped_rows.csv `
  --require-full-run
```

## Step5B: Assign Train-Only Predictions

Step5B reads `step5a_validation_rows_with_splits`, builds `sigma0_ref(T)` from train rows only, and assigns point-level predictions to test rows:

```text
sigma_pred_S_per_m = sigma0_ref_S_per_m * F0_eta
```

Step5B keeps test-row `sigma0_S_per_m` for diagnostics, but does not use it to compute predictions. Aggregate accuracy metrics such as MAE, RMSE, and factor accuracy are left for Step5C.

Small test:

```powershell
python experiments/exp006/build_step5b_assign_predictions.py `
  --input experiments/exp006/data/processed/step5a_validation_rows_with_splits.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step5b_prediction_assignment_report_test.md `
  --min-rows-per-bin 3 `
  --min-samples-per-bin 3 `
  --min-papers-per-bin 1 `
  --max-rows 5000 `
  --output-suffix _test
```

```powershell
python experiments/exp006/check_step5b_predictions.py `
  --predictions experiments/exp006/data/processed/step5b_test_predictions_test.csv `
  --valid experiments/exp006/data/processed/step5b_test_predictions_valid_test.csv `
  --coverage experiments/exp006/data/processed/step5b_prediction_coverage_by_config_test.csv `
  --reference experiments/exp006/data/processed/step5b_train_reference_curve_bins_test.csv `
  --dropped experiments/exp006/data/processed/step5b_dropped_rows_test.csv `
  --unavailable experiments/exp006/data/processed/step5b_test_predictions_unavailable_test.csv `
  --default experiments/exp006/data/processed/step5b_test_predictions_default_test.csv `
  --global-default experiments/exp006/data/processed/step5b_test_predictions_global_default_test.csv
```

Full run:

```powershell
python experiments/exp006/build_step5b_assign_predictions.py `
  --input experiments/exp006/data/processed/step5a_validation_rows_with_splits.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step5b_prediction_assignment_report.md `
  --min-rows-per-bin 3 `
  --min-samples-per-bin 3 `
  --min-papers-per-bin 1
```

```powershell
python experiments/exp006/check_step5b_predictions.py `
  --predictions experiments/exp006/data/processed/step5b_test_predictions.csv `
  --valid experiments/exp006/data/processed/step5b_test_predictions_valid.csv `
  --coverage experiments/exp006/data/processed/step5b_prediction_coverage_by_config.csv `
  --reference experiments/exp006/data/processed/step5b_train_reference_curve_bins.csv `
  --dropped experiments/exp006/data/processed/step5b_dropped_rows.csv `
  --unavailable experiments/exp006/data/processed/step5b_test_predictions_unavailable.csv `
  --default experiments/exp006/data/processed/step5b_test_predictions_default.csv `
  --global-default experiments/exp006/data/processed/step5b_test_predictions_global_default.csv `
  --require-full-run
```

## Step5B Outputs

```text
experiments/exp006/data/processed/step5b_train_reference_curve_bins.csv
experiments/exp006/data/processed/step5b_train_reference_curve_bins.parquet
experiments/exp006/data/processed/step5b_test_predictions.csv
experiments/exp006/data/processed/step5b_test_predictions.parquet
experiments/exp006/data/processed/step5b_test_predictions_valid.csv
experiments/exp006/data/processed/step5b_test_predictions_valid.parquet
experiments/exp006/data/processed/step5b_test_predictions_unavailable.csv
experiments/exp006/data/processed/step5b_test_predictions_default.csv
experiments/exp006/data/processed/step5b_test_predictions_default.parquet
experiments/exp006/data/processed/step5b_test_predictions_global_default.csv
experiments/exp006/data/processed/step5b_test_predictions_global_default.parquet
experiments/exp006/data/processed/step5b_prediction_coverage_by_config.csv
experiments/exp006/data/processed/step5b_prediction_unavailable_summary.csv
experiments/exp006/data/processed/step5b_dropped_rows.csv
experiments/exp006/reports/step5b_prediction_assignment_report.md
```

Default config:

```text
sample_holdout + conservative_valid reference + all_valid eval + material_family + sample_median
```

Global default config:

```text
sample_holdout + conservative_valid reference + all_valid eval + global + sample_median
```

## Step5C: Evaluation Metrics

Step5C reads Step5B valid prediction rows and aggregates prediction accuracy. It does not recompute predictions and does not create figures.

Main error:

```text
log10_sigma_pred_over_exp = log10(sigma_pred_S_per_m / sigma_S_per_m)
```

Small test:

```powershell
python experiments/exp006/build_step5c_evaluation_metrics.py `
  --input experiments/exp006/data/processed/step5b_test_predictions_valid.parquet `
  --coverage experiments/exp006/data/processed/step5b_prediction_coverage_by_config.csv `
  --unavailable experiments/exp006/data/processed/step5b_test_predictions_unavailable.csv `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step5c_evaluation_metrics_report_test.md `
  --min-eval-rows 30 `
  --min-eval-samples 5 `
  --max-rows-per-config 200 `
  --output-suffix _test
```

```powershell
python experiments/exp006/check_step5c_evaluation_metrics.py `
  --metrics-config experiments/exp006/data/processed/step5c_metrics_by_config_test.csv `
  --default-comparison experiments/exp006/data/processed/step5c_default_comparison_test.csv `
  --ranking experiments/exp006/data/processed/step5c_config_ranking_test.csv `
  --largest-errors experiments/exp006/data/processed/step5c_largest_abs_error_rows_test.csv `
  --dropped experiments/exp006/data/processed/step5c_dropped_rows_test.csv
```

Full run:

```powershell
python experiments/exp006/build_step5c_evaluation_metrics.py `
  --input experiments/exp006/data/processed/step5b_test_predictions_valid.parquet `
  --coverage experiments/exp006/data/processed/step5b_prediction_coverage_by_config.csv `
  --unavailable experiments/exp006/data/processed/step5b_test_predictions_unavailable.csv `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step5c_evaluation_metrics_report.md `
  --min-eval-rows 30 `
  --min-eval-samples 5
```

```powershell
python experiments/exp006/check_step5c_evaluation_metrics.py `
  --metrics-config experiments/exp006/data/processed/step5c_metrics_by_config.csv `
  --default-comparison experiments/exp006/data/processed/step5c_default_comparison.csv `
  --ranking experiments/exp006/data/processed/step5c_config_ranking.csv `
  --largest-errors experiments/exp006/data/processed/step5c_largest_abs_error_rows.csv `
  --dropped experiments/exp006/data/processed/step5c_dropped_rows.csv
```

## Step5C Outputs

```text
experiments/exp006/data/processed/step5c_metrics_by_config.csv
experiments/exp006/data/processed/step5c_metrics_by_config.parquet
experiments/exp006/data/processed/step5c_metrics_by_carrier_type.csv
experiments/exp006/data/processed/step5c_metrics_by_material_family.csv
experiments/exp006/data/processed/step5c_metrics_by_temperature_bin.csv
experiments/exp006/data/processed/step5c_metrics_by_eta_bin.csv
experiments/exp006/data/processed/step5c_metrics_by_reliability_level.csv
experiments/exp006/data/processed/step5c_metrics_by_sigma_source.csv
experiments/exp006/data/processed/step5c_metrics_by_match_method.csv
experiments/exp006/data/processed/step5c_default_comparison.csv
experiments/exp006/data/processed/step5c_config_ranking.csv
experiments/exp006/data/processed/step5c_largest_abs_error_rows.csv
experiments/exp006/data/processed/step5c_dropped_rows.csv
experiments/exp006/reports/step5c_evaluation_metrics_report.md
```

## Step5D-1: Visual Diagnostics

Step5D-1 uses Step5B prediction rows and Step5C metrics to create the main diagnostic figures and tables. It does not read raw data, does not recompute predictions, and does not use Step4 full-data reference curves.

Small test:

```powershell
python experiments/exp006/build_step5d_visual_diagnostics.py `
  --predictions-valid experiments/exp006/data/processed/step5b_test_predictions_valid.parquet `
  --predictions-all experiments/exp006/data/processed/step5b_test_predictions.csv `
  --reference-bins experiments/exp006/data/processed/step5b_train_reference_curve_bins.csv `
  --metrics-config experiments/exp006/data/processed/step5c_metrics_by_config.csv `
  --metrics-carrier experiments/exp006/data/processed/step5c_metrics_by_carrier_type.csv `
  --metrics-material experiments/exp006/data/processed/step5c_metrics_by_material_family.csv `
  --metrics-temperature experiments/exp006/data/processed/step5c_metrics_by_temperature_bin.csv `
  --metrics-eta experiments/exp006/data/processed/step5c_metrics_by_eta_bin.csv `
  --default-comparison experiments/exp006/data/processed/step5c_default_comparison.csv `
  --ranking experiments/exp006/data/processed/step5c_config_ranking.csv `
  --largest-errors experiments/exp006/data/processed/step5c_largest_abs_error_rows.csv `
  --output experiments/exp006/data/processed `
  --figures experiments/exp006/figures/step5d `
  --report experiments/exp006/reports/step5d_visual_diagnostics_report_test.md `
  --max-rows-per-config 2000 `
  --plot-sample-size 5000 `
  --seed 20260618 `
  --output-suffix _test
```

```powershell
python experiments/exp006/check_step5d_visual_diagnostics.py `
  --figure-index experiments/exp006/data/processed/step5d_figure_index_test.csv `
  --diagnostics-summary experiments/exp006/data/processed/step5d_visual_diagnostics_summary_test.csv `
  --diff-summary experiments/exp006/data/processed/step5d_global_vs_material_family_prediction_diff_summary_test.csv `
  --largest-error-diagnostics experiments/exp006/data/processed/step5d_largest_error_diagnostics_top100_test.csv `
  --report experiments/exp006/reports/step5d_visual_diagnostics_report_test.md
```

Full run:

```powershell
python experiments/exp006/build_step5d_visual_diagnostics.py `
  --predictions-valid experiments/exp006/data/processed/step5b_test_predictions_valid.parquet `
  --predictions-all experiments/exp006/data/processed/step5b_test_predictions.csv `
  --reference-bins experiments/exp006/data/processed/step5b_train_reference_curve_bins.csv `
  --metrics-config experiments/exp006/data/processed/step5c_metrics_by_config.csv `
  --metrics-carrier experiments/exp006/data/processed/step5c_metrics_by_carrier_type.csv `
  --metrics-material experiments/exp006/data/processed/step5c_metrics_by_material_family.csv `
  --metrics-temperature experiments/exp006/data/processed/step5c_metrics_by_temperature_bin.csv `
  --metrics-eta experiments/exp006/data/processed/step5c_metrics_by_eta_bin.csv `
  --default-comparison experiments/exp006/data/processed/step5c_default_comparison.csv `
  --ranking experiments/exp006/data/processed/step5c_config_ranking.csv `
  --largest-errors experiments/exp006/data/processed/step5c_largest_abs_error_rows.csv `
  --output experiments/exp006/data/processed `
  --figures experiments/exp006/figures/step5d `
  --report experiments/exp006/reports/step5d_visual_diagnostics_report.md `
  --plot-sample-size 20000 `
  --seed 20260618
```

```powershell
python experiments/exp006/check_step5d_visual_diagnostics.py `
  --figure-index experiments/exp006/data/processed/step5d_figure_index.csv `
  --diagnostics-summary experiments/exp006/data/processed/step5d_visual_diagnostics_summary.csv `
  --diff-summary experiments/exp006/data/processed/step5d_global_vs_material_family_prediction_diff_summary.csv `
  --largest-error-diagnostics experiments/exp006/data/processed/step5d_largest_error_diagnostics_top100.csv `
  --report experiments/exp006/reports/step5d_visual_diagnostics_report.md
```

Step5D-1 outputs figures under `experiments/exp006/figures/step5d/` and writes figure/table indexes under `experiments/exp006/data/processed/`.
