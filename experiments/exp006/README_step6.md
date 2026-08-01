# Step6 Material Group Key Rebuild

Step6A diagnoses the collapsed `material_group_key` found in Step5D-1 and creates replacement key candidates for rerunning Step5B.

This step does not read Starrydata2 raw data, does not recompute `sigma_pred`, and does not aggregate Step5C metrics.

## Step6A

Small test:

```powershell
python experiments/exp006/build_step6a_material_group_keys.py `
  --input experiments/exp006/data/processed/step5a_validation_rows_with_splits.parquet `
  --step3 experiments/exp006/data/processed/step3_sigma0_valid.parquet `
  --step0 data/processed/step0_te_analysis_base.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step6a_material_group_key_rebuild_report_test.md `
  --min-rows-per-material-group 30 `
  --min-samples-per-material-group 3 `
  --min-rows-per-bin 3 `
  --min-samples-per-bin 3 `
  --min-papers-per-bin 1 `
  --max-rows 5000 `
  --output-suffix _test
```

```powershell
python experiments/exp006/check_step6a_material_group_keys.py `
  --candidate-rows experiments/exp006/data/processed/step6a_material_group_candidate_rows_test.csv `
  --summary experiments/exp006/data/processed/step6a_material_group_key_summary_test.csv `
  --preflight experiments/exp006/data/processed/step6a_material_group_key_preflight_coverage_test.csv `
  --recommended experiments/exp006/data/processed/step6a_recommended_material_key_variants_test.csv `
  --report experiments/exp006/reports/step6a_material_group_key_rebuild_report_test.md
```

Full run:

```powershell
python experiments/exp006/build_step6a_material_group_keys.py `
  --input experiments/exp006/data/processed/step5a_validation_rows_with_splits.parquet `
  --step3 experiments/exp006/data/processed/step3_sigma0_valid.parquet `
  --step0 data/processed/step0_te_analysis_base.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step6a_material_group_key_rebuild_report.md `
  --min-rows-per-material-group 30 `
  --min-samples-per-material-group 3 `
  --min-rows-per-bin 3 `
  --min-samples-per-bin 3 `
  --min-papers-per-bin 1
```

```powershell
python experiments/exp006/check_step6a_material_group_keys.py `
  --candidate-rows experiments/exp006/data/processed/step6a_material_group_candidate_rows.csv `
  --summary experiments/exp006/data/processed/step6a_material_group_key_summary.csv `
  --preflight experiments/exp006/data/processed/step6a_material_group_key_preflight_coverage.csv `
  --recommended experiments/exp006/data/processed/step6a_recommended_material_key_variants.csv `
  --report experiments/exp006/reports/step6a_material_group_key_rebuild_report.md
```

## Outputs

```text
experiments/exp006/data/processed/step6a_material_group_candidate_rows.csv
experiments/exp006/data/processed/step6a_material_group_candidate_rows.parquet
experiments/exp006/data/processed/step6a_material_group_key_summary.csv
experiments/exp006/data/processed/step6a_material_group_key_counts.csv
experiments/exp006/data/processed/step6a_formula_parse_failures.csv
experiments/exp006/data/processed/step6a_ambiguous_material_group_examples.csv
experiments/exp006/data/processed/step6a_recommended_material_key_variants.csv
experiments/exp006/data/processed/step6a_material_group_key_preflight_coverage.csv
experiments/exp006/reports/step6a_material_group_key_rebuild_report.md
```

Step5B-ready variant files are also written:

```text
step6a_validation_rows_with_splits_key_formula_system.csv/parquet
step6a_validation_rows_with_splits_key_broad_family.csv/parquet
step6a_validation_rows_with_splits_key_hybrid_v1.csv/parquet
step6a_validation_rows_with_splits_key_hybrid_v2_broad_first.csv/parquet
step6a_validation_rows_with_splits_key_formula_system_collapsed.csv/parquet
step6a_validation_rows_with_splits_key_hybrid_v1_collapsed.csv/parquet
```

Next step: select one recommended variant and rerun Step5B with that variant file as `--input`.

## Step6B

Step6B reruns Step5B and Step5C using only the Step6A `broad_family` variant input. Outputs are isolated under `experiments/exp006/data/processed/step6b_broad_family/` so the original Step5B/Step5C outputs are not overwritten.

```powershell
python experiments/exp006/run_step6b_broad_family_revalidation.py `
  --input experiments/exp006/data/processed/step6a_validation_rows_with_splits_key_broad_family.parquet `
  --output experiments/exp006/data/processed/step6b_broad_family `
  --report-dir experiments/exp006/reports/step6b_broad_family `
  --min-rows-per-bin 3 `
  --min-samples-per-bin 3 `
  --min-papers-per-bin 1 `
  --min-eval-rows 30 `
  --min-eval-samples 5 `
  --max-rows 5000 `
  --max-rows-per-config 200
```

```powershell
python experiments/exp006/check_step6b_broad_family_revalidation.py `
  --output experiments/exp006/data/processed/step6b_broad_family `
  --report experiments/exp006/reports/step6b_broad_family/step6b_broad_family_revalidation_report.md
```

Step6B writes comparison tables:

```text
step6b_material_family_vs_global_prediction_diff_summary.csv
step6b_material_family_vs_global_prediction_diff_examples.csv
step6b_reference_group_diagnostics.csv
step6b_broad_family_vs_original_default_metrics_comparison.csv
step6b_broad_family_default_metrics_summary.csv
step6b_revalidation_summary.csv
```

## Step6C

Step6C visualizes and diagnoses the Step6B `broad_family` outputs. It reads only existing Step6B and original Step5C/Step5D summary outputs; it does not rerun Step5B/Step5C, does not read Starrydata2 raw data, and does not use Step4 full-data reference curves.

Small test:

```powershell
python experiments/exp006/build_step6c_broad_family_visualization.py `
  --step6b-dir experiments/exp006/data/processed/step6b_broad_family `
  --original-dir experiments/exp006/data/processed `
  --output experiments/exp006/data/processed/step6c_broad_family `
  --figures experiments/exp006/figures/step6c_broad_family `
  --report experiments/exp006/reports/step6c_broad_family/step6c_broad_family_visual_report_test.md `
  --max-rows-per-config 2000 `
  --plot-sample-size 5000 `
  --seed 20260618 `
  --output-suffix _test
```

```powershell
python experiments/exp006/check_step6c_broad_family_visualization.py `
  --figure-index experiments/exp006/data/processed/step6c_broad_family/step6c_figure_index_test.csv `
  --diagnostics-summary experiments/exp006/data/processed/step6c_broad_family/step6c_visual_diagnostics_summary_test.csv `
  --original-vs-broad experiments/exp006/data/processed/step6c_broad_family/step6c_original_vs_broad_metrics_summary_test.csv `
  --group-performance experiments/exp006/data/processed/step6c_broad_family/step6c_broad_family_group_performance_summary_test.csv `
  --largest-error-diagnostics experiments/exp006/data/processed/step6c_broad_family/step6c_broad_largest_error_diagnostics_top100_test.csv `
  --report experiments/exp006/reports/step6c_broad_family/step6c_broad_family_visual_report_test.md
```

Full run:

```powershell
python experiments/exp006/build_step6c_broad_family_visualization.py `
  --step6b-dir experiments/exp006/data/processed/step6b_broad_family `
  --original-dir experiments/exp006/data/processed `
  --output experiments/exp006/data/processed/step6c_broad_family `
  --figures experiments/exp006/figures/step6c_broad_family `
  --report experiments/exp006/reports/step6c_broad_family/step6c_broad_family_visual_report.md `
  --plot-sample-size 20000 `
  --seed 20260618
```

```powershell
python experiments/exp006/check_step6c_broad_family_visualization.py `
  --figure-index experiments/exp006/data/processed/step6c_broad_family/step6c_figure_index.csv `
  --diagnostics-summary experiments/exp006/data/processed/step6c_broad_family/step6c_visual_diagnostics_summary.csv `
  --original-vs-broad experiments/exp006/data/processed/step6c_broad_family/step6c_original_vs_broad_metrics_summary.csv `
  --group-performance experiments/exp006/data/processed/step6c_broad_family/step6c_broad_family_group_performance_summary.csv `
  --largest-error-diagnostics experiments/exp006/data/processed/step6c_broad_family/step6c_broad_largest_error_diagnostics_top100.csv `
  --report experiments/exp006/reports/step6c_broad_family/step6c_broad_family_visual_report.md
```

Step6C writes figures to `experiments/exp006/figures/step6c_broad_family/`, summary CSVs to `experiments/exp006/data/processed/step6c_broad_family/`, and the visual report to `experiments/exp006/reports/step6c_broad_family/`.

## Step6D

Step6D audits broad_family outliers and robustness using existing Step6B/Step6C outputs. It does not create figures, does not rerun Step5B/Step5C, does not read Starrydata2 raw data, and does not use Step4 full-data reference curves.

Small test:

```powershell
python experiments/exp006/build_step6d_outlier_robustness_audit.py `
  --step6b-dir experiments/exp006/data/processed/step6b_broad_family `
  --step6c-dir experiments/exp006/data/processed/step6c_broad_family `
  --original-dir experiments/exp006/data/processed `
  --metadata-input experiments/exp006/data/processed/step6a_validation_rows_with_splits_key_broad_family.parquet `
  --step3-input experiments/exp006/data/processed/step3_sigma0_valid.parquet `
  --step0-input data/processed/step0_te_analysis_base.parquet `
  --output experiments/exp006/data/processed/step6d_broad_family_audit `
  --report experiments/exp006/reports/step6d_broad_family_audit/step6d_outlier_robustness_audit_report_test.md `
  --max-rows-per-config 2000 `
  --top-n-outliers 200 `
  --seed 20260618 `
  --output-suffix _test
```

```powershell
python experiments/exp006/check_step6d_outlier_robustness_audit.py `
  --output experiments/exp006/data/processed/step6d_broad_family_audit `
  --outliers experiments/exp006/data/processed/step6d_broad_family_audit/step6d_outlier_rows_topN_test.csv `
  --robust-filter experiments/exp006/data/processed/step6d_broad_family_audit/step6d_robust_metrics_by_filter_test.csv `
  --robust-config experiments/exp006/data/processed/step6d_broad_family_audit/step6d_robust_metrics_by_config_test.csv `
  --manual-review experiments/exp006/data/processed/step6d_broad_family_audit/step6d_manual_review_shortlist_test.csv `
  --readiness experiments/exp006/data/processed/step6d_broad_family_audit/step6d_broad_family_main_result_readiness_summary_test.csv `
  --report experiments/exp006/reports/step6d_broad_family_audit/step6d_outlier_robustness_audit_report_test.md
```

Full run:

```powershell
python experiments/exp006/build_step6d_outlier_robustness_audit.py `
  --step6b-dir experiments/exp006/data/processed/step6b_broad_family `
  --step6c-dir experiments/exp006/data/processed/step6c_broad_family `
  --original-dir experiments/exp006/data/processed `
  --metadata-input experiments/exp006/data/processed/step6a_validation_rows_with_splits_key_broad_family.parquet `
  --step3-input experiments/exp006/data/processed/step3_sigma0_valid.parquet `
  --step0-input data/processed/step0_te_analysis_base.parquet `
  --output experiments/exp006/data/processed/step6d_broad_family_audit `
  --report experiments/exp006/reports/step6d_broad_family_audit/step6d_outlier_robustness_audit_report.md `
  --top-n-outliers 1000 `
  --seed 20260618
```

```powershell
python experiments/exp006/check_step6d_outlier_robustness_audit.py `
  --output experiments/exp006/data/processed/step6d_broad_family_audit `
  --outliers experiments/exp006/data/processed/step6d_broad_family_audit/step6d_outlier_rows_topN.csv `
  --robust-filter experiments/exp006/data/processed/step6d_broad_family_audit/step6d_robust_metrics_by_filter.csv `
  --robust-config experiments/exp006/data/processed/step6d_broad_family_audit/step6d_robust_metrics_by_config.csv `
  --manual-review experiments/exp006/data/processed/step6d_broad_family_audit/step6d_manual_review_shortlist.csv `
  --readiness experiments/exp006/data/processed/step6d_broad_family_audit/step6d_broad_family_main_result_readiness_summary.csv `
  --report experiments/exp006/reports/step6d_broad_family_audit/step6d_outlier_robustness_audit_report.md
```

Step6D writes audit tables to `experiments/exp006/data/processed/step6d_broad_family_audit/` and the report to `experiments/exp006/reports/step6d_broad_family_audit/`.
