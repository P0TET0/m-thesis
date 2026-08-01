# Step7 Manual Review Workflow

Step7 prepares and consumes human review decisions for broad_family outliers. Step7A creates the review packet only; it does not exclude rows, recompute predictions, create figures, read raw Starrydata2 files, or use Step4 full-data reference curves.

## Step7A

Small test:

```powershell
python experiments/exp006/build_step7a_manual_review_packet.py `
  --step6d-dir experiments/exp006/data/processed/step6d_broad_family_audit `
  --step6c-dir experiments/exp006/data/processed/step6c_broad_family `
  --step6b-dir experiments/exp006/data/processed/step6b_broad_family `
  --metadata-input experiments/exp006/data/processed/step6a_validation_rows_with_splits_key_broad_family.parquet `
  --step3-input experiments/exp006/data/processed/step3_sigma0_valid.parquet `
  --step0-input data/processed/step0_te_analysis_base.parquet `
  --output experiments/exp006/data/processed/step7a_manual_review_packet `
  --report experiments/exp006/reports/step7a_manual_review_packet/step7a_manual_review_packet_report_test.md `
  --max-row-cases 50 `
  --max-sample-cases 30 `
  --max-paper-cases 30 `
  --casebook-top-n 20 `
  --output-suffix _test
```

```powershell
python experiments/exp006/check_step7a_manual_review_packet.py `
  --output experiments/exp006/data/processed/step7a_manual_review_packet `
  --master experiments/exp006/data/processed/step7a_manual_review_packet/step7a_manual_review_master_test.csv `
  --decision-template experiments/exp006/data/processed/step7a_manual_review_packet/step7a_review_decisions_template_test.csv `
  --source-trace experiments/exp006/data/processed/step7a_manual_review_packet/step7a_source_traceability_table_test.csv `
  --casebook experiments/exp006/data/processed/step7a_manual_review_packet/step7a_manual_review_casebook_test.md `
  --packet-index experiments/exp006/data/processed/step7a_manual_review_packet/step7a_review_packet_index_test.csv `
  --report experiments/exp006/reports/step7a_manual_review_packet/step7a_manual_review_packet_report_test.md
```

Full run:

```powershell
python experiments/exp006/build_step7a_manual_review_packet.py `
  --step6d-dir experiments/exp006/data/processed/step6d_broad_family_audit `
  --step6c-dir experiments/exp006/data/processed/step6c_broad_family `
  --step6b-dir experiments/exp006/data/processed/step6b_broad_family `
  --metadata-input experiments/exp006/data/processed/step6a_validation_rows_with_splits_key_broad_family.parquet `
  --step3-input experiments/exp006/data/processed/step3_sigma0_valid.parquet `
  --step0-input data/processed/step0_te_analysis_base.parquet `
  --output experiments/exp006/data/processed/step7a_manual_review_packet `
  --report experiments/exp006/reports/step7a_manual_review_packet/step7a_manual_review_packet_report.md `
  --max-row-cases 200 `
  --max-sample-cases 100 `
  --max-paper-cases 100 `
  --casebook-top-n 50
```

```powershell
python experiments/exp006/check_step7a_manual_review_packet.py `
  --output experiments/exp006/data/processed/step7a_manual_review_packet `
  --master experiments/exp006/data/processed/step7a_manual_review_packet/step7a_manual_review_master.csv `
  --decision-template experiments/exp006/data/processed/step7a_manual_review_packet/step7a_review_decisions_template.csv `
  --source-trace experiments/exp006/data/processed/step7a_manual_review_packet/step7a_source_traceability_table.csv `
  --casebook experiments/exp006/data/processed/step7a_manual_review_packet/step7a_manual_review_casebook.md `
  --packet-index experiments/exp006/data/processed/step7a_manual_review_packet/step7a_review_packet_index.csv `
  --report experiments/exp006/reports/step7a_manual_review_packet/step7a_manual_review_packet_report.md
```

Step7A writes review CSVs to `experiments/exp006/data/processed/step7a_manual_review_packet/` and the report to `experiments/exp006/reports/step7a_manual_review_packet/`.

The human-editable file is `step7a_review_decisions_template.csv`. Step7B should read the completed template and create primary/sensitivity flags from the review decisions.

## Step7B

Step7B applies the Step7A decision template to existing Step6B broad_family prediction rows. It does not recompute predictions, rerun Step5B/Step5C, read raw Starrydata2 data, use Step4 full-data reference curves, or create figures.

Small test:

```powershell
python experiments/exp006/build_step7b_apply_review_decisions.py `
  --predictions experiments/exp006/data/processed/step6b_broad_family/step5b_test_predictions_valid.parquet `
  --decision-template experiments/exp006/data/processed/step7a_manual_review_packet/step7a_review_decisions_template.csv `
  --review-master experiments/exp006/data/processed/step7a_manual_review_packet/step7a_manual_review_master.csv `
  --source-trace experiments/exp006/data/processed/step7a_manual_review_packet/step7a_source_traceability_table.csv `
  --output experiments/exp006/data/processed/step7b_review_applied `
  --report experiments/exp006/reports/step7b_review_applied/step7b_review_application_report_test.md `
  --pending-policy keep_with_pending_flag `
  --suspect-policy exclude_primary_keep_sensitivity `
  --max-rows-per-config 2000 `
  --output-suffix _test
```

```powershell
python experiments/exp006/check_step7b_review_applied.py `
  --output experiments/exp006/data/processed/step7b_review_applied `
  --rows experiments/exp006/data/processed/step7b_review_applied/step7b_prediction_rows_with_review_flags_test.csv `
  --primary experiments/exp006/data/processed/step7b_review_applied/step7b_primary_analysis_predictions_test.csv `
  --sensitivity experiments/exp006/data/processed/step7b_review_applied/step7b_sensitivity_analysis_predictions_test.csv `
  --metrics experiments/exp006/data/processed/step7b_review_applied/step7b_metrics_by_review_scenario_config_test.csv `
  --default-metrics experiments/exp006/data/processed/step7b_review_applied/step7b_default_metrics_by_review_scenario_test.csv `
  --readiness experiments/exp006/data/processed/step7b_review_applied/step7b_review_readiness_summary_test.csv `
  --report experiments/exp006/reports/step7b_review_applied/step7b_review_application_report_test.md
```

Full run:

```powershell
python experiments/exp006/build_step7b_apply_review_decisions.py `
  --predictions experiments/exp006/data/processed/step6b_broad_family/step5b_test_predictions_valid.parquet `
  --decision-template experiments/exp006/data/processed/step7a_manual_review_packet/step7a_review_decisions_template.csv `
  --review-master experiments/exp006/data/processed/step7a_manual_review_packet/step7a_manual_review_master.csv `
  --source-trace experiments/exp006/data/processed/step7a_manual_review_packet/step7a_source_traceability_table.csv `
  --output experiments/exp006/data/processed/step7b_review_applied `
  --report experiments/exp006/reports/step7b_review_applied/step7b_review_application_report.md `
  --pending-policy keep_with_pending_flag `
  --suspect-policy exclude_primary_keep_sensitivity
```

```powershell
python experiments/exp006/check_step7b_review_applied.py `
  --output experiments/exp006/data/processed/step7b_review_applied `
  --rows experiments/exp006/data/processed/step7b_review_applied/step7b_prediction_rows_with_review_flags.csv `
  --primary experiments/exp006/data/processed/step7b_review_applied/step7b_primary_analysis_predictions.csv `
  --sensitivity experiments/exp006/data/processed/step7b_review_applied/step7b_sensitivity_analysis_predictions.csv `
  --metrics experiments/exp006/data/processed/step7b_review_applied/step7b_metrics_by_review_scenario_config.csv `
  --default-metrics experiments/exp006/data/processed/step7b_review_applied/step7b_default_metrics_by_review_scenario.csv `
  --readiness experiments/exp006/data/processed/step7b_review_applied/step7b_review_readiness_summary.csv `
  --report experiments/exp006/reports/step7b_review_applied/step7b_review_application_report.md
```

Default policy keeps pending/unresolved decisions in primary and sensitivity while marking `review_is_pending = True`. Use `--pending-policy fail_if_pending` only after human review is complete.

## Step7C

Step7C applies the human-reviewed Step7A decision file. The required input is `experiments/exp006/data/processed/step7a_manual_review_packet/step7a_review_decisions_template_reviewed.csv`; Step7C intentionally does not fall back to the original pending template. It validates decision values, checks whether extreme cases have human review evidence, reruns Step7B under keep-pending and exclude-pending-primary policies, compares both policies with the previous Step7B all-pending baseline, and writes a final candidate dataset manifest.

Small test:

```powershell
python experiments/exp006/run_step7c_reviewed_decisions.py `
  --predictions experiments/exp006/data/processed/step6b_broad_family/step5b_test_predictions_valid.parquet `
  --reviewed-decision-template experiments/exp006/data/processed/step7a_manual_review_packet/step7a_review_decisions_template_reviewed.csv `
  --review-master experiments/exp006/data/processed/step7a_manual_review_packet/step7a_manual_review_master.csv `
  --source-trace experiments/exp006/data/processed/step7a_manual_review_packet/step7a_source_traceability_table.csv `
  --previous-step7b-dir experiments/exp006/data/processed/step7b_review_applied `
  --output experiments/exp006/data/processed/step7c_reviewed_decisions_applied `
  --report-dir experiments/exp006/reports/step7c_reviewed_decisions_applied `
  --max-rows-per-config 2000 `
  --output-suffix _test
```

```powershell
python experiments/exp006/check_step7c_reviewed_decisions.py `
  --output experiments/exp006/data/processed/step7c_reviewed_decisions_applied `
  --report-dir experiments/exp006/reports/step7c_reviewed_decisions_applied `
  --output-suffix _test `
  --allow-no-human-review
```

Full run:

```powershell
python experiments/exp006/run_step7c_reviewed_decisions.py `
  --predictions experiments/exp006/data/processed/step6b_broad_family/step5b_test_predictions_valid.parquet `
  --reviewed-decision-template experiments/exp006/data/processed/step7a_manual_review_packet/step7a_review_decisions_template_reviewed.csv `
  --review-master experiments/exp006/data/processed/step7a_manual_review_packet/step7a_manual_review_master.csv `
  --source-trace experiments/exp006/data/processed/step7a_manual_review_packet/step7a_source_traceability_table.csv `
  --previous-step7b-dir experiments/exp006/data/processed/step7b_review_applied `
  --output experiments/exp006/data/processed/step7c_reviewed_decisions_applied `
  --report-dir experiments/exp006/reports/step7c_reviewed_decisions_applied
```

```powershell
python experiments/exp006/check_step7c_reviewed_decisions.py `
  --output experiments/exp006/data/processed/step7c_reviewed_decisions_applied `
  --report-dir experiments/exp006/reports/step7c_reviewed_decisions_applied
```

Step7C writes final candidate dataset manifests and comparison tables to `experiments/exp006/data/processed/step7c_reviewed_decisions_applied/` and the final report to `experiments/exp006/reports/step7c_reviewed_decisions_applied/`.

Step7C does not compute new `sigma_pred_S_per_m`, does not read raw Starrydata2 data, does not rerun Step5/Step6, and does not create figures.
