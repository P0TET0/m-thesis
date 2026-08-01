# exp006

Research code and generated intermediate files for the thermoelectric study targeting the 2026 meeting.

All implementation for this experiment lives under `experiments/exp006`. Generated tables are written to `experiments/exp006/data/processed/`, reports to `experiments/exp006/reports/`, and figures to `experiments/exp006/figures/`.

## Scripts

| Path | Purpose |
| --- | --- |
| `build_step0_table.py` | Build Step0 analysis table from Starrydata2 table exports |
| `check_step0_outputs.py` | Check Step0 outputs |
| `build_step1_carrier_table.py` | Add p/n/unknown carrier labels |
| `check_step1_outputs.py` | Check Step1 outputs |
| `build_step2_eta_lookup.py` | Build Step2A eta lookup table |
| `check_step2_eta_lookup.py` | Check Step2A lookup table |
| `build_step2_eta_table.py` | Assign eta from Step2A lookup |
| `check_step2_eta_outputs.py` | Check Step2B outputs |
| `build_step3_sigma0_table.py` | Compute sigma0 for eta >= 1 rows |
| `check_step3_sigma0_outputs.py` | Check Step3 outputs |
| `build_step4_sigma0_reference_curves.py` | Build full-data sigma0 reference bins |
| `check_step4_sigma0_reference_curves.py` | Check Step4 outputs |
| `build_step5a_validation_splits.py` | Build independent validation splits |
| `check_step5a_validation_splits.py` | Check Step5A outputs |
| `build_step5b_assign_predictions.py` | Assign train-only sigma predictions to test rows |
| `check_step5b_predictions.py` | Check Step5B outputs |
| `build_step5c_evaluation_metrics.py` | Aggregate prediction accuracy metrics |
| `check_step5c_evaluation_metrics.py` | Check Step5C outputs |
| `build_step5d_visual_diagnostics.py` | Build major figures and diagnostics |
| `check_step5d_visual_diagnostics.py` | Check Step5D-1 outputs |
| `build_step6a_material_group_keys.py` | Build repaired material group key candidates |
| `check_step6a_material_group_keys.py` | Check Step6A outputs |
| `run_step6b_broad_family_revalidation.py` | Rerun Step5B/5C using Step6A broad_family keys |
| `check_step6b_broad_family_revalidation.py` | Check Step6B broad_family revalidation |
| `build_step6c_broad_family_visualization.py` | Build Step6C broad_family visual diagnostics |
| `check_step6c_broad_family_visualization.py` | Check Step6C broad_family visual diagnostics |
| `build_step6d_outlier_robustness_audit.py` | Build Step6D outlier and robustness audit tables |
| `check_step6d_outlier_robustness_audit.py` | Check Step6D outlier and robustness audit outputs |
| `build_step7a_manual_review_packet.py` | Build Step7A manual review packet |
| `check_step7a_manual_review_packet.py` | Check Step7A manual review packet |
| `build_step7b_apply_review_decisions.py` | Apply Step7A review decisions to prediction rows |
| `check_step7b_review_applied.py` | Check Step7B review-applied outputs |
| `run_step7c_reviewed_decisions.py` | Apply human-reviewed Step7A decisions through Step7B policies |
| `check_step7c_reviewed_decisions.py` | Check Step7C reviewed-decision outputs |

## Step READMEs

```text
README_step0.md
README_step1.md
README_step2.md
README_step3.md
README_step4.md
README_step5.md
README_step6.md
README_step7.md
```

## Current Scope

Step6B reruns Step5B/5C with the Step6A `broad_family` variant in a dedicated output directory, then compares material-family and global predictions plus original-vs-broad-family metrics. Step6C visualizes those Step6B outputs and writes broad-family diagnostic figures, comparison tables, and a report. Step6D audits outliers and robustness using existing Step6B/Step6C outputs. Step7A packages the Step6D audit into human-readable review tables and a decision template. Step7B applies the completed or still-pending review decisions to prediction rows and rebuilds review-scenario metrics. Step7C requires the human-reviewed decision file, applies it under keep-pending and exclude-pending-primary policies, compares the results with the previous Step7B baseline, and writes the final candidate dataset manifest.

Step6B, Step6C, Step6D, Step7A, Step7B, and Step7C do not read raw data. Step6C/Step6D/Step7A/Step7B/Step7C do not rerun Step5B/Step5C and do not use Step4 full-data reference curves. Step7C does not compute new prediction values or create figures.
