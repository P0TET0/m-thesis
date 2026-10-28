# Phase 8: robustness check

## Purpose and fixed conditions

Phase 7 concluded that oxide_v1 lowered paired conductivity-prediction error. Phase 8 checks whether that direction depends on a single bin width, support rule, weighting unit, extreme rows, or one oxide group. The sample holdout, train/evaluation subsets, source data, and frozen oxide_v1 rules remain unchanged.

## Model-setting sensitivity

- Both variants improved in MAE and RMSE in **39/39** model-setting × weighting comparisons.
- Under the frozen 100 K / 3-row / 3-sample / 1-paper / sample-median setting, row MAE changed from 1.023496 to 0.875409 (delta -0.148087); RMSE changed by -0.227671.
- Across the 13 row-equal settings, MAE delta ranged from -0.153509 to -0.121755; RMSE delta ranged from -0.240317 to -0.179311.
- The largest coverage penalty was -5.01 percentage points (bw050_strict_sample_median). Finer grouping can still lose predictions when a reference bin lacks support.

## Evaluation sensitivity and influence checks

- Row-, sample-, and paper-equal weighting plus 1%, 2.5%, and 5% symmetric extreme-row removal all retained lower oxide_v1 MAE and RMSE (6/6 checks).
- Omitting each oxide group in turn retained lower MAE and RMSE for both row- and sample-equal views (24/24 checks). Therefore the overall gain is not caused by one group alone.
- At group level across model settings: 7 groups were consistently improved, 4 mixed, and 1 consistently worsened.
- Group-level caveat: Cu-containing oxide, Fe/Ni/Cr-containing oxide, Mixed-anion O + As/P, Mixed-anion O + S/Se/Te, Other inorganic O-containing did not improve in every model setting; these should remain explicit in later discussion.

## Conclusion

**robust_with_coverage_and_group_specific_caveats**. The overall error reduction is stable to the predefined internal sensitivity tests. This does not constitute external validation: the same dataset and sample holdout are used throughout, and the coverage cost and group-specific exceptions remain material.

## Outputs

- `experiments/exp007/data/processed/phase8_model_sensitivity_metrics.csv`: model-setting and weighting metrics
- `experiments/exp007/data/processed/phase8_model_sensitivity_predictions.csv`: row-level scenario predictions and statuses
- `experiments/exp007/data/processed/phase8_evaluation_sensitivity.csv`: weighting and symmetric-outlier sensitivity
- `experiments/exp007/data/processed/phase8_leave_one_group_out.csv`: leave-one-group-out results
- `experiments/exp007/data/processed/phase8_group_direction_consistency.csv`: group-level direction consistency
- `experiments/exp007/data/processed/phase8_decision_summary.csv`: concise decision table
- `experiments/exp007/data/processed/phase8_validation_checks.csv` and `experiments/exp007/reports/phase8_validation_log.txt`: build validation evidence
- `experiments/exp007/figures/figure_phase8_model_sensitivity.png`, `experiments/exp007/figures/figure_phase8_evaluation_sensitivity.png`, `experiments/exp007/figures/figure_phase8_leave_one_group_out.png`: figures

## Validation

Build checks: 14/14 PASS.
