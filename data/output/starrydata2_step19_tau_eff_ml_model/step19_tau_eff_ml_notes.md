# Step19 tau_eff ML Model Notes

## Purpose
Train machine learning models that predict fitted `log_tau_eff` from Step18 material features.

## Target
The target is `target_log_tau_eff_step18`. `tau_eff` is retained only as an exponentiated relative-scale value.

## Features
The model uses only columns from `tau_eff_ml_feature_matrix_step18.csv` after leakage checks, numeric conversion, imputation, and zero-variance filtering.

## Models
The run compares a mean baseline, Ridge regression, RandomForest, ExtraTrees, and GradientBoosting.

## Splits
Models are evaluated on random 80/20, random 70/15/15, and DOI group 80/20 splits.

## Selected Model
The selected model is chosen by validation RMSE on `split_random_70_15_15_step18`.

## Main Results
See `step19_tau_eff_ml_report.txt` and `tau_eff_ml_model_comparison_step19.csv`.

## Important Caveats
The model predicts log_tau_eff, not sigma directly.
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.
No sigma, PF, or ZT recalculation is performed in Step19.
The DOI group split is more reliable than random split for evaluating generalization.
Final all-sample predictions are for downstream screening, not unbiased evaluation.

## Next Step
Step20 should use the predicted tau_eff values to compute sigma_pred_ML, PF_pred_ML, and ZT_pred_ML.
