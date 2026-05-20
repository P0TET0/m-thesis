# Step18 tau_eff ML Dataset Notes

## Purpose
Step18 prepares a one-row-per-sample machine learning dataset for predicting fitted tau_eff in Step19.

## Target Variable
The primary target is `target_log_tau_eff_step18`, copied from `log_tau_eff_step12`.
`target_tau_eff_step18` is retained for metadata and inspection.

## Input Features
Features include material metadata, final n/p annotations, additive and structure annotations, sintering annotations, nanocarbon flags, and regex-derived element indicators from composition.

## Data Exclusion Policy
Recommended ML rows require an available target, `fit_status_step12 == ok`, enough fitting rows, and fitting/validation errors within the configured thresholds. Rows that fail these checks remain in the full dataset with exclusion reasons.

## Train/Test Split Policy
Random splits are reproducible hashes of `sample_key`. DOI group splits hash DOI where available so samples from the same DOI stay in the same split; rows without DOI fall back to sample_key.

## Leakage Prevention
Fitting error columns, PF/ZT prediction results, and target columns are excluded from the feature matrix.

## Important Caveats
The target tau_eff is a relative effective scalar, not a physical relaxation time in seconds.
Step18 does not train a model.
Step18 prepares the dataset for Step19.
Features derived from fitting errors or PF/ZT prediction results are not used as model inputs to avoid leakage.
DOI-based split is recommended for more reliable evaluation.

## Next Step
Step19 should train and evaluate models using `tau_eff_ml_dataset_recommended_step18.csv`, `tau_eff_ml_feature_matrix_step18.csv`, `tau_eff_ml_target_step18.csv`, and `tau_eff_ml_splits_step18.csv`.
