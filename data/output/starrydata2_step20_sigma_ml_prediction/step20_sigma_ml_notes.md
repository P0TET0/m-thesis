# Step20 sigma_pred_ML Notes

## Purpose
Compute machine-learning-derived electrical conductivity from Step19 tau_eff predictions.

## Formula
`sigma_pred_ML(T) = prefactor_C(T) * tau_eff_pred`.

## Inputs
The calculation uses Step19 tau_eff predictions and Step12 `prefactor_C_S_per_m_step12` temperature rows.

## Evaluation Data
The primary evaluation uses the DOI group split test rows. Final all-samples predictions are downstream-only.

## Selected Model vs DOI-best Model
Step20 distinguishes the Step19 selected model from the model with the best DOI split test RMSE.

## Main Results
See `step20_sigma_ml_report.txt`.

## Comparison with Direct Fitting
Direct fitting uses sigma_obs to fit tau_eff, so it is expected to outperform material-feature ML prediction.

## Important Caveats
Step20 does not calculate PF or ZT.
Step20 uses tau_eff predicted by ML to calculate sigma_pred_ML.
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.
Direct fitting performance is expected to be better than ML prediction because it uses sigma_obs directly.
Final all-samples predictions are for downstream screening, not unbiased model evaluation.

## Next Step
Step21 should compute PF_pred_ML and ZT_pred_ML from `sigma_ml_downstream_ready_step20.csv`.
