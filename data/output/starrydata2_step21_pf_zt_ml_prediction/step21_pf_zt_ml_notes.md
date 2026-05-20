# Step21 PF/ZT Prediction from ML sigma

## Purpose
Calculate PF_pred_ML and ZT_pred_ML from Step20 sigma_pred_ML.

## Formula
PF_pred_ML = S_obs^2 * sigma_pred_ML.
ZT_pred_ML = S_obs^2 * sigma_pred_ML * T / kappa_obs.

## Evaluation Scope
Primary DOI test rows are used for evaluation. Downstream all-samples predictions are candidate screening outputs.

## Primary DOI Test Results
See `step21_pf_zt_ml_report.txt`.

## Downstream Candidate Screening
Candidate samples are selected from downstream all-samples predictions by high observed, calculated, or predicted ZT.

## Comparison with Direct Fitting
The direct fitting comparison uses the same observed S and kappa with Step12 fitted sigma.

## Important Caveats
Step21 does not predict Seebeck coefficient or thermal conductivity.
PF_pred_ML and ZT_pred_ML are calculated using sigma_pred_ML and observed S/kappa.
Downstream all-samples predictions are for screening, not unbiased evaluation.
Direct fitting is expected to outperform ML tau prediction because it uses sigma observations directly.

## Next Step
Step22 should compare direct fitting and ML versions of sigma, PF, and ZT for reporting.
