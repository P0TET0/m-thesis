# Step26 Predicted vs Observed Scatter Notes

## Purpose

Step26 visualizes existing predicted values against experimental values for sigma, PF, and ZT.
It does not run new prediction, tau_eff refitting, or ML retraining.

## How to Read the Figures

The horizontal axis is the observed experimental value and the vertical axis is the predicted value.
The dashed diagonal line is y = x. Points closer to this line indicate better agreement.
Sigma and PF are shown on log-log axes because their ranges are wide. ZT is shown on linear axes,
with optional positive-only log-log figures.

## Direct Fitting Version

The direct fitting version uses the fitted tau_eff-derived predictions already produced in earlier steps.
Direct fitting uses sigma observations to fit tau_eff, so it is expected to perform better.

## ML Version

The ML version uses predictions based on the ML-predicted tau_eff columns already produced in earlier steps.
ML version predicts tau_eff from material features and is closer to unknown-material screening.

## Metrics

The metrics table reports MAE, RMSE, MAPE, log MAE, log RMSE, R2, log R2, Pearson, Spearman,
and rates within 25%, 50%, and a factor of 2. Log metrics are computed only where both observed
and predicted values are positive.

## Important Caveats

Direct fitting uses sigma observations to fit tau_eff, so it is expected to perform better.
ML version predicts tau_eff from material features and is closer to unknown-material screening.
S and kappa are not predicted; PF and ZT use observed S and kappa.
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.
