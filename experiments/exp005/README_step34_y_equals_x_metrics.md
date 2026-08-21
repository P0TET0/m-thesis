# Step34: agreement with the y=x line

Step34 quantifies how closely the exp005 DOI-holdout sigma scatter points follow
the `y=x` line for each exp006 broad material system.

It reports metrics for both:

- all positive points; and
- points visible in the Step33 fixed range (`10^2`--`10^8 S/m` on both axes).

```powershell
python experiments/exp005/evaluate_sigma_ml_y_equals_x_step34.py
```

Outputs are written to
`data/output/starrydata2_step34_sigma_ml_y_equals_x_metrics/`:

- `y_equals_x_metrics_by_material_system_step34.csv`
- `y_equals_x_metrics_ranked_visible_step34.csv`
- `y_equals_x_metric_definitions_step34.csv`
- `step34_y_equals_x_metrics_report.txt`

The primary direct-agreement metrics are log10 RMSE, log10 MAE, perpendicular
distance from `y=x`, multiplicative-error factor, factor-2 rate, and Lin's
concordance correlation coefficient on log10 values.
