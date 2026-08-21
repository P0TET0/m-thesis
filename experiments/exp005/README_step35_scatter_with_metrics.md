# Step35: fixed-axis scatter plots with y=x metrics

Step35 creates a new annotated version of every Step33 sigma scatter figure.
Existing Step33 figures remain unchanged.

Each panel shows:

- visible point count (`n`);
- RMSE(log10);
- MAE(log10);
- factor-2 rate; and
- Lin's concordance correlation coefficient on log10 values.

Both axes remain fixed to `10^2`--`10^8 S/m`.

```powershell
python experiments/exp005/plot_sigma_ml_scatter_with_metrics_step35.py
```

Outputs are written to:

```text
data/output/starrydata2_step35_sigma_ml_scatter_annotated_metrics/
```
