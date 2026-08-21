# Step36: calculate metrics before clipping the scatter display

Step36 uses every positive DOI-holdout point to calculate each material
system's `y=x` agreement metrics. Only after those metrics have been selected
and validated are the displayed x/y axes limited to `10^2`--`10^8 S/m`.

Each annotation distinguishes:

- `n(metric, all)`: points used for metric calculation; and
- `visible`: points visible after applying the fixed axes.

Existing Step35 figures are not changed.

```powershell
python experiments/exp005/plot_sigma_ml_scatter_all_point_metrics_step36.py
```

Outputs are written to:

```text
data/output/starrydata2_step36_sigma_ml_scatter_all_point_metrics/
```
