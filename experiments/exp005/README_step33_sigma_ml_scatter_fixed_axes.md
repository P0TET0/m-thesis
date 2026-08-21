# Step33: fixed-axis exp005 ML sigma scatter figures

This step leaves every Step32 figure unchanged and creates a new figure set in:

```text
data/output/starrydata2_step33_sigma_ml_scatter_fixed_1e2_1e8/
```

Both the measured-sigma and ML-predicted-sigma axes are fixed to
`10^2`--`10^8 S/m` for the combined figure, material-system grid, and all
individual material-system figures.

```powershell
python experiments/exp005/plot_sigma_ml_scatter_fixed_axes_step33.py
```

Points outside the fixed range are clipped from the view. Their counts are
recorded in `scatter_fixed_axes_visibility_audit_step33.csv`.
