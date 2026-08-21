# Step37: minimal transparent scatter figures

Step37 creates exactly one PNG per exp006 broad material system.

- RMSE is calculated from all positive points before display clipping.
- RMSE is not written on the figure; it is provided in a separate CSV that
  maps each material system to its PNG.
- Figures contain tick values only—no title or axis labels.
- Both axes use `10^2`--`10^8 S/m`.
- The plotting field is opaque white; all surrounding padding is transparent.
- Existing figures are not modified.

```powershell
python experiments/exp005/plot_sigma_ml_minimal_transparent_step37.py
```

Outputs:

```text
data/output/starrydata2_step37_sigma_ml_minimal_transparent/
```

The alpha-channel audit is saved as `png_transparency_audit_step37.csv`.
