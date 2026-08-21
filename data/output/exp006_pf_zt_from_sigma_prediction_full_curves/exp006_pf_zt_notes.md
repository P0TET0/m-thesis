# exp006 PF/ZT downstream calculation notes

This output reuses the existing exp006 `sigma_pred_S_per_m`. It does not retrain a model, refit `tau_eff`, or replace any source file.

## Equations

- `PF_pred_exp006_W_per_mK2 = seebeck_obs_V_per_K ** 2 * sigma_pred_exp006_S_per_m`
- `ZT_pred_exp006 = PF_pred_exp006_W_per_mK2 * temperature_K / kappa_obs_W_per_mK`
- Observed-reference PF/ZT use the same expressions with `sigma_obs_S_per_m`.

## Data handling

- Seebeck values came from exp006 `S_uV_per_K` and were converted to V/K by multiplying by `1e-6`.
- Original kappa and ZT curves were expanded with the exp005 Step10 rules.
- Kappa and ZT were independently joined to exp006 `T_K` by same-sample nearest temperature within `±2 K`.
- Interpolation and extrapolation were not used.
- Unit normalization and abnormal-ZT-unit rejection use the exp005 Step11 rules.
- PF-computable prediction rows: 19,000.
- ZT-computable prediction rows: 3,053.
- ZT-vs-observed comparable rows: 2,279.

Use the report and row-level provenance columns to audit temperature matches and unit exclusions.
