# exp006 PF/ZT downstream calculation notes

This output reuses the existing exp006 `sigma_pred_S_per_m`. It does not retrain a model, refit `tau_eff`, or replace any source file.

## Equations

- `PF_pred_exp006_W_per_mK2 = seebeck_obs_V_per_K ** 2 * sigma_pred_exp006_S_per_m`
- `ZT_pred_exp006 = PF_pred_exp006_W_per_mK2 * temperature_K / kappa_obs_W_per_mK`
- Observed-reference PF/ZT use the same expressions with `sigma_obs_S_per_m`.

## Data handling

- Seebeck values came from exp006 `S_uV_per_K` and were converted to V/K by multiplying by `1e-6`.
- Kappa and observed ZT were accepted only from exact `sample_key + temperature_K` matches.
- Nearest-temperature matching and interpolation were intentionally not used.
- PF-computable prediction rows: 19,000.
- ZT-computable prediction rows: 21.
- ZT-vs-observed comparable rows: 15.

Use the report and problem-row tables to audit exclusions and the small exact-match ZT evaluation set.
