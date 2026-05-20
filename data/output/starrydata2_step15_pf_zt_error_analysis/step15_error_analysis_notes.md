# Step15 PF/ZT Error Analysis Notes

## Purpose

Step15 analyzes the PF and ZT prediction errors produced in Step14. It does not train a new model and does not refit tau_eff.

## Assumptions Up To Step14

The electrical conductivity prediction comes from fitted tau_eff. The Seebeck coefficient and thermal conductivity are not predicted in Step14 or Step15.

## Equations

PF_pred = S_obs^2 * sigma_pred

ZT_pred = S_obs^2 * sigma_pred * T / kappa_obs

Here, S_obs and kappa_obs are experimental values. Only sigma is predicted.

## Interpretation

ZT_pred vs ZT_calc_from_obs mainly shows the effect of sigma prediction error because both use standardized observed S and kappa.

ZT_pred vs ZT_obs compares against the Starrydata/literature ZT value and can include effects from reported ZT values, unit handling, temperature alignment, or data consistency.

Large errors can come from sigma prediction error, inconsistency in observed ZT, temperature matching, units, additives/structure information, or unconfirmed sintering method.

## Sintering

Sintering methods are still not investigated in Step15. Use `sintering_check_candidates_step15.csv` to check only important samples in later steps.
