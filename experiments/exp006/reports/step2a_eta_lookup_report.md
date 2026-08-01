# Step2A Eta Lookup Report

## Summary

- eta_min: -50
- eta_max: 500
- d_eta: 0.005
- grid points: 110001
- S_abs_uV_per_K max: 4481.01
- S_abs_uV_per_K min: 0.566999
- S_abs_uV_per_K at eta = 0: 204.501
- S_abs_uV_per_K at eta = 1: 150.876
- S_abs_uV_per_K at eta = 2: 112.389
- S_abs_uV_per_K at eta = 5: 55.8143
- eta >= 1 corresponds roughly to S_abs_uV_per_K <= 150.876
- parquet status: saved
- elapsed_seconds: 0.95

## Monotonic Check

- eta monotonic increasing: True
- S_abs_uV_per_K monotonic decreasing: True

## Sanity Check

- eta_monotonic_increasing: True
- F0_eta_positive_finite: True
- F1_eta_nonnegative_finite: True
- s_model_positive_finite: True
- S_abs_uV_per_K_positive_finite: True
- S_abs_uV_per_K_monotonic_decreasing: True
- S_abs_uV_at_eta_0_within_1_uV: True
- S_abs_uV_at_eta_1_within_1_uV: True
- S_abs_uV_at_eta_2_within_1_uV: True

## Notes

- This Step2A creates only the numerical lookup table.
- Step1 data is not modified and eta is not assigned in this step.
- F1 is computed by cumulative trapezoid integration of F0 over eta.
