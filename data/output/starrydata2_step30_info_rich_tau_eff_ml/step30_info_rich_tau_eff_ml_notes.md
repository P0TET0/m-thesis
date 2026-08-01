# Step30 Information-Rich tau_eff ML Notes

Step30 is an additional evaluation restricted to samples with richer material information.
It uses the Step18 `log_tau_eff` target and does not refit `tau_eff` or recalculate sigma, PF, or ZT.

This analysis does not causally prove the reason for lower ML performance. Restricting to information-rich samples may improve feature quality, but it also reduces the number of training samples.

- Level 0 (all_recommended): representative model `gradient_boosting`, RMSE=1.8467, Spearman=0.4672.
- Level 1 (basic_material_info): representative model `gradient_boosting`, RMSE=1.7316, Spearman=0.4316.

If performance improves for information-rich samples, this supports insufficient material information as a candidate contributor to ML performance loss.
If performance does not improve, other candidates remain important, including the tau_eff definition, C(T) construction, data variability, and extrapolation difficulty.

- Level 0: Reference level or skipped level.
- Level 1: Information-rich restriction is associated with lower RMSE.
- Level 2: Reference level or skipped level.
- Level 3: Reference level or skipped level.
- Level 4: Reference level or skipped level.
