# Step23 Error Cause Analysis Notes

## Purpose
Organize candidate causes for fitting-vs-ML error differences using material annotations.

## Inputs
Step23 primarily uses Step22 sample-level comparison results and enriches them with Step17/18/19/15 metadata when available.

## Error Pattern Definitions
Patterns separate cases where fitting is good but ML is bad, both are bad, fitting is bad but ML is better, both are good, and not evaluable cases.

## Main Error Hypotheses
Hypotheses include ML tau_eff prediction error, insufficient material features, missing additive/structure annotations, possible sintering or microstructure effects, ZT observation/unit inconsistency, and direct fitting limitations.

## Material System Trends
See `step23_error_by_material_system.csv`.

## n/p Type Trends
See `step23_error_by_np_type.csv`.

## Additive and Structure Information
Unknown additive and structure groups are treated as missing information and review priorities, not proven causes.

## Sintering Information Policy
Sintering methods are mostly unknown; unknown sintering is treated as missing information, not as a confirmed error cause.

## High-ZT Error Cases
See `step23_high_zt_error_cases.csv`.

## Recommended Manual Review
See `step23_manual_review_priority_samples.csv` and `step23_sintering_check_priority_samples.csv`.

## Important Caveats
Step23 does not prove causal mechanisms.
Sintering methods are mostly unknown; unknown sintering is treated as missing information, not as a confirmed error cause.
ML errors may reflect insufficient features, missing annotations, or poor generalization under DOI split.
Seebeck coefficient and thermal conductivity were not predicted; PF/ZT errors depend on observed S and kappa.
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.

## Next Step
Step24 should use these review priorities and feature flags to extract candidate materials.
