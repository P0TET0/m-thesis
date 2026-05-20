# Step22 Fitting vs ML Comparison Notes

## Purpose
Compare direct fitting and ML tau_eff prediction versions for sigma, PF, ZT, high-ZT classification, and ranking.

## What is Compared
The comparison uses the same primary DOI test rows and temperature points where possible.

## Direct Fitting Version
The fitting version uses Step12 fitted tau_eff derived directly from sigma observations.

## ML tau_eff Prediction Version
The ML version uses Step19 predicted tau_eff from material features, then Step20/21 calculated sigma/PF/ZT.

## Main Results
See `step22_comparison_report.txt`.

## Interpretation
The fitting version is an upper-reference style result, while the ML version is closer to unknown-material prediction.

## Why ML Can Be Worse Than Fitting
The ML model has limited material features and does not use sigma observations to fit tau_eff for the target sample.

## Important Caveats
The fitting version uses sigma observations to fit tau_eff, so it is not a fair unknown-material prediction baseline.
The ML version predicts tau_eff from material features and is closer to the intended ML task.
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.
Seebeck coefficient and thermal conductivity are not predicted in either version.
PF and ZT are computed using predicted sigma and observed S/kappa.

## Next Step
Step23 should analyze error causes by material system, n/p type, additives, structure, nanocarbon, rare-metal/toxicity flags, and sintering status.
