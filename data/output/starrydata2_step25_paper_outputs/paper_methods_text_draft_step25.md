# Methods Draft

## Data source
Starrydata2 was used as the literature-derived source of thermoelectric transport data.

## Data preprocessing
Five transport properties were extracted and normalized: Electrical conductivity, Electrical resistivity, Seebeck coefficient, Thermal conductivity, and ZT.

## n/p classification
Samples were provisionally classified as n-type or p-type using the sign of the Seebeck coefficient where available.

## Effective relaxation parameter fitting
The workflow fitted sigma_obs(T) = C(T) * tau_eff for each sample. tau_eff is a relative effective scalar, not physical seconds.

## Unit normalization
The units of sigma, rho, Seebeck, kappa, and ZT were normalized before fitting and downstream evaluation.

## PF/ZT calculation
PF = S^2 sigma and ZT = S^2 sigma T / kappa were used. PF/ZT use predicted sigma and observed S/kappa.

## ML model
The ML model used fitted log_tau_eff as the supervised label. Features included composition, material_system, n/p, additive, structure, and element flags. The primary evaluation used a DOI group split.

## Candidate extraction
High ZT, low kappa and high sigma, rare-metal attention, toxicity attention, and nanocarbon candidates were summarized from existing outputs.
