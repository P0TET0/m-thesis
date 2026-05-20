# Discussion Draft

## Interpretation of fitted tau_eff
The present tau_eff is a relative effective scalar, not an absolute physical relaxation time.

## Why direct fitting outperforms ML prediction
Direct fitting uses sigma observations for each sample and is expected to outperform ML prediction.

## Implications for thermoelectric screening
The ML workflow is closer to an unknown-material screening task because it estimates tau_eff from material descriptors.

## Importance of additive and structure information
The lack of detailed additive, structure, and sintering annotations limits ML performance.

## Role of sintering and microstructure
Sintering and microstructure can influence transport properties, but most sintering fields remain unknown, so they should not be treated as confirmed error causes.

## Candidate materials and screening priorities
Candidate tables prioritize high ZT, low kappa/high sigma, nanocarbon, and lower attention-flag materials.

## Future work
Future work should improve annotation completeness, confirm original papers, and expand prediction beyond sigma.

Seebeck coefficient and thermal conductivity were not predicted in this workflow.
