# Step24 Material Candidate Selection Notes

## Purpose
Step24 selects material candidates from existing Step21-23 fitting, ML, comparison, and error-analysis outputs.

## Candidate Criteria
Candidates are selected using observed/high predicted ZT, low thermal conductivity, high electrical conductivity, composition-based attention flags, nanocarbon keywords, and review priorities.

## High ZT Candidates
High ZT candidates satisfy observed, ML-predicted, or fitting-predicted ZT thresholds.

## Low Thermal Conductivity and High Electrical Conductivity Candidates
These candidates satisfy the configured kappa threshold and either observed or ML-predicted sigma threshold.

## Rare Metal Attention
Low rare metal attention means no configured rare-metal attention elements were detected from composition.

## Toxicity Attention
Low toxicity attention means no configured toxicity attention elements were detected from composition.

## Nanocarbon Candidates
Nanocarbon candidates are detected from Step9 annotations, nanocarbon type labels, or carbon-related keywords.

## Balanced Recommended Candidates
Balanced recommended candidates combine performance flags with lower toxicity attention or manual-review priority.

## Manual Review and Sintering Check
Manual review and sintering check outputs prioritize high-scoring candidates and samples with missing additive, structure, or sintering information.

## Important Caveats
Rare-metal-free and low-toxicity labels are provisional screening flags based on composition.
They are not final material safety or resource classifications.
Nanocarbon identification is based on available keywords and may miss cases without explicit annotations.
Many additive, structure, and sintering fields are still unknown.
Downstream ML predictions are for screening, not unbiased evaluation.
Step24 does not perform new prediction or model training.

## Next Step
Step25 should organize Step12-24 results into thesis-ready fitting, ML, comparison, error-analysis, and candidate-material tables.
