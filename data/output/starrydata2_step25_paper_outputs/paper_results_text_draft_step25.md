# Results Draft

## tau_eff fitting
Step12 fitted a relative effective tau_eff parameter for samples with sufficient sigma observations.

## sigma validation
Step13 evaluated fitted tau_eff using held-out temperature rows and summarized validation quality.

## PF/ZT prediction using fitted tau_eff
Step14-16 estimated PF/ZT using predicted sigma and observed S/kappa. S and kappa were not predicted.

## ML prediction of log_tau_eff
Step19 trained ML models to predict fitted log_tau_eff labels from material and annotation features.

## sigma/PF/ZT prediction using ML tau_eff
Step20-22 propagated ML tau_eff predictions into sigma/PF/ZT estimates and compared them with the direct fitting workflow.

## Error cause analysis
Step23 summarized error patterns and review targets. These are hypotheses and review targets, not proven causal mechanisms.

## Candidate materials
Step24 extracted material candidates for follow-up screening and manual review.

ML版はfitting版より性能が低いが、材料特徴量からtau_effを予測する未知材料スクリーニングに近い。
