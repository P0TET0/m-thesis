# ZT comparison by existing Step31 material system

## Scope

- ZT values come from the full-curve exp006 PF/ZT output and are not recomputed here.
- Step31 material-system labels are joined by `sample_key` only.
- Row-level comparisons require both `ZT_obs` and `ZT_pred_exp006` at the same exp006 anchor row.
- Sample-level comparisons use maximum ZT within those comparable rows.
- High-ZT classification threshold: 1.

## Coverage

- Comparable rows: 2,279; classified by Step31: 2,104 (92.3%).
- Comparable samples: 619; classified by Step31: 556 (89.8%).
- Main figures include classified groups with at least 10 comparable samples.
- Main groups (13): Other formula systems, selenide, SnTe, CoSb skutterudite, oxide, BiTe, PbTe, GeTe, sulfide, BiSbTe, SbTe, telluride, Mg2SiSn.

## Reference-only groups

- Unclassified: 63 samples
- SiGe: 4 samples
- Unknown material group: 4 samples

## Interpretation cautions

- The number of temperature rows differs by sample, so material-system conclusions should prioritize the sample-level tables and figures.
- MAPE excludes observed ZT equal to zero; MAE and RMSE retain those rows.
- Results describe propagation of the existing sigma prediction with observed S and kappa, not a fully independent ZT model.
