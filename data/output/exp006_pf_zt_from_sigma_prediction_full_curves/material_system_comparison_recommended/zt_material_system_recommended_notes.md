# Recommended material-system ZT comparison

## Scenario roles

- `reported_zt_2K`: primary analysis. Reported ZT and kappa are independently matched within ±2 K.
- `calculated_zt_2K`: coverage extension. The reference is calculated from observed sigma, S, and kappa at ±2 K; it is not reported ZT.
- `reported_zt_3K`: temperature-tolerance sensitivity analysis. It is not substituted for the primary ±2 K result.
- No interpolation or extrapolation is used in any scenario.

## Coverage

- Reported ZT, ±2 K: 2,279 rows, 619 samples, 556 Step31-classified samples.
- Calculated ZT reference, ±2 K: 3,053 rows, 806 samples, 722 Step31-classified samples.
- Reported ZT sensitivity, ±3 K: 2,770 rows, 734 samples, 660 Step31-classified samples.

## Material-system figures

- The common main-group set is defined from the primary scenario using at least 10 classified samples.
- Main groups (13): Other formula systems, selenide, SnTe, CoSb skutterudite, oxide, BiTe, PbTe, GeTe, sulfide, BiSbTe, SbTe, telluride, Mg2SiSn.
- Parity plots use a common x/y scale across all three scenarios.
- Sample-level summaries use maximum ZT within rows where both the scenario reference and prediction are available.

## Interpretation

- Prefer the reported-ZT ±2 K result for conclusions.
- Use calculated-ZT ±2 K to assess whether conclusions persist with greater coverage.
- Use reported-ZT ±3 K only to assess sensitivity to the matching tolerance.
