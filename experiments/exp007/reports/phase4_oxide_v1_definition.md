# Phase 4: oxide taxonomy v1 definition

## Definition

- Version: `oxide_v1`
- Definition date: `2026-10-13`
- Ordered-rule SHA-256: `185239a6fa68e0403c1984b9e6338dd5eec4842b116cbab6201d33e2fc706999`
- Source: Phase 3 broad-oxide held-out cohort
- Scope: composition-only; no crystal-structure class is inferred

## Rule order

1. Exclude organic/polymer/carbon-composite O-containing labels.
2. Separate mixed-anion O+S/Se/Te, O+As/P, and O+N/halogen entries.
3. Split remaining inorganic O-containing entries by priority-ordered element markers.
4. Use an explicit other-inorganic-O group instead of inventing a structure class.
5. Use a fail-safe outside-scope class if parsed O is absent.

## Applied cohort

- Rows: 3,437
- Samples: 610
- Papers: 439
- Material labels: 506
- Non-empty v1 groups: 14
- Default-eligible samples: 553
- Review samples: 36
- Excluded samples: 21

## Descriptive findings

- The largest v1 group is `oxide_v1::Co_containing_oxide` with 226 samples.
- 0 sample-level assignment conflicts were found.
- Error metrics describe the existing broad-family predictions and are not post-retraining performance estimates.

## Limitations

`oxide_v1` uses formula text and parsed elements only. It does not distinguish perovskite, spinel, delafossite, rocksalt, layered cobaltite, or other crystal structures. The O+N/halogen group and non-formula-like or multi-component labels remain review items before model revalidation.
