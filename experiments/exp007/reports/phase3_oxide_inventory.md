# Phase 3: oxide data inventory

## Scope

Phase 3 audits the `broad::oxide` rows in the frozen `broad_family_100K` held-out cohort. It does not change the model, residuals, or exp006 taxonomy.

## Inventory size

- Rows: 3,437
- Samples: 610
- Papers: 439
- Distinct normalized labels: 506
- Provisional composition families: 13

## Main findings

- Every current oxide row was admitted by the same rule: the parser found element `O` in `formula_raw`.
- The largest provisional group is `oxide::cobalt_based` with 228 samples and 1,309 rows.
- 184 samples carry at least one review flag.
- 18 samples are heuristic organic/polymer O-containing false-positive candidates rather than clear inorganic oxides.
- 54 samples are mixed-anion O+S/Se/Te systems and should not be silently merged with ordinary oxides.
- 469 of 506 material labels occur in only one held-out sample, so label-level subclasses would be too sparse without pooling.
- 69 provisional family/carrier/100 K evaluation cells have fewer than 3 rows, 3 samples, or 1 paper. This is an evaluation-cohort warning, not a training-feasibility test.

## Interpretation

The current `broad::oxide` group is compositionally heterogeneous. A finer taxonomy should first separate likely non-oxide organic entries and mixed-anion materials, then test a small number of pooled oxide families. Crystal structure labels such as perovskite, layered cobaltite, or delafossite cannot be assigned reliably from the present formula strings alone and require explicit rules or manual metadata review.

## Outputs

The CSV files are canonical machine-readable outputs. The Excel workbook is a filtered review view. `phase3_oxide_review_queue.csv` contains the samples needing manual chemistry review before the final taxonomy is fixed.
