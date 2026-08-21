# exp005 vs exp006 common-cohort ZT comparison

## Primary comparison

- Shared comparable rows: 383.
- Shared samples: 109.
- Shared DOI: 83.
- Both models are evaluated on exactly the same exp006 temperature rows.
- Reference is reported ZT from the Step5 full property curves, matched to the same sample within +/-2 K.
- Both models use the same observed Seebeck coefficient, thermal conductivity, and temperature.
- No interpolation or extrapolation is used.
- Sample-level metrics use the maximum over the same shared comparable rows.

## Models

- exp005: `gradient_boosting` from `split_doi_group_80_20_step18` / `test`.
- exp005 sigma is reconstructed as the Step12 empirical prefactor times the Step19 predicted relative tau_eff.
- The Step12 prefactor hierarchy and 25 K bins are rebuilt from the same Step5 electrical curves used for this comparison.
- exp006: existing sample-holdout sigma prediction; it is not retrained here.

## Overall results

- exp005 row MAE: 0.19763; sample-max MAE: 0.206127; ZT>=1 F1: undefined.
- exp006 row MAE: 0.198765; sample-max MAE: 0.225476; ZT>=1 F1: undefined.
- Observed sample-maximum ZT>=1 count: 0; F1 is undefined when this count is zero.
- Lower sample-max absolute error: exp005=61, exp006=48, tie=0.

## Reconstruction and interpretation cautions

- The old full Step20/Step21 prediction CSV files are not present; therefore exp005 is method-reexecuted from its saved Step19 DOI-test predictions rather than copied from the truncated Excel preview.
- This is the preferred fair comparison for the current files, but it is not a byte-for-byte reproduction of the historical Step21 output because the shared Step5 source is deliberately used for both experiments.
- exp005 is DOI-held-out; exp006 is sample-held-out. Restricting to shared held-out samples removes evaluation-row differences, but exp006 may still have other training samples from the same DOI.
- The existing `figure_material_system_parity_reported_zt_2K.png` remains the full exp006-cohort result. Use the new common-cohort material-system figure for direct exp005-vs-exp006 claims.

## Audit counts

- exp006_input_rows: 19,008
- exp006_input_samples: 3,191
- exp005_primary_test_samples: 3,201
- common_sample_rows_before_doi_check: 2,010
- common_samples_before_doi_check: 390
- doi_mismatched_or_missing_rows: 0
- common_sample_and_doi_rows: 2,010
- common_sample_and_doi_samples: 390
- common_sample_and_doi_DOIs: 275
- prefactor_available_rows: 2,010
- reported_ZT_shared_comparable_rows: 383
- reported_ZT_shared_comparable_samples: 109
- reported_ZT_shared_comparable_DOIs: 83
- source_electrical_curve_rows: 18,986
- expanded_electrical_point_rows: 328,691
- selected_electrical_point_rows: 326,513
- baseline_electrical_rows: 326,260
- baseline_electrical_samples: 18,466
- step12_fit_source_samples: 18,466
