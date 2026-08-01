# Step2B Eta Assignment Report

## Summary

- input_file: experiments\exp006\data\processed\step1_eta_input_candidates.parquet
- input_rows: 170363
- max_rows: none
- lookup_table_file: experiments\exp006\data\processed\step2_eta_lookup_table.parquet
- lookup eta range: -50 to 500
- lookup S_abs_uV_per_K range: 0.566999 to 4481.01
- eta_status counts: {'ok': 169831, 'out_of_range_high_S': 532}
- eta ok rows: 169831
- eta failed or out-of-range rows: 532
- eta >= 1 rows: 97086
- is_valid_for_sigma0_step3 == True rows: 97086
- conservative eta >= 1 candidate rows: 91134
- eta summary: min=-49.9929, median=1.41037, max=283.169
- S_abs_uV_per_K summary: min=1.00117, median=133.746, max=7.90015e+06
- S_eta_abs_error_uV_per_K summary: min=-9.09495e-13, median=0, max=4.54747e-13
- max abs S_eta_abs_error_uV_per_K: 9.09495e-13
- eta >= 1 roughly corresponds to S_abs_uV_per_K <= 150.876
- elapsed_seconds: 14.92

## Parquet Status

- step2_eta_calculated.parquet: saved
- step2_eta_ge1_candidates.parquet: saved
- step2_conservative_eta_ge1_candidates.parquet: saved

## Carrier Type Eta >= 1 Counts

| carrier_type | eta_ge1_count |
| --- | --- |
| n | 44393 |
| p | 52693 |

## Material Family Eta >= 1 Top 10

| material_family_raw | carrier_type | eta_ge1_count | row_count | sample_count | paper_count |
| --- | --- | --- | --- | --- | --- |
| unknown | p | 52693 | 95504 | 13469 | 3343 |
| unknown | n | 44393 | 74859 | 10744 | 2687 |

## Sanity Check

- output_rows_equal_input_rows: True
- row_id_unique: True
- carrier_type_p_or_n_only: True
- S_abs_matches_abs_signed_S: True
- s_abs_dimensionless_consistent: True
- eta_status_not_missing: True
- eta_status_allowed: True
- eta_finite_for_ok: True
- eta_nan_for_not_ok: True
- F0_positive_finite_for_ok: True
- S_eta_abs_error_le_0_1_uV_for_ok: True
- eta_ge_1_rule: True
- is_valid_for_sigma0_step3_rule: True
- ge1_candidates_rule: True
- conservative_ge1_candidates_rule: True
- no_sigma0_columns: True

## Warnings And Step3 Notes

- WARNING: none
- Step3 can compute sigma0 as sigma_S_per_m / F0_eta for is_valid_for_sigma0_step3 rows.
- sigma0 is intentionally not computed in Step2B.
