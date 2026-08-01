# Step3 Sigma0 Report

## Summary

- input_file: experiments\exp006\data\processed\step2_eta_ge1_candidates.parquet
- input_rows: 97086
- max_rows: none
- sigma0_calc_status counts: {'ok': 97086}
- is_valid_sigma0 == True rows: 97086
- is_conservative_valid_sigma0 == True rows: 91134
- eta summary: min=1.00006, median=3.45785, max=283.169
- F0_eta summary: min=1.3133, median=3.48886, max=283.169
- sigma_S_per_m summary: min=4.26326e-12, median=73802.4, max=2.67963e+20
- sigma0_S_per_m summary: min=1.76645e-12, median=19212.9, max=2.28273e+19
- log10_sigma0_S_per_m summary: min=-11.7529, median=4.28359, max=19.3585
- sigma0_reconstruction_log_error summary: min=-9.64327e-17, median=0, max=9.64327e-17
- sigma0_reconstruction_log_error max_abs: 9.64327e-17
- sample summary rows: 16542
- material family summary rows: 2
- elapsed_seconds: 36.44

## Parquet Status

- step3_sigma0_calculated.parquet: saved
- step3_sigma0_valid.parquet: saved
- step3_conservative_sigma0_valid.parquet: saved

## Valid Sigma0 By Carrier Type

| carrier_type | valid_sigma0_count |
| --- | --- |
| n | 44393 |
| p | 52693 |

## Valid Sigma0 By Material Family Top 10

| material_family_raw | carrier_type | row_count | valid_sigma0_count | T_min_K | T_max_K | eta_min | eta_median | eta_max | S_abs_median_uV_per_K | sigma_median_S_per_m | log10_sigma_median_S_per_m | sigma0_median_S_per_m | log10_sigma0_median_S_per_m | sigma0_min_S_per_m | sigma0_max_S_per_m | log10_sigma0_min_S_per_m | log10_sigma0_max_S_per_m | paper_count | sample_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unknown | p | 52693 | 52693 | 0.01 | 1824.224 | 1.0000563008067502 | 3.5022745757596407 | 283.16880204763635 | 76.27119 | 62573.1 | 4.796387671351883 | 16522.84634752974 | 4.2180848642038065 | 2.628767115742609e-07 | 1.6184983857558012e+19 | -6.580247886631493 | 19.20911227058586 | 2738 | 9269 |
| unknown | n | 44393 | 44393 | 0.126506 | 1426.798 | 1.0000694164133703 | 3.407687509228104 | 281.3801858489873 | 77.97957000000001 | 86716.65592825411 | 4.938102521732263 | 23194.79390746882 | 4.365390518038366 | 1.7664506896338227e-12 | 2.2827268909530087e+19 | -11.75289848136462 | 19.3584539549212 | 2245 | 7273 |

## Valid Sigma0 By Sigma Source

| sigma_source | valid_sigma0_count |
| --- | --- |
| conductivity_direct | 39537 |
| resistivity_converted | 57549 |

## Valid Sigma0 By Match Method

| match_method | valid_sigma0_count |
| --- | --- |
| exact | 2222 |
| nearest | 94864 |

## Sanity Check

- output_rows_equal_input_rows: True
- row_id_unique: True
- valid_sigma0_positive_finite: True
- valid_log10_sigma0_finite: True
- valid_reconstructed_sigma_positive_finite: True
- reconstruction_log_error_le_1e_10: True
- status_ok_matches_is_valid_sigma0: True
- valid_output_only_valid_rows: True
- conservative_output_only_conservative_valid_rows: True
- sigma0_formula_consistent: True
- F0_positive_for_valid: True
- sigma_positive_for_valid: True
- log10_sigma0_consistent: True

## Warnings And Step4 Notes

- WARNING: none
- Step4 should build 100 K temperature bins before calculating median curves.
- Compare median log10_sigma0 as well as median sigma0 because sigma0 spans many orders of magnitude.
