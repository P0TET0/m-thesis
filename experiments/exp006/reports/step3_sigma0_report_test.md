# Step3 Sigma0 Report

## Summary

- input_file: experiments\exp006\data\processed\step2_eta_ge1_candidates.parquet
- input_rows: 1000
- max_rows: 1000
- sigma0_calc_status counts: {'ok': 1000}
- is_valid_sigma0 == True rows: 1000
- is_conservative_valid_sigma0 == True rows: 987
- eta summary: min=1.00208, median=2.53381, max=207.236
- F0_eta summary: min=1.31479, median=2.61017, max=207.236
- sigma_S_per_m summary: min=7.56579e-06, median=61210.8, max=1.46533e+06
- sigma0_S_per_m summary: min=1.88146e-07, median=20714.7, max=163525
- log10_sigma0_S_per_m summary: min=-6.7255, median=4.31628, max=5.21359
- sigma0_reconstruction_log_error summary: min=-9.64327e-17, median=0, max=9.64327e-17
- sigma0_reconstruction_log_error max_abs: 9.64327e-17
- sample summary rows: 258
- material family summary rows: 2
- elapsed_seconds: 0.85

## Parquet Status

- step3_sigma0_calculated_test.parquet: saved
- step3_sigma0_valid_test.parquet: saved
- step3_conservative_sigma0_valid_test.parquet: saved

## Valid Sigma0 By Carrier Type

| carrier_type | valid_sigma0_count |
| --- | --- |
| n | 481 |
| p | 519 |

## Valid Sigma0 By Material Family Top 10

| material_family_raw | carrier_type | row_count | valid_sigma0_count | T_min_K | T_max_K | eta_min | eta_median | eta_max | S_abs_median_uV_per_K | sigma_median_S_per_m | log10_sigma_median_S_per_m | sigma0_median_S_per_m | log10_sigma0_median_S_per_m | sigma0_min_S_per_m | sigma0_max_S_per_m | log10_sigma0_min_S_per_m | log10_sigma0_max_S_per_m | paper_count | sample_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unknown | p | 519 | 519 | 10.6618 | 1233.419 | 1.0020848479640618 | 2.526231758757481 | 207.23630120671191 | 97.28146 | 50981.63 | 4.707413716748957 | 20864.636508111696 | 4.31941082308097 | 2.628767115742609e-07 | 151260.4447714762 | -6.580247886631493 | 5.179725373076576 | 38 | 133 |
| unknown | n | 481 | 481 | 13.89363 | 913.9761 | 1.0033964086261165 | 2.534032231279692 | 93.25633558609215 | 97.08 | 78109.8052019568 | 4.892705554720008 | 20290.68927337838 | 4.30729680023888 | 1.881462358436912e-07 | 163525.3447357928 | -6.725504465989701 | 5.213585073362448 | 32 | 125 |

## Valid Sigma0 By Sigma Source

| sigma_source | valid_sigma0_count |
| --- | --- |
| conductivity_direct | 529 |
| resistivity_converted | 471 |

## Valid Sigma0 By Match Method

| match_method | valid_sigma0_count |
| --- | --- |
| exact | 62 |
| nearest | 938 |

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
