# Step1 Carrier Classification Report

## Summary

- input_file: C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step0_te_analysis_base.parquet
- input_rows: 172891
- output_rows: 172891
- zero_threshold_uV: 1
- carrier_type row counts: {'p': 95504, 'n': 74859, 'unknown_near_zero': 2528}
- carrier_type row shares percent: {'p': 55.239, 'n': 43.298, 'unknown_near_zero': 1.462}
- p data points: 95504
- n data points: 74859
- unknown_near_zero data points: 2528
- is_usable_for_eta == True rows: 170363
- is_conservative_main_analysis == True rows: 163148
- sample_group_id count: 23640
- p_only samples: 12833
- n_only samples: 10108
- mixed_sign samples: 636
- unknown_only samples: 63
- S_uV_per_K summary: min=-7.90015e+06, max=1.0295e+06, median=19.3013
- S_abs_uV_per_K summary: min=0, max=7.90015e+06, median=131.943
- elapsed_seconds: 21.35

## Parquet Status

- step1_te_carrier_classified.parquet: saved
- step1_eta_input_candidates.parquet: saved
- step1_conservative_main_candidates.parquet: saved

## Mixed Sign Sample Examples

| sample_group_id | paper_id | sample_id | sample_key | formula_raw | n_points_sample | n_p_points_sample | n_n_points_sample | n_unknown_points_sample | T_min_K | T_max_K | S_min_uV_per_K | S_max_uV_per_K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5959::174 | 5959 | 174 | 10.1002_adfm.200901905__sample_174 | Pb1Te1 | 9 | 6 | 3 | 0 | 319.7628 | 667.1937 | -111.2792 | 408.6657 |
| 33875::41898 | 33875 | 41898 | 10.1002_adfm.201900615__sample_41898 | CH3NH3PbI3 | 9 | 8 | 1 | 0 | 303.1588 | 343.5231 | -7.050066 | 24.65488 |
| 37713::44898 | 37713 | 44898 | 10.1002_aenm.202100883__sample_44898 | (Ag0.2Cu0.8)2S0.7Se0.3 | 8 | 6 | 2 | 0 | 305.9524 | 799.3556 | -34.63169 | 493.44540000000006 |
| 29129::38945 | 29129 | 38945 | 10.1002_andp.201900340__sample_38945 | Bi6Cu2Se3.8Br0.2O6 | 9 | 5 | 4 | 0 | 297.9522 | 873.3788 | -219.4271 | 521.5442 |
| 29129::38950 | 29129 | 38950 | 10.1002_andp.201900340__sample_38950 | Bi6Cu2Se3.8Br0.2O6 | 8 | 5 | 3 | 0 | 297.9522 | 873.3788 | -144.70729999999998 | 494.14689999999996 |
| 195::6599 | 195 | 6599 | 10.1002_anie.201505517__sample_6599 | Cu1.17Fe0.83S2 | 26 | 13 | 12 | 1 | 7.340624 | 328.471 | -132.4418 | 186.1166 |
| 18659::11795 | 18659 | 11795 | 10.1002_anie.201601420__sample_11795 | SnSe | 9 | 2 | 7 | 0 | 327.0883 | 550.5891 | -149.47889999999998 | 82.06273 |
| 926::19621 | 926 | 19621 | 10.1002_ejic.201100864__sample_19621 | FeSb2 | 3 | 1 | 2 | 0 | 66.99029 | 222.3301 | -41.76707 | 14.13655 |
| 4515::4699 | 4515 | 4699 | 10.1002_pssa.200925491__sample_4699 | CoSb3 | 3 | 1 | 2 | 0 | 281.7617 | 723.7872 | -661.2179 | 114.5105 |
| 15282::2492 | 15282 | 2492 | 10.1002_pssa.201532642__sample_2492 | CoSb3 | 6 | 1 | 5 | 0 | 322.449 | 823.0769 | -99.62611000000001 | 529.0323 |
| 4201::23319 | 4201 | 23319 | 10.1007_978-94-007-4984-9_3__sample_23319 | Yb2Fe12P7 | 28 | 8 | 20 | 0 | 3.009499 | 298.5905 | -8.844541 | 56.17643 |
| 4204::5805 | 4204 | 5805 | 10.1007_978-94-007-4984-9_9__sample_5805 | LaPt4Ge12 | 15 | 7 | 3 | 5 | 5.307596 | 278.9645 | -1.864803 | 3.021407 |
| 11266::23429 | 11266 | 23429 | 10.1007_bf00683310__sample_23429 | CeB6 | 114 | 39 | 74 | 1 | 0.126506 | 4.40324 | -11.240509999999999 | 48.00133 |
| 11266::23430 | 11266 | 23430 | 10.1007_bf00683310__sample_23430 | CeB6 | 31 | 21 | 5 | 5 | 0.1536145 | 4.201807 | -2.483554 | 48.73605 |
| 2925::21829 | 2925 | 21829 | 10.1007_bf01168943__sample_21829 | Si76.41Ge1.91Ga1.96P2.53 | 2 | 1 | 1 | 0 | 820.2786 | 1014.792 | -220.0626 | 265.26320000000004 |
| 2925::21832 | 2925 | 21832 | 10.1007_bf01168943__sample_21832 | Si76.41Ge1.91Ga1.96P2.53 | 3 | 1 | 2 | 0 | 357.2482 | 1118.375 | -268.8308 | 138.9474 |
| 7705::24479 | 7705 | 24479 | 10.1007_bf02562809__sample_24479 | (Bi1.6Pb0.4)Sr2Ca3(Cu0.875Si0.125)4O12 | 6 | 5 | 1 | 0 | 100.9577 | 302.111 | -1.223844 | 17.778589999999998 |
| 7707::17862 | 7707 | 17862 | 10.1007_bf02570272__sample_17862 | CeOs2 | 8 | 5 | 2 | 1 | 2.569928 | 290.0459 | -1.683853 | 9.38884 |
| 227::3344 | 227 | 3344 | 10.1007_s00339-009-5329-5__sample_3344 | Co4Sb12 | 4 | 3 | 1 | 0 | 286.1635 | 771.6981 | -24.133680000000002 | 164.0934 |
| 6591::11543 | 6591 | 11543 | 10.1007_s00339-011-6431-z__sample_11543 | FeSb2 | 40 | 12 | 28 | 0 | 4.166667 | 298.6111 | -87.11655999999999 | 22.69939 |

## Material Family Carrier Overview

| material_family_raw | n | p | unknown_near_zero |
| --- | --- | --- | --- |
| unknown | 74859 | 95504 | 2528 |

## Match Method Carrier Overview

| match_method | n | p | unknown_near_zero |
| --- | --- | --- | --- |
| exact | 2296 | 2128 | 14 |
| nearest | 72563 | 93376 | 2514 |

## Sigma Source Carrier Overview

| sigma_source | n | p | unknown_near_zero |
| --- | --- | --- | --- |
| conductivity_direct | 34680 | 46109 | 370 |
| resistivity_converted | 40179 | 49395 | 2158 |

## Carrier Counts By Material Family

| material_family_raw | carrier_type | row_count | sample_count | paper_count | T_min_K | T_max_K | S_min_uV_per_K | S_max_uV_per_K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unknown | p | 95504 | 13469 | 3343 | 0.01 | 1824.224 | 1.001167 | 1029504.0 |
| unknown | n | 74859 | 10744 | 2687 | 0.126506 | 1426.798 | -7900150.0 | -1.0075310000000002 |
| unknown | unknown_near_zero | 2528 | 681 | 374 | 0.01416633 | 1068.267 | -1.0 | 1.0 |

## Sanity Check

- input_rows_equal_output_rows: True
- row_id_unique: True
- S_V_per_K_finite: True
- S_uV_per_K_finite: True
- S_uV_per_K_consistent: True
- carrier_type_not_missing: True
- carrier_type_allowed: True
- is_usable_for_eta_rule: True
- sample_has_sign_change_rule: True
- is_conservative_main_analysis_rule: True
- eta_candidates_carrier_type_p_or_n_only: True
- conservative_candidates_no_sign_change: True
- T_K_positive_finite: True
- sigma_S_per_m_positive_finite: True

## Warnings And Step2 Notes

- WARNING: none
- Step2 では eta 計算に進む前に、mixed_sign sample を主解析から除くか感度分析に回すかを判断してください。
- unknown_near_zero は S がしきい値近傍のため、eta 入力候補からは外しています。
- eta、F0_eta、sigma0 はこの Step1 では計算していません。
