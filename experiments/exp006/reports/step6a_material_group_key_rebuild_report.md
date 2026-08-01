# Step6A Material Group Key Rebuild Report

## Summary

- input_file: experiments\exp006\data\processed\step5a_validation_rows_with_splits.parquet
- input_rows: 97086
- used_step3_metadata: True
- used_step0_metadata: False
- existing material_group_key unique count: 1
- existing material_group_key unknown fraction: 1.0
- formula_raw missing fraction: 0.0008034114084419999
- material_name_raw missing fraction: 0.0008034114084419999
- material_family_raw missing fraction: 1.0
- formula_parse_status counts: {'ok': 96074, 'low_confidence': 554, 'failed': 458}
- elapsed_seconds: 110.87

## Candidate Key Summary

| material_key_variant | unique_group_count | unknown_row_count | unknown_row_fraction | unknown_sample_count | row_count | sample_count | paper_count | median_rows_per_group | max_rows_per_group | median_samples_per_group | max_samples_per_group | top20_groups_by_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existing_clean | 1 | 97086 | 1.0 | 16013 | 97086 | 16013 | 4362 | 97086.0 | 97086 | 16013.0 | 16013 | {'unknown_material_group': 97086} |
| formula_system | 2972 | 458 | 0.00471746698803123 | 56 | 97086 | 16013 | 4362 | 11.0 | 2471 | 3.0 | 337 | {'system::Bi-Te': 2471, 'system::Bi-Sb': 1991, 'system::Bi-Sb-Te': 1740, 'system::Cu-Se': 1285, 'system::Ca-Co-O': 1199, 'system::Bi-Se-Te': 1142, 'system::Sb-Zn': 986, 'system::Co-Sb': 809, 'system::Bi-Sb-Sn': 654, 'system::Sb-Te': 619, 'system::C': 547, 'system::Bi-Se': 546, 'system::Sn-Te': 540, 'system::Ge-Te': 511, 'system::Ag-Se': 490, 'system::Al-O-Zn': 473, 'unknown_material_group': 458, 'system::Bi-Co-O-Sr': 456, 'system::Bi-Pb-Sb': 450, 'system::Bi-Pb-Te': 434} |
| broad_family | 16 | 458 | 0.00471746698803123 | 56 | 97086 | 16013 | 4362 | 2816.5 | 31682 | 591.5 | 4507 | {'broad::other_formula_system': 31682, 'broad::oxide': 17715, 'broad::selenide': 8815, 'broad::BiTe_like': 8807, 'broad::CoSb_skutterudite_like': 6278, 'broad::SbTe_like': 4993, 'broad::sulfide': 4544, 'broad::BiSbTe_tetradymite_like': 3283, 'broad::PbTe_like': 2350, 'broad::telluride': 2147, 'broad::SnTe_like': 2069, 'broad::SiGe_like': 1443, 'broad::Mg2SiSn_like': 1388, 'broad::GeTe_like': 1113, 'unknown_material_group': 458, 'broad::half_heusler': 1} |
| hybrid_v1 | 2972 | 458 | 0.00471746698803123 | 56 | 97086 | 16013 | 4362 | 11.0 | 2471 | 3.0 | 337 | {'system::Bi-Te': 2471, 'system::Bi-Sb': 1991, 'system::Bi-Sb-Te': 1740, 'system::Cu-Se': 1285, 'system::Ca-Co-O': 1199, 'system::Bi-Se-Te': 1142, 'system::Sb-Zn': 986, 'system::Co-Sb': 809, 'system::Bi-Sb-Sn': 654, 'system::Sb-Te': 619, 'system::C': 547, 'system::Bi-Se': 546, 'system::Sn-Te': 540, 'system::Ge-Te': 511, 'system::Ag-Se': 490, 'system::Al-O-Zn': 473, 'unknown_material_group': 458, 'system::Bi-Co-O-Sr': 456, 'system::Bi-Pb-Sb': 450, 'system::Bi-Pb-Te': 434} |
| hybrid_v2_broad_first | 16 | 458 | 0.00471746698803123 | 56 | 97086 | 16013 | 4362 | 2816.5 | 31682 | 591.5 | 4507 | {'broad::other_formula_system': 31682, 'broad::oxide': 17715, 'broad::selenide': 8815, 'broad::BiTe_like': 8807, 'broad::CoSb_skutterudite_like': 6278, 'broad::SbTe_like': 4993, 'broad::sulfide': 4544, 'broad::BiSbTe_tetradymite_like': 3283, 'broad::PbTe_like': 2350, 'broad::telluride': 2147, 'broad::SnTe_like': 2069, 'broad::SiGe_like': 1443, 'broad::Mg2SiSn_like': 1388, 'broad::GeTe_like': 1113, 'unknown_material_group': 458, 'broad::half_heusler': 1} |
| formula_system_collapsed | 694 | 458 | 0.00471746698803123 | 56 | 97086 | 16013 | 4362 | 63.0 | 9207 | 8.0 | 1814 | {'broad::other_formula_system': 9207, 'broad::oxide': 4659, 'system::Bi-Te': 2471, 'system::Bi-Sb': 1991, 'broad::sulfide': 1768, 'system::Bi-Sb-Te': 1740, 'broad::selenide': 1619, 'broad::CoSb_skutterudite_like': 1289, 'system::Cu-Se': 1285, 'system::Ca-Co-O': 1199, 'system::Bi-Se-Te': 1142, 'broad::telluride': 1004, 'system::Sb-Zn': 986, 'broad::BiTe_like': 968, 'broad::SbTe_like': 851, 'broad::PbTe_like': 829, 'system::Co-Sb': 809, 'system::Bi-Sb-Sn': 654, 'system::Sb-Te': 619, 'system::C': 547} |
| hybrid_v1_collapsed | 694 | 458 | 0.00471746698803123 | 56 | 97086 | 16013 | 4362 | 63.0 | 9207 | 8.0 | 1814 | {'broad::other_formula_system': 9207, 'broad::oxide': 4659, 'system::Bi-Te': 2471, 'system::Bi-Sb': 1991, 'broad::sulfide': 1768, 'system::Bi-Sb-Te': 1740, 'broad::selenide': 1619, 'broad::CoSb_skutterudite_like': 1289, 'system::Cu-Se': 1285, 'system::Ca-Co-O': 1199, 'system::Bi-Se-Te': 1142, 'broad::telluride': 1004, 'system::Sb-Zn': 986, 'broad::BiTe_like': 968, 'broad::SbTe_like': 851, 'broad::PbTe_like': 829, 'system::Co-Sb': 809, 'system::Bi-Sb-Sn': 654, 'system::Sb-Te': 619, 'system::C': 547} |

## Top Groups

| material_key_variant | material_group_key_value | row_count | sample_count | paper_count | carrier_type_values | T_min_K | T_max_K | formula_raw_examples | material_name_raw_examples | material_family_raw_examples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_family | broad::other_formula_system | 31682 | 4507 | 1312 | p | n | 0.04092544 | 1426.798 | Si99.24B0.76 | Si99.14B0.86 | CaZn2Sb2 | Ca0.75Yb0.25Zn2Sb2 | Ca0.5Yb0.5Zn2Sb2 | Si99.24B0.76 | Si99.14B0.86 | CaZn2Sb2 | Ca0.75Yb0.25Zn2Sb2 | Ca0.5Yb0.5Zn2Sb2 |  |
| broad_family | broad::oxide | 17715 | 3072 | 920 | n | p | 1.193243 | 1336.666 | TiO1.75 | TiO1.80 | TiO1.90 | Ca3Co4O9 | Ca2.8Ag0.15Lu0.05Co4O9 | TiO1.75 | TiO1.80 | TiO1.90 | Ca3Co4O9 | Ca2.8Ag0.15Lu0.05Co4O9 |  |
| broad_family | broad::selenide | 8815 | 1164 | 341 | p | n | 1.103835 | 1470.757 | Cu1.95Ag0.05SnSe3 | Cu1.9Ag0.1SnSe3 | Cu 2SnSe3 | Cu1.85Ag0.15Sn0.95In0.05Se3 | Cu1.85Ag0.15Sn0.9In0.1Se3 | Cu1.95Ag0.05SnSe3 | Cu1.9Ag0.1SnSe3 | Cu 2SnSe3 | Cu1.85Ag0.15Sn0.95In0.05Se3 | Cu1.85Ag0.15Sn0.9In0.1Se3 |  |
| broad_family | broad::BiTe_like | 8807 | 1255 | 375 | n | p | 1.530612 | 1115.167 | Bi2Te3 | Bi2Te2.9Se0.1 | Bi2Te2.3Se0.7 | Bi2Te1.7Se1.3 | Bi2Te1.1Se1.9 | Bi2Te3 | Bi2Te2.9Se0.1 | Bi2Te2.3Se0.7 | Bi2Te1.7Se1.3 | Bi2Te1.1Se1.9 |  |
| broad_family | broad::CoSb_skutterudite_like | 6278 | 1453 | 418 | p | n | 0.7832898 | 1091.654 | CoSb3 | Ce0.25Co3.95Cr0.05Sb12 | NbCoSb | Nb0.85CoSb | Nb0.84CoSb | CoSb3 | Ce0.25Co3.95Cr0.05Sb12 | NbCoSb | Nb0.85CoSb | Nb0.84CoSb |  |
| broad_family | broad::SbTe_like | 4993 | 985 | 293 | p | n | 1.0 | 1229.792 | Ag6.52Sb6.52Ge34.96Te50Dy2 | Ag6.52Sb6.52Ge35.96Te50Dy1 | Ag6.52Sb6.52Ge36.96Te49Dy1 | Ag6.52Sb6.52Ge36.96Te50 | Pb0.988Sb0.012Te | Ag6.52Sb6.52Ge34.96Te50Dy2 | Ag6.52Sb6.52Ge35.96Te50Dy1 | Ag6.52Sb6.52Ge36.96Te49Dy1 | Ag6.52Sb6.52Ge36.96Te50 | Pb0.988Sb0.012Te |  |
| broad_family | broad::sulfide | 4544 | 735 | 228 | p | n | 1.369863 | 1210.138 | CSA doped Polyaniline | MnBi4S7 | FeBi4S7 | Cu3P0.9Ge0.1S4 | Cu3P0.8Ge0.2S4 | CSA doped Polyaniline | MnBi4S7 | FeBi4S7 | Cu3P0.9Ge0.1S4 | Cu3P0.8Ge0.2S4 |  |
| broad_family | broad::BiSbTe_tetradymite_like | 3283 | 514 | 182 | p | n | 1.346547 | 776.19 | Bi0.5Sb1.5Te3 | Bi1.95In0.05Sb0.33Te3 | Bi1.9In0.1Sb0.67Te3 | Bi1.8In0.2Sb0.13Te3 | Bi1.7In0.3Sb0.2Te3 | Bi0.5Sb1.5Te3 | Bi1.95In0.05Sb0.33Te3 | Bi1.9In0.1Sb0.67Te3 | Bi1.8In0.2Sb0.13Te3 | Bi1.7In0.3Sb0.2Te3 |  |
| broad_family | broad::PbTe_like | 2350 | 669 | 174 | p | n | 11.17647 | 1074.757 | Pb1Te1 | Pb1Te0.7Se0.3 | (Pb0.9906La0.0094Te)0.9457(Ag2Te)0.0543 | (Pb0.9814La0.0186Te)0.9462(Ag2Te)0.0538 | (Pb0.9636La0.0364Te)0.945(Ag2Te)0.055 | Pb1Te1 | Pb1Te0.7Se0.3 | (Pb0.9906La0.0094Te)0.9457(Ag2Te)0.0543 | (Pb0.9814La0.0186Te)0.9462(Ag2Te)0.0538 | (Pb0.9636La0.0364Te)0.945(Ag2Te)0.055 |  |
| broad_family | broad::telluride | 2147 | 345 | 119 | n | p | 0.01 | 1824.224 | PtTe2 | Cu2S0.50Te0.50 | Cu2S0.54Te0.46 | In0.997Te | BaCu5.7Se0.6Te6.4 | PtTe2 | Cu2S0.50Te0.50 | Cu2S0.54Te0.46 | In0.997Te | BaCu5.7Se0.6Te6.4 |  |
| broad_family | broad::SnTe_like | 2069 | 421 | 105 | p | n | 5.150215 | 1663.82 | Sn0.86Mn0.14Te(Cu2Te)0 | Sn0.86Mn0.14Te(Cu2Te)0.01 | Sn0.86Mn0.14Te(Cu2Te)0.03 | Sn0.86Mn0.14Te(Cu2Te)0.04 | Sn0.86Mn0.14Te(Cu2Te)0.05 | Sn0.86Mn0.14Te(Cu2Te)0 | Sn0.86Mn0.14Te(Cu2Te)0.01 | Sn0.86Mn0.14Te(Cu2Te)0.03 | Sn0.86Mn0.14Te(Cu2Te)0.04 | Sn0.86Mn0.14Te(Cu2Te)0.05 |  |
| broad_family | broad::SiGe_like | 1443 | 224 | 80 | n | p | 0.5026065 | 969.1852 | Fe0.241Co0.063Si0.686Ge0.06Cu0.013P0.05 | Fe0.241Co0.063Si0.686Ge0.06Cu0.007P0.03Sb0.02 | Si0.7Ge0.3(WSi2)0.02 | Si0.8Ge0.2 | Ba8.0Cu4.7Ge35.1Si6.3 | Fe0.241Co0.063Si0.686Ge0.06Cu0.013P0.05 | Fe0.241Co0.063Si0.686Ge0.06Cu0.007P0.03Sb0.02 | Si0.7Ge0.3(WSi2)0.02 | Si0.8Ge0.2 | Ba8.0Cu4.7Ge35.1Si6.3 |  |
| broad_family | broad::Mg2SiSn_like | 1388 | 421 | 132 | n | p | 3.012203 | 922.939 | Mg2Si0.4Sn0.58Sb0.02 | Mg2Si0.4Sn0.48Sb0.12 | Mg2Si0.4Sn0.33Sb0.27 | Mg2Si0.4Sn0.22Sb0.38 | Mg2.16Si0.45Sn0.537Sb0.013 | Mg2Si0.4Sn0.58Sb0.02 | Mg2Si0.4Sn0.48Sb0.12 | Mg2Si0.4Sn0.33Sb0.27 | Mg2Si0.4Sn0.22Sb0.38 | Mg2.16Si0.45Sn0.537Sb0.013 |  |
| broad_family | broad::GeTe_like | 1113 | 191 | 56 | p | n | 2.724796 | 1500.862 | GeTe | Ge0.98Ta0.02Te | Ge0.95Ta0.05Te | Ge0.99Ta0.01Te1 | Ge0.97Ta0.03Te1 | GeTe | Ge0.98Ta0.02Te | Ge0.95Ta0.05Te | Ge0.99Ta0.01Te1 | Ge0.97Ta0.03Te1 |  |
| broad_family | unknown_material_group | 458 | 56 | 19 | p | n | 48.69 | 533.5084 | poly[2,5-bis(3-dodecylthiophen-2-yl)selenophene] /phosphomolybdic acid/(trifluoromethylsulfonyl)imide | undefined | poly(3,4-ethylenedioxythiophene):poly(styrenesulfonate) | poly(3,4-ethylenedioxythiophene)- block-poly(ethylene glycol)/singlewalled carbon nanotubes | poly(3,4-ethylenedioxythiophene) : polystyrene sulfonic acid | poly[2,5-bis(3-dodecylthiophen-2-yl)selenophene] /phosphomolybdic acid/(trifluoromethylsulfonyl)imide | undefined | poly(3,4-ethylenedioxythiophene):poly(styrenesulfonate) | poly(3,4-ethylenedioxythiophene)- block-poly(ethylene glycol)/singlewalled carbon nanotubes | poly(3,4-ethylenedioxythiophene) : polystyrene sulfonic acid |  |
| broad_family | broad::half_heusler | 1 | 1 | 1 | n | 324.454 | 324.454 | half-Heusler | half-Heusler |  |
| existing_clean | unknown_material_group | 97086 | 16013 | 4362 | p | n | 0.01 | 1824.224 | CSA doped Polyaniline | Si99.24B0.76 | Si99.14B0.86 | TiO1.75 | TiO1.80 | CSA doped Polyaniline | Si99.24B0.76 | Si99.14B0.86 | TiO1.75 | TiO1.80 |  |
| formula_system | system::Bi-Te | 2471 | 337 | 161 | n | p | 2.642906 | 1115.167 | Bi2Te3 | Bi0.46Te0.54 | Bi2Te2.90 | Bi2Te2.99 | Bi2Te2.92 | Bi2Te3 | Bi0.46Te0.54 | Bi2Te2.90 | Bi2Te2.99 | Bi2Te2.92 |  |
| formula_system | system::Bi-Sb | 1991 | 109 | 42 | n | p | 1.961609 | 529.7795 | Bi90Sb10 | Bi0.88Sb0.12 | Bi88Sb12 | Bi85Sb15 | Bi78Sb22 | Bi90Sb10 | Bi0.88Sb0.12 | Bi88Sb12 | Bi85Sb15 | Bi78Sb22 |  |
| formula_system | system::Bi-Sb-Te | 1740 | 292 | 119 | p | n | 3.448276 | 674.7012 | Bi0.5Sb1.5Te3 | Bi0.4Sb1.6Te3 | (Sb0.84Bi0.16)2Te3 | (Sb0.8Bi0.2)2Te3 | (Sb0.76Bi0.24)2Te3 | Bi0.5Sb1.5Te3 | Bi0.4Sb1.6Te3 | (Sb0.84Bi0.16)2Te3 | (Sb0.8Bi0.2)2Te3 | (Sb0.76Bi0.24)2Te3 |  |

## Preflight Coverage Default-like Settings

| material_key_variant | split_scheme | coverage_fraction | material_group_count_train | material_group_count_test |
| --- | --- | --- | --- | --- |
| formula_system | sample_holdout | 0.5755471380471381 | 2658 | 1434 |
| formula_system | paper_holdout | 0.4108861764414012 | 2480 | 884 |
| broad_family | sample_holdout | 0.997895622895623 | 16 | 15 |
| broad_family | paper_holdout | 0.9983129899771758 | 16 | 15 |
| hybrid_v1 | sample_holdout | 0.5755471380471381 | 2658 | 1434 |
| hybrid_v1 | paper_holdout | 0.4108861764414012 | 2480 | 884 |
| hybrid_v2_broad_first | sample_holdout | 0.997895622895623 | 16 | 15 |
| hybrid_v2_broad_first | paper_holdout | 0.9983129899771758 | 16 | 15 |
| formula_system_collapsed | sample_holdout | 0.788510101010101 | 691 | 569 |
| formula_system_collapsed | paper_holdout | 0.6465714002183189 | 652 | 358 |
| hybrid_v1_collapsed | sample_holdout | 0.788510101010101 | 691 | 569 |
| hybrid_v1_collapsed | paper_holdout | 0.6465714002183189 | 652 | 358 |

## Recommended Variants

| rank | material_key_variant | reason | unique_group_count | unknown_row_fraction | representative_coverage_fraction_sample_holdout | representative_coverage_fraction_paper_holdout | comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | broad_family | ranked by representative coverage, unknown fraction, and group count | 16 | 0.00471746698803123 | 0.997895622895623 | 0.9983129899771758 | candidate for Step5B rerun |
| 2 | hybrid_v2_broad_first | ranked by representative coverage, unknown fraction, and group count | 16 | 0.00471746698803123 | 0.997895622895623 | 0.9983129899771758 | candidate for Step5B rerun |
| 3 | formula_system_collapsed | ranked by representative coverage, unknown fraction, and group count | 694 | 0.00471746698803123 | 0.788510101010101 | 0.6465714002183189 | collapsed variants may improve coverage but reduce chemical specificity |
| 4 | hybrid_v1_collapsed | ranked by representative coverage, unknown fraction, and group count | 694 | 0.00471746698803123 | 0.788510101010101 | 0.6465714002183189 | collapsed variants may improve coverage but reduce chemical specificity |
| 5 | formula_system | ranked by representative coverage, unknown fraction, and group count | 2972 | 0.00471746698803123 | 0.5755471380471381 | 0.4108861764414012 | candidate for Step5B rerun |
| 6 | hybrid_v1 | ranked by representative coverage, unknown fraction, and group count | 2972 | 0.00471746698803123 | 0.5755471380471381 | 0.4108861764414012 | candidate for Step5B rerun |

## Sanity Check

- candidate_rows_match_input: True
- row_id_unique: True
- material_group_key_original_exists: True
- candidate_columns_exist: True
- candidate_columns_not_missing: True
- formula_parse_status_allowed: True
- parsed_element_count_nonnegative: True
- formula_system_key_not_empty: True
- six_variant_files_created: True
- variant_file_rows_match_input: True
- variant_material_group_key_replaced: True
- variant_original_preserved: True
- variant_splits_preserved: True
- sample_holdout_no_leakage: True
- paper_holdout_no_leakage: True
- preflight_coverage_range: True
- preflight_nonempty: True
- candidate_unique_group_count_gt_1: True
- formula_or_hybrid_unknown_not_all: True
- report_exists: True

## Notes

- WARNING: none
- broad_family is a heuristic grouping for validation only, not a final material taxonomy.
- formula_system can over-split doped or noisy formula strings and can include regex false positives.
- collapsed variants improve bin coverage by mapping rare formula systems to broad families, but lose specificity.
- Next: choose one recommended variant, rerun Step5B with that variant input, then rerun Step5C and Step5D-1.
