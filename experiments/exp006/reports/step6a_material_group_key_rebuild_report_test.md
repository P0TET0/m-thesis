# Step6A Material Group Key Rebuild Report

## Summary

- input_file: experiments\exp006\data\processed\step5a_validation_rows_with_splits.parquet
- input_rows: 5000
- used_step3_metadata: True
- used_step0_metadata: False
- existing material_group_key unique count: 1
- existing material_group_key unknown fraction: 1.0
- formula_raw missing fraction: 0.0156
- material_name_raw missing fraction: 0.0156
- material_family_raw missing fraction: 1.0
- formula_parse_status counts: {'ok': 4818, 'failed': 171, 'low_confidence': 11}
- elapsed_seconds: 6.84

## Candidate Key Summary

| material_key_variant | unique_group_count | unknown_row_count | unknown_row_fraction | unknown_sample_count | row_count | sample_count | paper_count | median_rows_per_group | max_rows_per_group | median_samples_per_group | max_samples_per_group | top20_groups_by_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existing_clean | 1 | 5000 | 1.0 | 855 | 5000 | 855 | 232 | 5000.0 | 5000 | 855.0 | 855 | {'unknown_material_group': 5000} |
| formula_system | 276 | 171 | 0.0342 | 14 | 5000 | 855 | 232 | 8.5 | 324 | 2.0 | 26 | {'system::Cu-Yb': 324, 'system::Bi-Te': 183, 'system::Al-Ce-La': 176, 'unknown_material_group': 171, 'system::B-Ce': 154, 'system::Bi-Sb': 120, 'system::Bi-In-Sn-Te': 101, 'system::Bi-Ga-Sn-Te': 98, 'system::Bi-Sn-Te': 95, 'system::As-Cd': 85, 'system::Cu-Se-Ti': 80, 'system::Bi-Se-Te': 77, 'system::Bi-Sn-Te-Tl': 74, 'system::B-Ca-Cu-O-Pb-Sr': 70, 'system::Ba-C-Cu-Si': 66, 'system::Co-Sb': 60, 'system::Sb-Sn-Te-Zn': 59, 'system::Bi-Sb-Te': 55, 'system::Ba-Ca-Cu-La-O': 55, 'system::Ag-Pb-Sb-Se-Te': 54} |
| broad_family | 15 | 171 | 0.0342 | 14 | 5000 | 855 | 232 | 171.0 | 1726 | 48.0 | 210 | {'broad::other_formula_system': 1726, 'broad::BiTe_like': 722, 'broad::SbTe_like': 507, 'broad::oxide': 487, 'broad::selenide': 370, 'broad::CoSb_skutterudite_like': 248, 'broad::sulfide': 233, 'unknown_material_group': 171, 'broad::PbTe_like': 164, 'broad::SnTe_like': 144, 'broad::BiSbTe_tetradymite_like': 82, 'broad::Mg2SiSn_like': 60, 'broad::GeTe_like': 31, 'broad::SiGe_like': 28, 'broad::telluride': 27} |
| hybrid_v1 | 276 | 171 | 0.0342 | 14 | 5000 | 855 | 232 | 8.5 | 324 | 2.0 | 26 | {'system::Cu-Yb': 324, 'system::Bi-Te': 183, 'system::Al-Ce-La': 176, 'unknown_material_group': 171, 'system::B-Ce': 154, 'system::Bi-Sb': 120, 'system::Bi-In-Sn-Te': 101, 'system::Bi-Ga-Sn-Te': 98, 'system::Bi-Sn-Te': 95, 'system::As-Cd': 85, 'system::Cu-Se-Ti': 80, 'system::Bi-Se-Te': 77, 'system::Bi-Sn-Te-Tl': 74, 'system::B-Ca-Cu-O-Pb-Sr': 70, 'system::Ba-C-Cu-Si': 66, 'system::Co-Sb': 60, 'system::Sb-Sn-Te-Zn': 59, 'system::Bi-Sb-Te': 55, 'system::Ba-Ca-Cu-La-O': 55, 'system::Ag-Pb-Sb-Se-Te': 54} |
| hybrid_v2_broad_first | 15 | 171 | 0.0342 | 14 | 5000 | 855 | 232 | 171.0 | 1726 | 48.0 | 210 | {'broad::other_formula_system': 1726, 'broad::BiTe_like': 722, 'broad::SbTe_like': 507, 'broad::oxide': 487, 'broad::selenide': 370, 'broad::CoSb_skutterudite_like': 248, 'broad::sulfide': 233, 'unknown_material_group': 171, 'broad::PbTe_like': 164, 'broad::SnTe_like': 144, 'broad::BiSbTe_tetradymite_like': 82, 'broad::Mg2SiSn_like': 60, 'broad::GeTe_like': 31, 'broad::SiGe_like': 28, 'broad::telluride': 27} |
| formula_system_collapsed | 49 | 171 | 0.0342 | 14 | 5000 | 855 | 232 | 60.0 | 718 | 9.0 | 137 | {'broad::other_formula_system': 718, 'system::Cu-Yb': 324, 'broad::oxide': 279, 'broad::selenide': 218, 'broad::sulfide': 197, 'system::Bi-Te': 183, 'broad::SbTe_like': 182, 'system::Al-Ce-La': 176, 'unknown_material_group': 171, 'broad::PbTe_like': 164, 'system::B-Ce': 154, 'broad::CoSb_skutterudite_like': 152, 'system::Bi-Sb': 120, 'system::Bi-In-Sn-Te': 101, 'system::Bi-Ga-Sn-Te': 98, 'system::Bi-Sn-Te': 95, 'broad::BiTe_like': 94, 'system::As-Cd': 85, 'system::Cu-Se-Ti': 80, 'system::Bi-Se-Te': 77} |
| hybrid_v1_collapsed | 49 | 171 | 0.0342 | 14 | 5000 | 855 | 232 | 60.0 | 718 | 9.0 | 137 | {'broad::other_formula_system': 718, 'system::Cu-Yb': 324, 'broad::oxide': 279, 'broad::selenide': 218, 'broad::sulfide': 197, 'system::Bi-Te': 183, 'broad::SbTe_like': 182, 'system::Al-Ce-La': 176, 'unknown_material_group': 171, 'broad::PbTe_like': 164, 'system::B-Ce': 154, 'broad::CoSb_skutterudite_like': 152, 'system::Bi-Sb': 120, 'system::Bi-In-Sn-Te': 101, 'system::Bi-Ga-Sn-Te': 98, 'system::Bi-Sn-Te': 95, 'broad::BiTe_like': 94, 'system::As-Cd': 85, 'system::Cu-Se-Ti': 80, 'system::Bi-Se-Te': 77} |

## Top Groups

| material_key_variant | material_group_key_value | row_count | sample_count | paper_count | carrier_type_values | T_min_K | T_max_K | formula_raw_examples | material_name_raw_examples | material_family_raw_examples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_family | broad::other_formula_system | 1726 | 210 | 61 | p | n | 0.05407739 | 1233.419 | Si99.24B0.76 | Si99.14B0.86 | CaZn2Sb2 | Ca0.75Yb0.25Zn2Sb2 | Ca0.5Yb0.5Zn2Sb2 | Si99.24B0.76 | Si99.14B0.86 | CaZn2Sb2 | Ca0.75Yb0.25Zn2Sb2 | Ca0.5Yb0.5Zn2Sb2 |  |
| broad_family | broad::BiTe_like | 722 | 99 | 27 | n | p | 38.6 | 667.2093 | Bi2Te3 | Bi2Te2.9Se0.1 | Bi2Te2.3Se0.7 | Bi2Te1.7Se1.3 | Bi2Te1.1Se1.9 | Bi2Te3 | Bi2Te2.9Se0.1 | Bi2Te2.3Se0.7 | Bi2Te1.7Se1.3 | Bi2Te1.1Se1.9 |  |
| broad_family | broad::SbTe_like | 507 | 103 | 23 | p | n | 80.52285 | 882.0833 | Ag6.52Sb6.52Ge34.96Te50Dy2 | Ag6.52Sb6.52Ge35.96Te50Dy1 | Ag6.52Sb6.52Ge36.96Te49Dy1 | Ag6.52Sb6.52Ge36.96Te50 | Pb0.988Sb0.012Te | Ag6.52Sb6.52Ge34.96Te50Dy2 | Ag6.52Sb6.52Ge35.96Te50Dy1 | Ag6.52Sb6.52Ge36.96Te49Dy1 | Ag6.52Sb6.52Ge36.96Te50 | Pb0.988Sb0.012Te |  |
| broad_family | broad::oxide | 487 | 100 | 27 | n | p | 8.319644 | 973.15 | TiO1.75 | TiO1.80 | TiO1.90 | Ca3Co4O9 | Ca2.8Ag0.15Lu0.05Co4O9 | TiO1.75 | TiO1.80 | TiO1.90 | Ca3Co4O9 | Ca2.8Ag0.15Lu0.05Co4O9 |  |
| broad_family | broad::selenide | 370 | 67 | 24 | p | n | 10.17198 | 823.1103 | Cu1.95Ag0.05SnSe3 | Cu1.9Ag0.1SnSe3 | Cu 2SnSe3 | Cu1.85Ag0.15Sn0.95In0.05Se3 | Cu1.85Ag0.15Sn0.9In0.1Se3 | Cu1.95Ag0.05SnSe3 | Cu1.9Ag0.1SnSe3 | Cu 2SnSe3 | Cu1.85Ag0.15Sn0.95In0.05Se3 | Cu1.85Ag0.15Sn0.9In0.1Se3 |  |
| broad_family | broad::CoSb_skutterudite_like | 248 | 64 | 23 | p | n | 42.58173 | 973.5053 | CoSb3 | Ce0.25Co3.95Cr0.05Sb12 | NbCoSb | Nb0.85CoSb | Nb0.84CoSb | CoSb3 | Ce0.25Co3.95Cr0.05Sb12 | NbCoSb | Nb0.85CoSb | Nb0.84CoSb |  |
| broad_family | broad::sulfide | 233 | 48 | 17 | p | n | 7.340624 | 823.3702 | CSA doped Polyaniline | MnBi4S7 | FeBi4S7 | Cu3P0.9Ge0.1S4 | Cu3P0.8Ge0.2S4 | CSA doped Polyaniline | MnBi4S7 | FeBi4S7 | Cu3P0.9Ge0.1S4 | Cu3P0.8Ge0.2S4 |  |
| broad_family | unknown_material_group | 171 | 14 | 4 | p | 48.69 | 368.45 | poly[2,5-bis(3-dodecylthiophen-2-yl)selenophene] /phosphomolybdic acid/(trifluoromethylsulfonyl)imide | undefined | poly[2,5-bis(3-dodecylthiophen-2-yl)selenophene] /phosphomolybdic acid/(trifluoromethylsulfonyl)imide | undefined |  |
| broad_family | broad::PbTe_like | 164 | 54 | 14 | p | n | 296.6398 | 673.8776 | Pb1Te1 | Pb1Te0.7Se0.3 | (Pb0.9906La0.0094Te)0.9457(Ag2Te)0.0543 | (Pb0.9814La0.0186Te)0.9462(Ag2Te)0.0538 | (Pb0.9636La0.0364Te)0.945(Ag2Te)0.055 | Pb1Te1 | Pb1Te0.7Se0.3 | (Pb0.9906La0.0094Te)0.9457(Ag2Te)0.0543 | (Pb0.9814La0.0186Te)0.9462(Ag2Te)0.0538 | (Pb0.9636La0.0364Te)0.945(Ag2Te)0.055 |  |
| broad_family | broad::SnTe_like | 144 | 36 | 7 | p | 140.2118 | 873.6634 | Sn0.86Mn0.14Te(Cu2Te)0 | Sn0.86Mn0.14Te(Cu2Te)0.01 | Sn0.86Mn0.14Te(Cu2Te)0.03 | Sn0.86Mn0.14Te(Cu2Te)0.04 | Sn0.86Mn0.14Te(Cu2Te)0.05 | Sn0.86Mn0.14Te(Cu2Te)0 | Sn0.86Mn0.14Te(Cu2Te)0.01 | Sn0.86Mn0.14Te(Cu2Te)0.03 | Sn0.86Mn0.14Te(Cu2Te)0.04 | Sn0.86Mn0.14Te(Cu2Te)0.05 |  |
| broad_family | broad::BiSbTe_tetradymite_like | 82 | 18 | 8 | p | n | 30.01564 | 574.7774 | Bi0.5Sb1.5Te3 | Bi1.95In0.05Sb0.33Te3 | Bi1.9In0.1Sb0.67Te3 | Bi1.8In0.2Sb0.13Te3 | Bi1.7In0.3Sb0.2Te3 | Bi0.5Sb1.5Te3 | Bi1.95In0.05Sb0.33Te3 | Bi1.9In0.1Sb0.67Te3 | Bi1.8In0.2Sb0.13Te3 | Bi1.7In0.3Sb0.2Te3 |  |
| broad_family | broad::Mg2SiSn_like | 60 | 15 | 6 | n | 3.760923 | 713.1445 | Mg2Si0.4Sn0.58Sb0.02 | Mg2Si0.4Sn0.48Sb0.12 | Mg2Si0.4Sn0.33Sb0.27 | Mg2Si0.4Sn0.22Sb0.38 | Mg2.16Si0.45Sn0.537Sb0.013 | Mg2Si0.4Sn0.58Sb0.02 | Mg2Si0.4Sn0.48Sb0.12 | Mg2Si0.4Sn0.33Sb0.27 | Mg2Si0.4Sn0.22Sb0.38 | Mg2.16Si0.45Sn0.537Sb0.013 |  |
| broad_family | broad::GeTe_like | 31 | 9 | 4 | p | 298.7654 | 704.172 | GeTe | Ge0.98Ta0.02Te | Ge0.95Ta0.05Te | Ge0.99Ta0.01Te1 | Ge0.97Ta0.03Te1 | GeTe | Ge0.98Ta0.02Te | Ge0.95Ta0.05Te | Ge0.99Ta0.01Te1 | Ge0.97Ta0.03Te1 |  |
| broad_family | broad::SiGe_like | 28 | 12 | 5 | n | p | 292.3742 | 875.5294 | Fe0.241Co0.063Si0.686Ge0.06Cu0.013P0.05 | Fe0.241Co0.063Si0.686Ge0.06Cu0.007P0.03Sb0.02 | Si0.7Ge0.3(WSi2)0.02 | Si0.8Ge0.2 | Ba8.0Cu4.7Ge35.1Si6.3 | Fe0.241Co0.063Si0.686Ge0.06Cu0.013P0.05 | Fe0.241Co0.063Si0.686Ge0.06Cu0.007P0.03Sb0.02 | Si0.7Ge0.3(WSi2)0.02 | Si0.8Ge0.2 | Ba8.0Cu4.7Ge35.1Si6.3 |  |
| broad_family | broad::telluride | 27 | 6 | 5 | n | p | 138.9611 | 664.0854 | PtTe2 | Cu2S0.50Te0.50 | Cu2S0.54Te0.46 | In0.997Te | BaCu5.7Se0.6Te6.4 | PtTe2 | Cu2S0.50Te0.50 | Cu2S0.54Te0.46 | In0.997Te | BaCu5.7Se0.6Te6.4 |  |
| existing_clean | unknown_material_group | 5000 | 855 | 232 | p | n | 0.05407739 | 1233.419 | CSA doped Polyaniline | Si99.24B0.76 | Si99.14B0.86 | TiO1.75 | TiO1.80 | CSA doped Polyaniline | Si99.24B0.76 | Si99.14B0.86 | TiO1.75 | TiO1.80 |  |
| formula_system | system::Cu-Yb | 324 | 13 | 1 | n | 1.103748 | 198.5201 | YbCu4.5 | YbCu4.5 |  |
| formula_system | system::Bi-Te | 183 | 25 | 13 | n | p | 38.6 | 600.4446 | Bi2Te3 | Bi0.46Te0.54 | Bi2Te2.90 | Bi2Te2.99 | Bi2Te2.92 | Bi2Te3 | Bi0.46Te0.54 | Bi2Te2.90 | Bi2Te2.99 | Bi2Te2.92 |  |
| formula_system | system::Al-Ce-La | 176 | 6 | 1 | p | 0.4009652 | 7.009004 | (La0.9936Ce0.0064)Al2 | (La0.9901Ce0.0099)Al2 | (La0.985Ce0.015)Al2 | (La0.9936Ce0.0064)Al2 | (La0.9901Ce0.0099)Al2 | (La0.985Ce0.015)Al2 |  |
| formula_system | unknown_material_group | 171 | 14 | 4 | p | 48.69 | 368.45 | poly[2,5-bis(3-dodecylthiophen-2-yl)selenophene] /phosphomolybdic acid/(trifluoromethylsulfonyl)imide | undefined | poly[2,5-bis(3-dodecylthiophen-2-yl)selenophene] /phosphomolybdic acid/(trifluoromethylsulfonyl)imide | undefined |  |

## Preflight Coverage Default-like Settings

| material_key_variant | split_scheme | coverage_fraction | material_group_count_train | material_group_count_test |
| --- | --- | --- | --- | --- |
| formula_system | sample_holdout | 0.5220667384284177 | 242 | 119 |
| formula_system | paper_holdout | 0.12116182572614108 | 233 | 59 |
| broad_family | sample_holdout | 0.9429494079655544 | 15 | 15 |
| broad_family | paper_holdout | 0.8813278008298755 | 15 | 12 |
| hybrid_v1 | sample_holdout | 0.5220667384284177 | 242 | 119 |
| hybrid_v1 | paper_holdout | 0.12116182572614108 | 233 | 59 |
| hybrid_v2_broad_first | sample_holdout | 0.9429494079655544 | 15 | 15 |
| hybrid_v2_broad_first | paper_holdout | 0.8813278008298755 | 15 | 12 |
| formula_system_collapsed | sample_holdout | 0.8320775026910656 | 49 | 41 |
| formula_system_collapsed | paper_holdout | 0.3800829875518672 | 44 | 26 |
| hybrid_v1_collapsed | sample_holdout | 0.8320775026910656 | 49 | 41 |
| hybrid_v1_collapsed | paper_holdout | 0.3800829875518672 | 44 | 26 |

## Recommended Variants

| rank | material_key_variant | reason | unique_group_count | unknown_row_fraction | representative_coverage_fraction_sample_holdout | representative_coverage_fraction_paper_holdout | comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | broad_family | ranked by representative coverage, unknown fraction, and group count | 15 | 0.0342 | 0.9429494079655544 | 0.8813278008298755 | candidate for Step5B rerun |
| 2 | hybrid_v2_broad_first | ranked by representative coverage, unknown fraction, and group count | 15 | 0.0342 | 0.9429494079655544 | 0.8813278008298755 | candidate for Step5B rerun |
| 3 | formula_system_collapsed | ranked by representative coverage, unknown fraction, and group count | 49 | 0.0342 | 0.8320775026910656 | 0.3800829875518672 | collapsed variants may improve coverage but reduce chemical specificity |
| 4 | hybrid_v1_collapsed | ranked by representative coverage, unknown fraction, and group count | 49 | 0.0342 | 0.8320775026910656 | 0.3800829875518672 | collapsed variants may improve coverage but reduce chemical specificity |
| 5 | formula_system | ranked by representative coverage, unknown fraction, and group count | 276 | 0.0342 | 0.5220667384284177 | 0.12116182572614108 | candidate for Step5B rerun |
| 6 | hybrid_v1 | ranked by representative coverage, unknown fraction, and group count | 276 | 0.0342 | 0.5220667384284177 | 0.12116182572614108 | candidate for Step5B rerun |

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
- report_exists: True

## Notes

- WARNING: none
- broad_family is a heuristic grouping for validation only, not a final material taxonomy.
- formula_system can over-split doped or noisy formula strings and can include regex false positives.
- collapsed variants improve bin coverage by mapping rare formula systems to broad families, but lose specificity.
- Next: choose one recommended variant, rerun Step5B with that variant input, then rerun Step5C and Step5D-1.
