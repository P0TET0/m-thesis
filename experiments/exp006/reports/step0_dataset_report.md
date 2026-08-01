# Step0 Dataset Report

## Summary

- 読み込んだファイル数: 1
- 読み込んだ総行数: 111564
- Seebeck係数として認識したデータ数: 684250
- electrical conductivityとして認識したデータ数: 256428
- electrical resistivityとして認識したデータ数: 437430
- 最終的に S と sigma が対応づいた行数: 172891
- exact / nearest / interpolated の行数: {'exact': 4438, 'nearest': 168453}
- sigma_source の内訳: {'conductivity_direct': 81159, 'resistivity_converted': 91732}
- 温度範囲: 0.01 to 1824.22
- S の範囲: -7.90015 to 1.0295 V/K
- sigma の範囲: 2.84217e-12 to 2.67963e+20 S/m
- paper_id 数: 5272
- sample_id 数: 23640
- formula_raw 数: 15302
- material_family_raw 数: 1
- parquet 保存: 成功

## 欠損値の概要

- 欠損値はありません。

## 除外理由

- nonpositive_or_invalid_resistivity: 3984
- conflicting_duplicate_property_point: 3323
- nonpositive_or_invalid_conductivity: 1129
- invalid_seebeck_value_or_unit: 720
- invalid_temperature: 102
- conflicting_duplicate_matched_row: 44

## 重複候補

- 重複候補の件数: 4633

## Sanity Check

- required_columns_present: True
- missing_required_columns: []
- T_K_positive_finite: True
- S_V_per_K_finite: True
- sigma_S_per_m_positive_finite: True
- rho_ohm_m_positive_when_present: True
- S_uV_per_K_consistent: True
- T_delta_within_tolerance_for_non_interpolated: True
- row_id_unique: True
- row_id_duplicate_count: 0
- conductivity_direct_positive: True
- resistivity_converted_consistent: True

## 読み込んだ表と推定スキーマ

### data\output\starrydata2_step5_core_properties\property_core_curves_step5.csv
- rows: 111564
- columns: 83
- column_names: curve_id, curve_key, sample_key, SID, SID_curve, SID_sample, DOI, DOI_curve, DOI_sample, sample_id, sample_id_curve, sample_id_sample, paper_title, year, composition, composition_curve, composition_sample, material_system, n_or_p, n_or_p_basis, sintering_method, sintering_checked, record_checked, figure_id, prop_x, property_family, property, prop_y_canonical, prop_y, prop_y_raw, unit, unit_x, unit_y, n_points, n_points_step5, x_min, x_max, y_min, y_max, x_values_json, y_values_json, unit_check_note, unit_check_note_step5, xy_length_check, property_step5, property_step5_source, is_target_property_step5, property_filter_reason, merge_status, is_candidate_sample, is_learning_candidate, learning_candidate_reason, has_sigma_or_rho, has_seebeck, has_kappa_or_zt, is_target_property_for_relaxation, is_core_property, is_seebeck_curve, is_electrical_conductivity_curve, is_electrical_resistivity_curve, is_thermal_conductivity_curve, is_zt_curve, zt_unit_is_dimensionless, zt_unit_needs_check, zt_unit_check_status_x, is_relaxation_fit_candidate_x, is_accuracy_check_candidate_x, is_extended_transport_candidate_x, has_core_property, zt_curve_count, zt_unit_values, zt_unit_issue_count, zt_unit_all_dimensionless, has_seebeck_curve, has_electrical_conductivity_curve, has_electrical_resistivity_curve, has_any_electrical_transport_curve, has_thermal_conductivity_curve, has_zt_curve, is_relaxation_fit_candidate_y, is_accuracy_check_candidate_y, is_extended_transport_candidate_y, zt_unit_check_status_y
- property_like: prop_x, property_family, property, prop_y_canonical, prop_y, prop_y_raw, property_step5, property_step5_source, is_target_property_step5, property_filter_reason, has_sigma_or_rho, has_seebeck, is_target_property_for_relaxation, is_core_property, is_seebeck_curve, is_electrical_conductivity_curve, is_electrical_resistivity_curve, is_thermal_conductivity_curve, has_core_property, has_seebeck_curve, has_electrical_conductivity_curve, has_electrical_resistivity_curve, has_thermal_conductivity_curve
- temperature_like: n/a
- unit_like: unit, unit_x, unit_y, unit_check_note, unit_check_note_step5, zt_unit_is_dimensionless, zt_unit_needs_check, zt_unit_check_status_x, zt_unit_values, zt_unit_issue_count, zt_unit_all_dimensionless, zt_unit_check_status_y
- sample_id_like: sample_key, SID, SID_curve, SID_sample, DOI_sample, sample_id, sample_id_curve, sample_id_sample, composition_sample, is_candidate_sample
- paper_id_or_doi_like: SID, SID_curve, SID_sample, DOI, DOI_curve, DOI_sample, paper_title
- formula_or_material_like: composition, composition_curve, composition_sample, material_system
- unit_candidates: -, - | K^(-1), 0, 1, 2, False, K, K^(-1), S*m^(-1), True, V*K^(-1), V/K, W*m^(-1)*K^(-1), kg*m**2/A/s**3, needs_check, needs_check | ZT unit is not dimensionless; check later, not_applicable, not_zt, ohm*m, ohm*m^-1
- property_candidates: Calculated Electrical Conductivity, Calculated Seebeck coefficient, Conductivity, Contact resistivity, Electrical conductivity, Electrical resistivity, False, Figure of merit, Figure of merit Z, Resistance, Resistivity, S, Seebeck coefficient, Temperature, Thermal conductivity, Thermoelectric power, Thermopower, True, ZT, conductivity

| column | missing_rate | representative_values |
| --- | ---: | --- |
| curve_id | 0.000 | 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84409__figure_56246__prop_Electrical_conductivity__sid_49885__row_197902; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84409__figure_56248__prop_Seebeck_coefficient__sid_49885__row_197909; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84409__figure_56250__prop_ZT__sid_49885__row_197914; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84410__figure_56246__prop_Electrical_conductivity__sid_49885__row_197903; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84410__figure_56248__prop_Seebeck_coefficient__sid_49885__row_197908 |
| curve_key | 0.000 | 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84409__figure_56246__prop_Electrical_conductivity__sid_49885__row_197902; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84409__figure_56248__prop_Seebeck_coefficient__sid_49885__row_197909; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84409__figure_56250__prop_ZT__sid_49885__row_197914; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84410__figure_56246__prop_Electrical_conductivity__sid_49885__row_197903; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84410__figure_56248__prop_Seebeck_coefficient__sid_49885__row_197908 |
| sample_key | 0.000 | 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84409; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84410; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84415; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84416; 10.1002_1439-2054_20010301_286_3_139_aid-mame139_3.0.co_2-f__sample_84417 |
| SID | 0.000 | 49885; 122; 123; 124; 30874 |
| SID_curve | 0.000 | 49885; 122; 123; 124; 30874 |
| SID_sample | 0.000 | 49885; 122; 123; 124; 30874 |
| DOI | 0.000 | 10.1002/1439-2054(20010301)286:3<139::aid-mame139>3.0.co;2-f; 10.1002/adem.200500043; 10.1002/adem.201200233; 10.1002/adem.201400183; 10.1002/adem.202000816 |
| DOI_curve | 0.000 | 10.1002/1439-2054(20010301)286:3<139::aid-mame139>3.0.co;2-f; 10.1002/adem.200500043; 10.1002/adem.201200233; 10.1002/adem.201400183; 10.1002/adem.202000816 |
| DOI_sample | 0.000 | 10.1002/1439-2054(20010301)286:3<139::aid-mame139>3.0.co;2-f; 10.1002/adem.200500043; 10.1002/adem.201200233; 10.1002/adem.201400183; 10.1002/adem.202000816 |
| sample_id | 0.000 | 84409; 84410; 84415; 84416; 84417 |
| sample_id_curve | 0.000 | 84409; 84410; 84415; 84416; 84417 |
| sample_id_sample | 0.000 | 84409; 84410; 84415; 84416; 84417 |
| paper_title | 1.000 |  |
| year | 1.000 |  |
| composition | 0.000 | CSA doped Polyaniline; Ca3Co4O9; LaCoO3; La0.7Ca0.3CoO3; La0.6Ca0.4CoO3 |
| composition_curve | 0.000 | CSA doped Polyaniline; Ca3Co4O9; LaCoO3; La0.7Ca0.3CoO3; La0.6Ca0.4CoO3 |
| composition_sample | 0.000 | CSA doped Polyaniline; Ca3Co4O9; LaCoO3; La0.7Ca0.3CoO3; La0.6Ca0.4CoO3 |
| material_system | 0.000 | unknown |
| n_or_p | 0.000 | p; unknown; n; mixed |
| n_or_p_basis | 0.000 | Seebeck sign: positive; Seebeck sign: unknown; Seebeck sign: negative; Seebeck sign: mixed; Seebeck sign: zero |
| sintering_method | 0.000 | unknown |
| sintering_checked | 0.000 | no |
| record_checked | 0.000 | no |
| figure_id | 0.000 | 56246; 56248; 56250; 56255; 20342 |
| prop_x | 0.000 | Temperature |
| property_family | 0.000 | electrical_conductivity; seebeck; zt; electrical_resistivity; thermal_conductivity |
| property | 0.000 | Electrical conductivity; Seebeck coefficient; ZT; Electrical resistivity; Thermal conductivity |
| prop_y_canonical | 0.000 | Electrical conductivity; Seebeck coefficient; ZT; Electrical resistivity; Thermal conductivity |
| prop_y | 0.000 | Electrical conductivity; Seebeck coefficient; ZT; Electrical resistivity; Thermal conductivity |
| prop_y_raw | 0.000 | Electrical conductivity; Seebeck coefficient; ZT; Electrical resistivity; Thermal conductivity |
| unit | 0.000 | ohm^(-1)*m^(-1); V*K^(-1); -; ohm*m; W*m^(-1)*K^(-1) |
| unit_x | 0.000 | K |
| unit_y | 0.000 | ohm^(-1)*m^(-1); V*K^(-1); -; ohm*m; W*m^(-1)*K^(-1) |
| n_points | 0.000 | 13; 12; 3; 8; 5 |
| n_points_step5 | 0.000 | 13; 12; 3; 8; 5 |
| x_min | 0.000 | 299.3; 302.7; 306.1; 300.0; 304.0 |
| x_max | 0.000 | 428.2; 428.7; 429.4; 427.5; 430.0 |
| y_min | 0.000 | 15070.0; 1.624e-05; 0.01081; 4170.0; 1.422e-05 |
| y_max | 0.000 | 25960.0; 3.789e-05; 0.02887; 8744.0; 2.798e-05 |
| x_values_json | 0.000 | [299.3,308.2,318.4,327.3,338.2,348.4,360.7,375.0,388.0,403.6,414.5,420.7,428.2]; [302.7,317.3,324.7,336.7,346.0,359.3,373.3,388.0,401.3,410.0,420.0,428.7]; [306.1,361.4,429.4]; [300.0,306.1,318.4,327.3,336.8,347.7,358.6,375.7,386.6,400.9,411.1,418.6,427.5]; [304.0,317.3,326.0,337.3,347.3,359.3,372.7,390.0,403.3,410.7,420.0,430.0] |
| y_values_json | 0.000 | [25960.0,25560.0,23540.0,22200.0,21120.0,20310.0,19640.0,19240.0,18570.0,17350.0,16950.0,15870.0,15070.0]; [2.009e-05,2.009e-05,1.624e-05,1.972e-05,2.394e-05,2.651e-05,2.688e-05,3.11e-05,3.239e-05,3.202e-05,3.459e-05,3.789e-05]; [0.01081,0.0171,0.02887]; [8744.0,7937.0,7937.0,6861.0,6457.0,6592.0,5785.0,5785.0,5112.0,4978.0,4305.0,4709.0,4170.0]; [1.569e-05,1.679e-05,1.422e-05,1.661e-05,1.936e-05,2.138e-05,2.138e-05,2.284e-05,2.394e-05,2.633e-05,2.56e-05,2.798e-05] |
| unit_check_note | 0.000 | not_zt; ok; needs_check |
| unit_check_note_step5 | 0.000 | not_zt; ok; needs_check | ZT unit is not dimensionless; check later |
| xy_length_check | 0.000 | ok |
| property_step5 | 0.000 | Electrical conductivity; Seebeck coefficient; ZT; Electrical resistivity; Thermal conductivity |
| property_step5_source | 0.000 | property_family; property |
| is_target_property_step5 | 0.000 | True |
| property_filter_reason | 0.000 | target property |
| merge_status | 0.000 | matched |
| is_candidate_sample | 0.000 | True; False |
| is_learning_candidate | 0.000 | True; False |
| learning_candidate_reason | 1.000 |  |
| has_sigma_or_rho | 0.000 | True; False |
| has_seebeck | 0.000 | True; False |
| has_kappa_or_zt | 0.000 | True; False |
| is_target_property_for_relaxation | 0.000 | True; False |
| is_core_property | 0.000 | True; False |
| is_seebeck_curve | 0.000 | False; True |
| is_electrical_conductivity_curve | 0.000 | True; False |
| is_electrical_resistivity_curve | 0.000 | False; True |
| is_thermal_conductivity_curve | 0.000 | False; True |
| is_zt_curve | 0.000 | False; True |
| zt_unit_is_dimensionless | 0.000 | False; True |
| zt_unit_needs_check | 0.000 | False; True |
| zt_unit_check_status_x | 0.000 | not_zt; ok; needs_check |
| is_relaxation_fit_candidate_x | 0.000 | True; False |
| is_accuracy_check_candidate_x | 0.000 | False; True |
| is_extended_transport_candidate_x | 0.000 | False; True |
| has_core_property | 0.000 | True; False |
| zt_curve_count | 0.000 | 1; 0; 2; 3; 4 |
| zt_unit_values | 0.336 | -; K^(-1); - | K^(-1) |
| zt_unit_issue_count | 0.000 | 0; 1; 2 |
| zt_unit_all_dimensionless | 0.000 | True; False |
| has_seebeck_curve | 0.000 | True; False |
| has_electrical_conductivity_curve | 0.000 | True; False |
| has_electrical_resistivity_curve | 0.000 | False; True |
| has_any_electrical_transport_curve | 0.000 | True; False |
| has_thermal_conductivity_curve | 0.000 | False; True |
| has_zt_curve | 0.000 | True; False |
| is_relaxation_fit_candidate_y | 0.000 | True; False |
| is_accuracy_check_candidate_y | 0.000 | False; True |
| is_extended_transport_candidate_y | 0.000 | False; True |
| zt_unit_check_status_y | 0.000 | ok; not_applicable; needs_check |


## 人間に確認すべき事項

- 現時点で必須の確認事項はありません。
