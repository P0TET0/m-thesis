# Step9B: 25 K sigma0_ref(T) vs Step31 SS2026 C(T) overlay

## Inputs and detected Step31 schema

- exp006 input: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9a_25k_bin_broad_family\step5b_train_reference_curve_bins.parquet`
- exp005 Step31 C(T) CSV: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step31_reclassified_ct_curves\ct_curves_material_system_np_split_step31.csv`
- adopted C(T) column: `C_T_sigma_median_S_per_m_step31`
- adopted temperature column: `temp_bin_center_K`
- adopted material column: `material_system_reclass_step31`
- adopted carrier column: `n_or_p_reclass_step31`

## Scope and results

- target material groups (15): broad::BiSbTe_tetradymite_like, broad::PbTe_like, broad::SnTe_like, broad::BiTe_like, broad::SbTe_like, broad::SiGe_like, broad::Mg2SiSn_like, broad::CoSb_skutterudite_like, broad::GeTe_like, broad::oxide, broad::sulfide, broad::selenide, broad::telluride, broad::other_formula_system, unknown_material_group
- target carrier_type: p, n
- material-group x carrier combinations with both curves: 30
- overlay figures: 30
- log-ratio figures: 30
- combinations skipped because one side was missing: 0
- overall median log10(sigma0_ref_25k / C_T_step31): -0.094112

### Median ratio by material group

| material_group_key | median log10 ratio |
|---|---:|
| broad::BiSbTe_tetradymite_like | -0.201335 |
| broad::PbTe_like | 0.225284 |
| broad::SnTe_like | -0.123251 |
| broad::BiTe_like | -0.333465 |
| broad::SbTe_like | -0.219717 |
| broad::SiGe_like | 0.0139929 |
| broad::Mg2SiSn_like | 0.745417 |
| broad::CoSb_skutterudite_like | -0.148887 |
| broad::GeTe_like | -0.432625 |
| broad::oxide | -0.177326 |
| broad::sulfide | 0.0614205 |
| broad::selenide | -0.0195491 |
| broad::telluride | 0.181935 |
| broad::other_formula_system | -0.232052 |
| unknown_material_group | -0.190567 |

### Missing-side warnings

None.

## Figure index

| figure_id | group | carrier | type | PNG | PDF |
|---|---|---|---|---|---|
| step9b_001 | broad::BiSbTe_tetradymite_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_002 | broad::BiSbTe_tetradymite_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_003 | broad::BiSbTe_tetradymite_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_004 | broad::BiSbTe_tetradymite_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_005 | broad::PbTe_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_006 | broad::PbTe_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_007 | broad::PbTe_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_008 | broad::PbTe_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_009 | broad::SnTe_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_010 | broad::SnTe_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_011 | broad::SnTe_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_012 | broad::SnTe_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_013 | broad::BiTe_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiTe_like_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiTe_like_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_014 | broad::BiTe_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiTe_like_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiTe_like_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_015 | broad::BiTe_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiTe_like_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiTe_like_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_016 | broad::BiTe_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiTe_like_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiTe_like_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_017 | broad::SbTe_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SbTe_like_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SbTe_like_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_018 | broad::SbTe_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SbTe_like_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SbTe_like_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_019 | broad::SbTe_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SbTe_like_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SbTe_like_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_020 | broad::SbTe_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SbTe_like_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SbTe_like_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_021 | broad::SiGe_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SiGe_like_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SiGe_like_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_022 | broad::SiGe_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SiGe_like_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SiGe_like_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_023 | broad::SiGe_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SiGe_like_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SiGe_like_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_024 | broad::SiGe_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SiGe_like_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SiGe_like_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_025 | broad::Mg2SiSn_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_Mg2SiSn_like_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_Mg2SiSn_like_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_026 | broad::Mg2SiSn_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_Mg2SiSn_like_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_Mg2SiSn_like_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_027 | broad::Mg2SiSn_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_Mg2SiSn_like_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_Mg2SiSn_like_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_028 | broad::Mg2SiSn_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_Mg2SiSn_like_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_Mg2SiSn_like_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_029 | broad::CoSb_skutterudite_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_CoSb_skutterudite_like_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_CoSb_skutterudite_like_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_030 | broad::CoSb_skutterudite_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_CoSb_skutterudite_like_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_CoSb_skutterudite_like_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_031 | broad::CoSb_skutterudite_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_CoSb_skutterudite_like_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_CoSb_skutterudite_like_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_032 | broad::CoSb_skutterudite_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_CoSb_skutterudite_like_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_CoSb_skutterudite_like_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_033 | broad::GeTe_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_GeTe_like_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_GeTe_like_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_034 | broad::GeTe_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_GeTe_like_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_GeTe_like_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_035 | broad::GeTe_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_GeTe_like_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_GeTe_like_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_036 | broad::GeTe_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_GeTe_like_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_GeTe_like_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_037 | broad::oxide | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_oxide_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_oxide_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_038 | broad::oxide | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_oxide_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_oxide_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_039 | broad::oxide | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_oxide_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_oxide_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_040 | broad::oxide | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_oxide_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_oxide_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_041 | broad::sulfide | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_sulfide_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_sulfide_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_042 | broad::sulfide | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_sulfide_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_sulfide_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_043 | broad::sulfide | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_sulfide_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_sulfide_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_044 | broad::sulfide | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_sulfide_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_sulfide_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_045 | broad::selenide | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_selenide_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_selenide_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_046 | broad::selenide | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_selenide_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_selenide_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_047 | broad::selenide | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_selenide_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_selenide_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_048 | broad::selenide | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_selenide_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_selenide_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_049 | broad::telluride | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_telluride_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_telluride_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_050 | broad::telluride | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_telluride_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_telluride_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_051 | broad::telluride | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_telluride_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_telluride_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_052 | broad::telluride | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_telluride_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_telluride_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_053 | broad::other_formula_system | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_other_formula_system_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_other_formula_system_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_054 | broad::other_formula_system | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_other_formula_system_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_other_formula_system_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_055 | broad::other_formula_system | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_other_formula_system_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_other_formula_system_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_056 | broad::other_formula_system | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_other_formula_system_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_other_formula_system_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_057 | unknown_material_group | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\unknown_material_group_p_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\unknown_material_group_p_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_058 | unknown_material_group | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\unknown_material_group_p_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\unknown_material_group_p_p_log_ratio_sigma0ref25k_over_step31ct.pdf` |
| step9b_059 | unknown_material_group | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\unknown_material_group_n_25k_sigma0ref_vs_step31_ct_overlay.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\unknown_material_group_n_25k_sigma0ref_vs_step31_ct_overlay.pdf` |
| step9b_060 | unknown_material_group | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\unknown_material_group_n_log_ratio_sigma0ref25k_over_step31ct.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\unknown_material_group_n_n_log_ratio_sigma0ref25k_over_step31ct.pdf` |

## Unmatched material labels

None.

## CSV outputs

- parse: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_step31_ct_parse_summary.csv`
- sigma: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_25k_sigma0ref_curves.csv`
- ct: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_step31_ct_curves_normalized.csv`
- comparison: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_curve_comparison_table.csv`
- summary: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_summary_by_group_carrier.csv`
- figures: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_figure_index.csv`
- mapping: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_material_mapping.csv`
- unmatched: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_unmatched_material_labels.csv`

## Interpretation and safeguards

- The exp006 curve is the 25 K temperature-bin broad_family sigma0_ref(T) from Step9A.
- The exp005 curve is SS2026 C(T) from Step31, not the old Step12 prefactor C(T).
- No new sigma_pred was calculated.
- Starrydata2 raw data was not read.
- Existing Step0-Step9A and Step31 outputs were not overwritten.
- Input file metadata remained unchanged during this run: True.
