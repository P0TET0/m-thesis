# Step5C Evaluation Metrics Report

## Summary

- input_file: experiments\exp006\data\processed\step9a_25k_bin_broad_family\step5b_test_predictions_valid.parquet
- input_rows: 602862
- evaluated_rows: 602862
- dropped_rows: 0
- config_count: 32
- metric_weighting: ['row_equal', 'sample_equal']
- min_eval_rows: 30
- min_eval_samples: 5
- metrics_by_config rows: 64
- metrics_by_carrier_type rows: 128
- metrics_by_material_family rows: 960
- metrics_by_temperature_bin rows: 3264
- metrics_by_eta_bin rows: 384
- metrics_by_reliability_level rows: 192
- elapsed_seconds: 1250.54

## Parquet Status

- step5c_metrics_by_config.parquet: saved

## Default Comparison

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | metric_weighting | n_rows | n_samples | n_papers | n_material_families | n_T_bins | mean_log10_error | median_log10_error | mae_log10 | rmse_log10 | std_log10_error | q05_log10_error | q25_log10_error | q75_log10_error | q95_log10_error | max_abs_log10_error | overprediction_fraction | underprediction_fraction | near_exact_fraction | factor_2_accuracy | factor_3_accuracy | factor_5_accuracy | factor_10_accuracy | median_abs_factor_error | mean_abs_factor_error_equiv | sigma_exp_median_S_per_m | sigma_pred_median_S_per_m | eta_median | S_abs_median_uV_per_K | T_min_K | T_max_K | train_sample_count_median | train_paper_count_median | is_reliable_eval_group | eval_group_reliability | test_rows | prediction_ok_rows | prediction_unavailable_rows | coverage_fraction | reference_bins_total | reference_bins_reliable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 18995 | 3191 | 2202 | 15 | 53 | 0.2886266723414163 | -0.015945935309112674 | 0.8472664690445615 | 1.4041717750093219 | 1.3742243041978612 | -1.2236915094229532 | -0.40130226589041706 | 0.6809633193786402 | 2.8649979579184026 | 14.89607457267981 | 0.49065543564095815 | 0.5093445643590419 | 0.05480389576204264 | 0.33271913661489866 | 0.4888128454856541 | 0.6285338246907081 | 0.7541458278494341 | 3.10728186097033 | 7.03503834682973 | 74037.76814628678 | 65594.96804303405 | 3.363443872380753 | 78.80000000000001 | 0.1536145 | 1468.911 | 755.0 | 405.0 | True | high | 19008 | 18995 | 13 | 0.9993160774410774 | 122 | 111 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 18883 | 3178 | 2193 | 15 | 50 | 0.24131625300854082 | 0.0017694685822311263 | 0.7184792716957925 | 1.2438614350158783 | 1.2202608702597717 | -1.1390292439059015 | -0.31604205599737367 | 0.5084260256464908 | 2.429565257742036 | 14.539199102855582 | 0.5014563363872266 | 0.4985436636127734 | 0.08838637928295293 | 0.4293809246412117 | 0.5718900598421861 | 0.6899327437377535 | 0.7854154530530106 | 2.40735888351726 | 5.229730047730456 | 74577.95 | 75413.915314739 | 3.363443872380753 | 78.80000000000001 | 0.1536145 | 1284.759 | 130.0 | 73.0 | True | high | 19008 | 18883 | 125 | 0.9934238215488216 | 1076 | 902 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 20152 | 3178 | 866 | 15 | 53 | 0.1773323804181979 | -0.06824110270297784 | 0.8713881626714 | 1.4747338364525409 | 1.4640694935081693 | -1.582160998883858 | -0.49158280299998813 | 0.5636537065945064 | 2.518595052150179 | 10.298180034427356 | 0.4625843588725685 | 0.5374156411274316 | 0.05443628423977769 | 0.3179833267169512 | 0.4708217546645494 | 0.6179039301310044 | 0.7470226280269948 | 3.296326594951641 | 7.436835286088087 | 81830.11466231802 | 63998.3580189714 | 3.7016634394884456 | 72.86373 | 0.01 | 1470.757 | 739.0 | 351.0 | True | high | 20154 | 20152 | 2 | 0.9999007641163044 | 122 | 111 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 19984 | 3167 | 865 | 15 | 49 | 0.1604080139791811 | -0.028831109784510204 | 0.7903355280173384 | 1.4048574956129911 | 1.395704599684252 | -1.4956240449526306 | -0.4078619927088162 | 0.45326836804767634 | 2.324138575439809 | 10.30255809312047 | 0.47738190552441956 | 0.5226180944755805 | 0.07280824659727782 | 0.38781024819855886 | 0.5403322658126501 | 0.6704363490792634 | 0.7710168134507606 | 2.6594914785554566 | 6.170715560095066 | 81864.305 | 75165.1236076899 | 3.6975180758831128 | 72.93200999999999 | 0.01 | 1237.266 | 142.0 | 71.0 | True | high | 20154 | 19984 | 170 | 0.9915649498858788 | 1071 | 896 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 3191 | 3191 | 2202 | 15 | 88 | 0.3004822718903958 | -0.0007507120060974409 | 0.7330412995245859 | 1.2338943398621802 | 1.1969355093365428 | -0.8314355651197146 | -0.3464791314161041 | 0.6464668893495509 | 2.5414836927038023 | 14.640003241475045 | 0.49984330930742715 | 0.5001566906925728 | 0.05766217486681291 | 0.3632090253838922 | 0.5490441867753055 | 0.6966468191789408 | 0.7975556251958633 | 2.7079633454526695 | 5.40805748785191 | 67583.3217878492 | 61845.13476655806 | 2.5985643357794 | 96.071305 | 1.530612 | 1451.689 | 833.0 | 435.0 | True | high | 19008 | 18995 | 13 | 0.9993160774410774 | 122 | 111 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 3178 | 3178 | 2193 | 15 | 86 | 0.24921090296825937 | 0.008364805553037917 | 0.6263785403148054 | 1.1207591327179862 | 1.0928727208953064 | -0.8302024180700476 | -0.2577082561451062 | 0.436101462644282 | 2.28011150123057 | 14.293319569770352 | 0.5100692259282568 | 0.48993077407174324 | 0.10383889238514789 | 0.47954688483322844 | 0.6365638766519823 | 0.7404027690371303 | 0.8281938325991189 | 2.1052599591263537 | 4.230371818290402 | 67802.13368614137 | 73422.3728390104 | 2.6000461727564392 | 96.03996749999999 | 1.530612 | 1098.185 | 124.5 | 70.5 | True | high | 19008 | 18883 | 125 | 0.9934238215488216 | 1076 | 902 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 3178 | 3178 | 866 | 15 | 83 | 0.2149074791383537 | -0.03127786572251855 | 0.7303555281068714 | 1.2184897659323455 | 1.199576959114452 | -1.0946686361099869 | -0.39255864220173003 | 0.5883495648089206 | 2.252279436904081 | 10.02728768444927 | 0.4830081812460667 | 0.5169918187539333 | 0.05726872246696035 | 0.3524229074889868 | 0.5254877281309 | 0.6897419760855884 | 0.8061674008810573 | 2.806832263811225 | 5.374716087049902 | 73726.035 | 60747.89864067713 | 2.680385303076341 | 93.9533675 | 0.32348 | 1431.07 | 845.0 | 396.5 | True | high | 20154 | 20152 | 2 | 0.9999007641163044 | 122 | 111 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 3167 | 3167 | 865 | 15 | 80 | 0.16450282454846435 | -0.017505985722089265 | 0.639266991458817 | 1.134825213375764 | 1.1230161648165597 | -1.0735157687118122 | -0.31179819366381645 | 0.3796105322040041 | 2.0353729568472563 | 10.035896147441832 | 0.48468582254499526 | 0.5153141774550047 | 0.08778023365961478 | 0.4591095674139564 | 0.6217240290495737 | 0.7350805178402273 | 0.8326491948215977 | 2.1854718966572175 | 4.357796958771682 | 74109.53685986035 | 71137.80037089212 | 2.6771339222018993 | 93.981255 | 0.32348 | 1174.012 | 128.5 | 65.0 | True | high | 20154 | 19984 | 170 | 0.9915649498858788 | 1071 | 896 |

## Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.7184792716957925 | 1.2438614350158783 | 0.0017694685822311263 | 0.4293809246412117 | 0.6899327437377535 | 0.7854154530530106 | 0.9934238215488216 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.6263785403148054 | 1.1207591327179862 | 0.008364805553037917 | 0.47954688483322844 | 0.7404027690371303 | 0.8281938325991189 | 0.9934238215488216 |

## Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.8472664690445615 | 1.4041717750093219 | -0.015945935309112674 | 0.33271913661489866 | 0.6285338246907081 | 0.7541458278494341 | 0.9993160774410774 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.7330412995245859 | 1.2338943398621802 | -0.0007507120060974409 | 0.3632090253838922 | 0.6966468191789408 | 0.7975556251958633 | 0.9993160774410774 |

## Paper Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.7903355280173384 | 1.4048574956129911 | -0.028831109784510204 | 0.38781024819855886 | 0.6704363490792634 | 0.7710168134507606 | 0.9915649498858788 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.639266991458817 | 1.134825213375764 | -0.017505985722089265 | 0.4591095674139564 | 0.7350805178402273 | 0.8326491948215977 | 0.9915649498858788 |

## Paper Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.8713881626714 | 1.4747338364525409 | -0.06824110270297784 | 0.3179833267169512 | 0.6179039301310044 | 0.7470226280269948 | 0.9999007641163044 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.7303555281068714 | 1.2184897659323455 | -0.03127786572251855 | 0.3524229074889868 | 0.6897419760855884 | 0.8061674008810573 | 0.9999007641163044 |

## Best Configs By MAE

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | 0.6424510599671687 | 1.0808698497668845 | 0.43997734994337484 | 0.8120611551528879 | 0.9956587923549642 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | 0.6428148962736085 | 1.075155715982414 | 0.431580139274189 | 0.8143576968804846 | 0.9958279303151604 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | 0.652024507225062 | 1.0864103220315748 | 0.4399592368227368 | 0.8077902960991904 | 0.9958279303151604 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | 0.6521922273999797 | 1.0925237301786905 | 0.44331823329558323 | 0.8062853907134768 | 0.9956587923549642 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__row_median | 0.7076325459314547 | 1.2212987992199367 | 0.4185308583214342 | 0.7947009360621926 | 0.9947916666666666 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | 0.7078181919824547 | 1.2349660909061084 | 0.42694487104803264 | 0.7934120637610549 | 0.9934238215488216 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 0.7184792716957925 | 1.2438614350158783 | 0.4293809246412117 | 0.7854154530530106 | 0.9934238215488216 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | 0.7198270611222395 | 1.237071567067784 | 0.42572320059231056 | 0.7861864720503464 | 0.9947916666666666 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | 0.7553646955033245 | 1.2166259776675876 | 0.346808750563825 | 0.7787550744248986 | 0.9999436206799346 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__row_median | 0.7564755493018847 | 1.2132593296304148 | 0.3410577356788453 | 0.7817997293640054 | 0.9999436206799346 |

## Best Configs By Factor 2 Accuracy

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | 0.6521922273999797 | 1.0925237301786905 | 0.44331823329558323 | 0.8062853907134768 | 0.9956587923549642 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | 0.6424510599671687 | 1.0808698497668845 | 0.43997734994337484 | 0.8120611551528879 | 0.9956587923549642 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | 0.652024507225062 | 1.0864103220315748 | 0.4399592368227368 | 0.8077902960991904 | 0.9958279303151604 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | 0.6428148962736085 | 1.075155715982414 | 0.431580139274189 | 0.8143576968804846 | 0.9958279303151604 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 0.7184792716957925 | 1.2438614350158783 | 0.4293809246412117 | 0.7854154530530106 | 0.9934238215488216 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | 0.7078181919824547 | 1.2349660909061084 | 0.42694487104803264 | 0.7934120637610549 | 0.9934238215488216 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | 0.7198270611222395 | 1.237071567067784 | 0.42572320059231056 | 0.7861864720503464 | 0.9947916666666666 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__row_median | 0.7076325459314547 | 1.2212987992199367 | 0.4185308583214342 | 0.7947009360621926 | 0.9947916666666666 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | 0.7578830053066202 | 1.3745490325041965 | 0.4017024944776682 | 0.7836862238025969 | 0.9917183158794616 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | 0.7637401843913898 | 1.3837556286649413 | 0.3993858089542589 | 0.7837939766176392 | 0.9917183158794616 |

## Comparison Notes

- split_scheme median mae_log10: {'paper_holdout': 0.743865665470572, 'sample_holdout': 0.7015098247961462}
- group_scheme median mae_log10: {'global': 0.7442029975139552, 'material_family': 0.6416032997437875}
- curve_method median mae_log10: {'row_median': 0.7174914973769357, 'sample_median': 0.7250912946145555}
- reference_source_subset median mae_log10: {'all_valid': 0.7234959319468282, 'conservative_valid': 0.7229952953593313}
- p/n median mae_log10: {'n': 0.7080689225152152, 'p': 0.7556365207379607}
- eta bin median mae_log10: {'[1, 2)': 0.5271470194085369, '[10, 20)': 1.02309793066961, '[2, 5)': 0.6453193868594536, '[20, 50)': 1.1343922369339432, '[5, 10)': 0.8474291301376349, '[50, inf)': 1.551114100730357}
- temperature bin median mae_log10: {'-12.5_12.5K': 1.4193285256396908, '1012.5_1037.5K': 0.4183314216705054, '1037.5_1062.5K': 0.6977912895226897, '1062.5_1087.5K': 0.7795018003381167, '1087.5_1112.5K': 0.5446271764623456, '1112.5_1137.5K': 0.40097266577372237, '112.5_137.5K': 0.9026769191001305, '1137.5_1162.5K': 0.37393208346705736, '1162.5_1187.5K': 0.6333630260693872, '1187.5_1212.5K': 0.4958629253219146, '12.5_37.5K': 1.2894837854747654, '1212.5_1237.5K': 0.636395536378174, '1237.5_1262.5K': 0.39176055463562187, '1262.5_1287.5K': 0.4187855369568855, '137.5_162.5K': 0.9130645348080815, '1412.5_1437.5K': 0.6717267118966794, '1462.5_1487.5K': 0.8608276909464505, '162.5_187.5K': 0.892559652546838, '187.5_212.5K': 0.9368899643071948, '212.5_237.5K': 0.9640700137146365, '237.5_262.5K': 0.8961021148271069, '262.5_287.5K': 0.8692128281751885, '287.5_312.5K': 0.6611880192350472, '312.5_337.5K': 0.6767554345895495, '337.5_362.5K': 0.7037490704615081, '362.5_387.5K': 0.5598271272860549, '37.5_62.5K': 1.2259701044462128, '387.5_412.5K': 0.5350274452374326, '412.5_437.5K': 0.48539182860206376, '437.5_462.5K': 0.47533871419925944, '462.5_487.5K': 0.4904794629405941, '487.5_512.5K': 0.4697085577253769, '512.5_537.5K': 0.44431571135990977, '537.5_562.5K': 0.4659380191047172, '562.5_587.5K': 0.47337248945747545, '587.5_612.5K': 0.461317219020403, '612.5_637.5K': 0.4809293883929471, '62.5_87.5K': 0.9996176969321342, '637.5_662.5K': 0.49108089817026535, '662.5_687.5K': 0.5064397331302908, '687.5_712.5K': 0.4632362801297183, '712.5_737.5K': 0.4549922455179488, '737.5_762.5K': 0.46294472219065075, '762.5_787.5K': 0.5210400697418731, '787.5_812.5K': 0.47358118338517974, '812.5_837.5K': 0.5567160136926804, '837.5_862.5K': 0.5732215260035208, '862.5_887.5K': 0.5745758851976682, '87.5_112.5K': 0.9879790563918098, '887.5_912.5K': 0.43294414117800595, '912.5_937.5K': 0.4013749737000287, '937.5_962.5K': 0.6226292356058185, '962.5_987.5K': 0.6593576015686046, '987.5_1012.5K': 0.6024421893309733}
- reliability_level median mae_log10: {'high': 0.7252124839561396, 'low': 0.4593063309220641, 'medium': 0.6954881516836676}
- largest abs_log10 error: 14.927266698895366

## Sanity Check

- prediction_status_ok: True
- sigma_exp_positive: True
- sigma_pred_positive: True
- F0_positive: True
- log10_error_finite: True
- abs_error_consistent: True
- squared_error_consistent: True
- ratio_consistent: True
- log10_ratio_consistent: True
- factor_accuracy_range: True
- mae_nonnegative: True
- rmse_nonnegative: True
- max_abs_ge_mae: True
- n_rows_positive: True
- is_reliable_eval_group_rule: True
- eval_group_reliability_rule: True
- metrics_by_config_has_32_configs: True
- default_comparison_complete: True
- ranking_nonempty: True
- largest_error_rows_limit: True
- coverage_fraction_range: True

## Notes

- WARNING: none
- Main metric is log10(sigma_pred / sigma_exp).
- Sigma spans orders of magnitude, so log error is used instead of ordinary absolute error.
- Step5B train-only sigma0_ref is used; Step4 full-data curves are not used for independent validation.
- Test-side sigma0_S_per_m is not used to create predictions.
- Step5D should create predicted-vs-experimental, error distribution, config comparison, eta, temperature, and material-family plots.
