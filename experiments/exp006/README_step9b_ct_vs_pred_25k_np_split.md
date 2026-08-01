# Step9B: Step9A 25 K sigma_pred vs SS2026 old C(T)

この解析は Step9A 25K版の sigma_pred 点群と、SS2026 の p/n非分離 old
C(T) 線を、材料系ごと・p/n別に比較する可視化である。

Step9Aで保存済みのprimary-default predictionを読み取り専用で使用します。
新しい` sigma_pred`は計算しません。`fit_tau_eff_step12.py`も再実行せず、
静的に確認した`data/output/starrydata2_step12_tau_fit/sigma_predictions_step12.csv`
の`prefactor_C_S_per_m_step12`をold C(T)として使います。

利用可能なStep12 CSVでは`material_system`が全行`unknown`であるため、
同じCSVに含まれる`composition`をフォールバック材料ラベルとして
broad-familyへ対応づけます。old C(T)は材料系・温度ごとに、`n_or_p`を
group keyに含めず中央値を取り、p/n非分離の1本の線にします。各材料系の
p型図とn型図は同じold C(T)線を使用します。

主図にはStep9Aの`sigma_pred`点とold C(T)線だけを表示します。実測sigmaと
`sigma0_ref`は表示しません。PNG/PDFを両方保存し、seabornは使用しません。

## 小規模テスト

```powershell
python experiments/exp006/build_step9b_ct_vs_pred_25k_np_split.py `
    --predictions experiments/exp006/data/processed/step9a_25k_bin_broad_family/step5b_test_predictions_valid.parquet `
    --old-ct-script experiments/exp005/fit_tau_eff_step12.py `
    --config-id sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median `
    --target-groups broad::SnTe_like broad::PbTe_like broad::BiTe_like broad::SbTe_like broad::SiGe_like broad::oxide broad::sulfide `
    --output experiments/exp006/data/processed/step9b_ct_vs_pred_25k_np_split `
    --figures experiments/exp006/figures/step9b_ct_vs_pred_25k_np_split `
    --report experiments/exp006/reports/step9b_ct_vs_pred_25k_np_split/step9b_ct_vs_pred_25k_np_split_report_test.md `
    --max-groups 3 `
    --max-rows-per-group 2000 `
    --output-suffix _test
```

```powershell
python experiments/exp006/check_step9b_ct_vs_pred_25k_np_split_outputs.py `
    --summary experiments/exp006/data/processed/step9b_ct_vs_pred_25k_np_split/step9b_summary_by_group_carrier_test.csv `
    --figure-index experiments/exp006/data/processed/step9b_ct_vs_pred_25k_np_split/step9b_figure_index_test.csv `
    --report experiments/exp006/reports/step9b_ct_vs_pred_25k_np_split/step9b_ct_vs_pred_25k_np_split_report_test.md
```

## 全件実行

`--max-groups`、`--max-rows-per-group`、`--output-suffix`を外し、report名を
`step9b_ct_vs_pred_25k_np_split_report.md`にして実行します。その後、suffix
なしのsummary、figure index、reportをcheck scriptへ渡します。
