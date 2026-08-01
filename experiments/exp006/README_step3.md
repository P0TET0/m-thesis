# Step3 Sigma0 Calculation

Step3 では、Step2B の `eta >= 1` 候補に対して、実測電気伝導率から `sigma0` を計算します。

```text
sigma0 = sigma_S_per_m / F0_eta
```

この段階では 100 K ごとの温度ビンや中央値曲線は作りません。それらは Step4 で行います。

## 実行

小規模テスト:

```powershell
python experiments/exp006/build_step3_sigma0_table.py `
  --input experiments/exp006/data/processed/step2_eta_ge1_candidates.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step3_sigma0_report_test.md `
  --max-rows 1000 `
  --output-suffix _test
```

小規模テストの確認:

```powershell
python experiments/exp006/check_step3_sigma0_outputs.py `
  --input experiments/exp006/data/processed/step3_sigma0_calculated_test.csv `
  --valid experiments/exp006/data/processed/step3_sigma0_valid_test.csv `
  --conservative-valid experiments/exp006/data/processed/step3_conservative_sigma0_valid_test.csv
```

全件実行:

```powershell
python experiments/exp006/build_step3_sigma0_table.py `
  --input experiments/exp006/data/processed/step2_eta_ge1_candidates.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step3_sigma0_report.md
```

全件確認:

```powershell
python experiments/exp006/check_step3_sigma0_outputs.py `
  --input experiments/exp006/data/processed/step3_sigma0_calculated.csv `
  --valid experiments/exp006/data/processed/step3_sigma0_valid.csv `
  --conservative-valid experiments/exp006/data/processed/step3_conservative_sigma0_valid.csv
```

## 出力

```text
experiments/exp006/data/processed/step3_sigma0_calculated.csv
experiments/exp006/data/processed/step3_sigma0_calculated.parquet
experiments/exp006/data/processed/step3_sigma0_valid.csv
experiments/exp006/data/processed/step3_sigma0_valid.parquet
experiments/exp006/data/processed/step3_conservative_sigma0_valid.csv
experiments/exp006/data/processed/step3_conservative_sigma0_valid.parquet
experiments/exp006/data/processed/step3_sigma0_failed.csv
experiments/exp006/data/processed/step3_sigma0_summary_by_sample.csv
experiments/exp006/data/processed/step3_sigma0_summary_by_material_family.csv
experiments/exp006/reports/step3_sigma0_report.md
```

Step4 では、100 K ごとの温度ビンを作り、材料系・carrier_type ごとに `log10_sigma0` の中央値曲線を作る想定です。
