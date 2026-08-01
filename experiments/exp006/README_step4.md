# Step4 Sigma0 Reference Curves

Step4 では、Step3 で計算済みの有効な `sigma0` を 100 K ごとの温度ビンにまとめ、`sigma0(T)` の基準曲線候補を作成します。

この段階では、予測精度評価や train/test 分割は行いません。それらは Step5 で行います。

## 実行

小規模テスト:

```powershell
python experiments/exp006/build_step4_sigma0_reference_curves.py `
  --input experiments/exp006/data/processed/step3_sigma0_valid.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step4_sigma0_reference_curve_report_test.md `
  --bin-width-k 100 `
  --bin-start-k 50 `
  --min-rows-per-bin 3 `
  --min-samples-per-bin 3 `
  --min-papers-per-bin 1 `
  --max-rows 3000 `
  --output-suffix _test
```

小規模テストの確認:

```powershell
python experiments/exp006/check_step4_sigma0_reference_curves.py `
  --curve experiments/exp006/data/processed/step4_sigma0_reference_curve_bins_test.csv `
  --reliable experiments/exp006/data/processed/step4_sigma0_reference_curve_reliable_test.csv `
  --default experiments/exp006/data/processed/step4_sigma0_reference_curve_default_test.csv `
  --binned-rows experiments/exp006/data/processed/step4_sigma0_binned_input_rows_test.csv `
  --dropped experiments/exp006/data/processed/step4_sigma0_dropped_rows_test.csv
```

全件実行:

```powershell
python experiments/exp006/build_step4_sigma0_reference_curves.py `
  --input experiments/exp006/data/processed/step3_sigma0_valid.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step4_sigma0_reference_curve_report.md `
  --bin-width-k 100 `
  --bin-start-k 50 `
  --min-rows-per-bin 3 `
  --min-samples-per-bin 3 `
  --min-papers-per-bin 1
```

全件確認:

```powershell
python experiments/exp006/check_step4_sigma0_reference_curves.py `
  --curve experiments/exp006/data/processed/step4_sigma0_reference_curve_bins.csv `
  --reliable experiments/exp006/data/processed/step4_sigma0_reference_curve_reliable.csv `
  --default experiments/exp006/data/processed/step4_sigma0_reference_curve_default.csv `
  --binned-rows experiments/exp006/data/processed/step4_sigma0_binned_input_rows.csv `
  --dropped experiments/exp006/data/processed/step4_sigma0_dropped_rows.csv
```

## 出力

```text
experiments/exp006/data/processed/step4_sigma0_binned_input_rows.csv
experiments/exp006/data/processed/step4_sigma0_binned_input_rows.parquet
experiments/exp006/data/processed/step4_sigma0_reference_curve_bins.csv
experiments/exp006/data/processed/step4_sigma0_reference_curve_bins.parquet
experiments/exp006/data/processed/step4_sigma0_reference_curve_reliable.csv
experiments/exp006/data/processed/step4_sigma0_reference_curve_reliable.parquet
experiments/exp006/data/processed/step4_sigma0_reference_curve_default.csv
experiments/exp006/data/processed/step4_sigma0_reference_curve_default.parquet
experiments/exp006/data/processed/step4_sigma0_curve_coverage_by_group.csv
experiments/exp006/data/processed/step4_sigma0_dropped_rows.csv
experiments/exp006/reports/step4_sigma0_reference_curve_report.md
```

`recommended_default == True` は `conservative_valid + sample_median + reliable bin` を示します。Step5 では global と material_family の両方を比較できます。
