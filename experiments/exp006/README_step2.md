# Step2 Eta Workflow

Step2 は、Step1 で分類済みの Seebeck 係数から換算フェルミ準位 `eta` を扱う段階です。

- Step2A: `eta` grid 上で `F0(eta)`、`F1(eta)`、`S_abs(eta)` の lookup table を作成します。
- Step2B: Step1 の `step1_eta_input_candidates` に lookup table を照合し、各データ点へ `eta`、`F0_eta`、`F1_eta` を付与します。

Step2B でも `sigma0` はまだ計算しません。Step3 で `sigma0 = sigma_S_per_m / F0_eta` を計算します。

## Step2A

```powershell
python experiments/exp006/build_step2_eta_lookup.py `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step2a_eta_lookup_report.md `
  --eta-min -50 `
  --eta-max 500 `
  --d-eta 0.005
```

確認:

```powershell
python experiments/exp006/check_step2_eta_lookup.py `
  --lookup experiments/exp006/data/processed/step2_eta_lookup_table.csv
```

## Step2B

小規模テスト:

```powershell
python experiments/exp006/build_step2_eta_table.py `
  --input experiments/exp006/data/processed/step1_eta_input_candidates.parquet `
  --lookup experiments/exp006/data/processed/step2_eta_lookup_table.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step2_eta_report_test.md `
  --max-rows 1000 `
  --output-suffix _test
```

小規模テストの確認:

```powershell
python experiments/exp006/check_step2_eta_outputs.py `
  --input experiments/exp006/data/processed/step2_eta_calculated_test.csv `
  --ge1 experiments/exp006/data/processed/step2_eta_ge1_candidates_test.csv `
  --conservative-ge1 experiments/exp006/data/processed/step2_conservative_eta_ge1_candidates_test.csv
```

全件実行:

```powershell
python experiments/exp006/build_step2_eta_table.py `
  --input experiments/exp006/data/processed/step1_eta_input_candidates.parquet `
  --lookup experiments/exp006/data/processed/step2_eta_lookup_table.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step2_eta_report.md
```

全件確認:

```powershell
python experiments/exp006/check_step2_eta_outputs.py `
  --input experiments/exp006/data/processed/step2_eta_calculated.csv `
  --ge1 experiments/exp006/data/processed/step2_eta_ge1_candidates.csv `
  --conservative-ge1 experiments/exp006/data/processed/step2_conservative_eta_ge1_candidates.csv
```

## Step2B Outputs

```text
experiments/exp006/data/processed/step2_eta_calculated.csv
experiments/exp006/data/processed/step2_eta_calculated.parquet
experiments/exp006/data/processed/step2_eta_ge1_candidates.csv
experiments/exp006/data/processed/step2_eta_ge1_candidates.parquet
experiments/exp006/data/processed/step2_conservative_eta_ge1_candidates.csv
experiments/exp006/data/processed/step2_conservative_eta_ge1_candidates.parquet
experiments/exp006/data/processed/step2_eta_failed_or_out_of_range.csv
experiments/exp006/data/processed/step2_eta_counts_by_material_family.csv
experiments/exp006/reports/step2_eta_report.md
```

`step2_eta_calculated` は Step1 の `step1_eta_input_candidates` と同じ行数になります。`step2_eta_ge1_candidates` は `is_valid_for_sigma0_step3 == True` の行だけです。`step2_conservative_eta_ge1_candidates` はさらに `is_conservative_main_analysis == True` の行に絞ります。
