# Step1 Carrier Classification

Step0 で作成した解析用データ表に対して、ゼーベック係数 `S` の符号から `carrier_type` を付与します。

この Step1 では `eta`、`F0_eta`、`sigma0` は計算しません。Starrydata2 の raw data も読み込みません。

## 入力

優先入力は Step0 の Parquet です。

```text
experiments/exp006/data/processed/step0_te_analysis_base.parquet
experiments/exp006/data/processed/step0_te_analysis_base.csv
```

`--input` を省略した場合は、`experiments/exp006/data/processed/` 内の Parquet があれば Parquet、なければ CSV を使います。

## 実行

リポジトリ直下から実行します。

```powershell
python experiments/exp006/build_step1_carrier_table.py `
  --input experiments/exp006/data/processed/step0_te_analysis_base.parquet `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step1_carrier_report.md `
  --zero-threshold-uV 1.0
```

CSV 入力でも実行できます。

```powershell
python experiments/exp006/build_step1_carrier_table.py `
  --input experiments/exp006/data/processed/step0_te_analysis_base.csv `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step1_carrier_report.md `
  --zero-threshold-uV 1.0
```

## 分類ルール

`zero_threshold_uV` のデフォルトは `1.0` です。

```text
S_uV_per_K > +zero_threshold_uV  -> carrier_type = p
S_uV_per_K < -zero_threshold_uV  -> carrier_type = n
abs(S_uV_per_K) <= threshold     -> carrier_type = unknown_near_zero
```

`S` の符号は保持します。後続の eta 計算で `|S|` を使う場合でも、分類には符号付きの `S_uV_per_K` を使います。

## 出力

```text
experiments/exp006/data/processed/step1_te_carrier_classified.csv
experiments/exp006/data/processed/step1_te_carrier_classified.parquet
experiments/exp006/data/processed/step1_eta_input_candidates.csv
experiments/exp006/data/processed/step1_eta_input_candidates.parquet
experiments/exp006/data/processed/step1_conservative_main_candidates.csv
experiments/exp006/data/processed/step1_conservative_main_candidates.parquet
experiments/exp006/data/processed/step1_sample_sign_summary.csv
experiments/exp006/data/processed/step1_carrier_counts_by_material_family.csv
experiments/exp006/reports/step1_carrier_report.md
```

`step1_eta_input_candidates` は `carrier_type` が `p` または `n` の行だけです。`sample_has_sign_change == True` の行も、この段階では除外しません。

`step1_conservative_main_candidates` は `carrier_type` が `p` または `n` で、同一 sample 内に p/n の符号反転がない行だけです。

## 確認

```powershell
python experiments/exp006/check_step1_outputs.py `
  --input experiments/exp006/data/processed/step1_te_carrier_classified.csv `
  --zero-threshold-uV 1.0
```

この検査では、入力行数と出力行数の一致、`row_id` 一意性、`S_uV_per_K = S_V_per_K * 1e6`、分類ルール、sample 符号反転フラグ、候補ファイルの条件を確認します。
