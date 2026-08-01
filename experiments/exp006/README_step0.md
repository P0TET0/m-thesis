# Step0 Thermoelectric Analysis Base Table

Starrydata2 の熱電材料データから、後続解析に使う解析用データ表 `step0` を作成します。

この段階では、同一 paper / 同一 sample / 同一温度の `Seebeck coefficient` と `electrical conductivity` を 1 行に対応づけます。電気抵抗率しかない場合は `sigma = 1 / rho` で `S/m` に変換します。`eta`、`F0_eta`、`sigma0`、本格的な p/n 分類は計算しません。

## 入力

デフォルトの入力ディレクトリは、リポジトリ直下の次の場所です。

```text
data/raw/starrydata2
```

CSV、JSON、JSONL、Excel、Parquet を再帰的に探索します。Excel は全シートを表として読み込みます。

## 実行

リポジトリ直下から実行します。

```powershell
python experiments/exp006/build_step0_table.py `
  --input data/raw/starrydata2 `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step0_dataset_report.md `
  --match-tol-k 1.0
```

線形補間を許可する場合だけ、次のオプションを追加します。

```powershell
python experiments/exp006/build_step0_table.py `
  --input data/raw/starrydata2 `
  --output experiments/exp006/data/processed `
  --report experiments/exp006/reports/step0_dataset_report.md `
  --match-tol-k 1.0 `
  --allow-interpolation
```

引数を省略した場合も、出力は `experiments/exp006/data/processed/` と `experiments/exp006/reports/` に作成されます。

## 出力

```text
experiments/exp006/data/processed/step0_te_analysis_base.csv
experiments/exp006/data/processed/step0_te_analysis_base.parquet
experiments/exp006/data/processed/step0_rejected_rows.csv
experiments/exp006/data/processed/step0_duplicate_candidates.csv
experiments/exp006/data/processed/step0_schema_detected.json
experiments/exp006/reports/step0_dataset_report.md
```

`pyarrow` など Parquet 保存に必要な依存関係がない場合、CSV は保存し、Parquet 保存不可の理由を report に残します。

## 確認

```powershell
python experiments/exp006/check_step0_outputs.py `
  --processed experiments/exp006/data/processed `
  --match-tol-k 1.0
```

この検査では、必須列、単位変換、`S` の符号保持、`sigma > 0`、`rho > 0`、温度差、`row_id` 一意性、`sigma = 1 / rho` の整合性を確認します。

## 注意

- `S` の符号は保持します。
- `sigma <= 0`、`rho <= 0`、`T <= 0` は解析表から除外し、理由付きで reject に保存します。
- 同一 paper / sample / 温度に値が異なる重複がある場合は解析表から除外します。
- 列名や単位が曖昧な表は無理に推測せず、スキーマ調査結果と確認事項を report に出します。
