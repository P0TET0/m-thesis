# exp005

Starrydata2 から取得した熱電材料の曲線データを、緩和時間 fitting や機械学習用データセットの作成に使える形へ段階的に整形するための実験フォルダです。

このフォルダ内のスクリプトは、基本的にリポジトリ直下の `data/output/` に中間成果物と最終成果物を書き出します。

## フォルダ内のファイル

| ファイル | 役割 |
| --- | --- |
| `prepare_starrydata2_relaxation_time_workbook.py` | Starrydata2 の曲線 CSV/Excel から、`sample_master` と `property_data` を持つ Excel ワークブックを作成します。 |
| `export_relaxation_time_workbook_to_csv.py` | 上記 Excel ワークブックの `sample_master` / `property_data` シートを CSV に分割して出力します。 |
| `fix_starrydata2_step3_outputs.py` | Step3。前段 CSV の列・単位・候補サンプル情報を補正し、以降の処理で扱いやすい形に整えます。 |
| `merge_starrydata2_step4.py` | Step4。Step3 のサンプル情報と物性曲線を結合し、候補サンプルの曲線も抽出します。 |
| `filter_starrydata2_step5_core_properties.py` | Step5。対象物性を `Electrical conductivity`、`Electrical resistivity`、`Seebeck coefficient`、`Thermal conductivity`、`ZT` に絞り、サンプルごとの物性有無を集計します。 |
| `classify_np_step6.py` | Step6。Seebeck 係数の符号からサンプルを `n` / `p` / `mixed` / `unknown` に分類します。 |
| `standardize_sintering_step7.py` | Step7。焼結情報を一旦 `unknown` / `no` に標準化し、後段で必要なサンプルだけ確認できる状態にします。 |
| `select_learning_candidates_step8.py` | Step8。緩和時間 fitting や学習に使える候補サンプルを、物性の有無・点数・値の妥当性から選別します。 |
| `prepare_literature_review_step9.py` | Step9。文献確認・手動アノテーション用の表、レビューキュー、優先サンプル一覧を作成します。 |
| `build_training_dataset_step10.py` | Step10。曲線データを点単位に展開し、同一サンプル・同一温度で物性値を横持ちにした学習用データセットを作成します。 |

`__pycache__/` は Python の実行時に生成されるキャッシュで、手で編集する対象ではありません。

## 基本的な処理フロー

1. `prepare_starrydata2_relaxation_time_workbook.py`
   - 入力: `data/output/sige/starrydata_curves_fixed.csv`
   - 出力: `data/output/starrydata2_prepared_for_relaxation_time.xlsx`

2. `export_relaxation_time_workbook_to_csv.py`
   - 入力: `data/output/starrydata2_prepared_for_relaxation_time.xlsx`
   - 出力: `data/output/starrydata2_prepared_for_relaxation_time_csv/`

3. `fix_starrydata2_step3_outputs.py`
   - 入力: `data/output/starrydata2_prepared_for_relaxation_time_csv/`
   - 出力: `data/output/starrydata2_step3_fixed/`

4. `merge_starrydata2_step4.py`
   - 入力: `data/output/starrydata2_step3_fixed/`
   - 出力: `data/output/starrydata2_step4_merged/`

5. `filter_starrydata2_step5_core_properties.py`
   - 入力: `data/output/starrydata2_step4_merged/`
   - 出力: `data/output/starrydata2_step5_core_properties/`

6. `classify_np_step6.py`
   - 入力: `data/output/starrydata2_step5_core_properties/`
   - 出力: `data/output/starrydata2_step6_np_classification/`

7. `standardize_sintering_step7.py`
   - 入力: `data/output/starrydata2_step6_np_classification/`
   - 出力: `data/output/starrydata2_step7_sintering_unknown/`

8. `select_learning_candidates_step8.py`
   - 入力: `data/output/starrydata2_step7_sintering_unknown/`
   - 出力: `data/output/starrydata2_step8_learning_candidates/`

9. `prepare_literature_review_step9.py`
   - 入力: `data/output/starrydata2_step8_learning_candidates/`
   - 出力: `data/output/starrydata2_step9_literature_annotations/`

10. `build_training_dataset_step10.py`
    - 入力: `data/output/starrydata2_step9_literature_annotations/`
    - 出力: `data/output/starrydata2_step10_training_dataset/`

## 実行例

前処理ワークブックを作成します。

```powershell
python experiments/exp005/prepare_starrydata2_relaxation_time_workbook.py
```

ワークブックを CSV に分割します。

```powershell
python experiments/exp005/export_relaxation_time_workbook_to_csv.py
```

Step3 以降を順番に実行します。

```powershell
python experiments/exp005/fix_starrydata2_step3_outputs.py
python experiments/exp005/merge_starrydata2_step4.py
python experiments/exp005/filter_starrydata2_step5_core_properties.py
python experiments/exp005/classify_np_step6.py
python experiments/exp005/standardize_sintering_step7.py
python experiments/exp005/select_learning_candidates_step8.py
python experiments/exp005/prepare_literature_review_step9.py
python experiments/exp005/build_training_dataset_step10.py
```

## 主要な出力

| 出力先 | 内容 |
| --- | --- |
| `data/output/starrydata2_prepared_for_relaxation_time.xlsx` | 緩和時間 fitting 用の初期 Excel ワークブック。 |
| `data/output/starrydata2_prepared_for_relaxation_time_csv/` | `sample_master.csv` と `property_data.csv`。 |
| `data/output/starrydata2_step3_fixed/` | 補正済み sample/property CSV、候補サンプル一覧、品質レポート、Excel 版。 |
| `data/output/starrydata2_step4_merged/` | サンプル情報を結合した曲線データ、候補曲線、Step4 レポート、Excel 版。 |
| `data/output/starrydata2_step5_core_properties/` | 主要 5 物性に絞った曲線、候補曲線、サンプル別物性有無、除外曲線、Excel 版。 |
| `data/output/starrydata2_step6_np_classification/` | n/p 分類結果を付与したサンプル・曲線データ、レポート、Excel 版。 |
| `data/output/starrydata2_step7_sintering_unknown/` | 焼結情報を標準化した Step7 データ、レポート、Excel 版。 |
| `data/output/starrydata2_step8_learning_candidates/` | 学習候補、初期 fitting 候補、レビュー対象、非候補、fitting 用 sigma/rho 曲線、Excel 版。 |
| `data/output/starrydata2_step9_literature_annotations/` | 文献確認用アノテーション表、レビューキュー、手動入力テンプレート、指示書、Excel 版。 |
| `data/output/starrydata2_step10_training_dataset/` | 点単位 long データ、温度集約データ、横持ち学習データセット、fitting 用データ、品質確認用レポート。 |

## 引数で変更できる主な指定

各スクリプトは既定の入出力先を持っていますが、CLI 引数で変更できます。

- `--input-path`, `--input-sheet`, `--output-path`
- `--sample-master`, `--property-data`, `--candidate-samples`
- `--input`, `--candidate-input`
- `--step6_dir`, `--step7_dir`, `--step8_dir`, `--step9_dir`
- `--output-dir`, `--output_dir`
- `--manual_annotations`
- `--top_n_review`
- `--temperature_tolerance_K`
- `--temperature_round_decimals`

引数名はスクリプトごとに少し違うため、詳細は各ファイルの `parse_args()` を確認してください。
