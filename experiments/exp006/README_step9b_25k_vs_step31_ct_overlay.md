# Step9B: 25 K sigma0_ref(T) vs Step31 SS2026 C(T)

この解析は、exp006 Step9A の 25 K版 sigma0_ref(T) と、exp005 Step31 の SS2026版 C(T) を、材料系 × n/p型ごとに比較する別枠解析である。

既存のStep0～Step9AおよびStep31は再実行せず、Starrydata2 raw dataも読みません。新しい `sigma_pred` は計算せず、既存入力は上書きしません。図はmatplotlibのAgg backendでPNGとPDFの両方を作成します。

## 入力

- `experiments/exp006/data/processed/step9a_25k_bin_broad_family/step5b_train_reference_curve_bins.parquet`
- `experiments/exp005/plot_reclassified_material_system_ct_curves_step31.py`
- `data/output/starrydata2_step31_reclassified_ct_curves/step31_ct_curve_report.txt`
- `data/output/starrydata2_step31_reclassified_ct_curves/` 内で自動検出したn/p別C(T) CSV

Step9A側では次のconfigだけを採用します。

`sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median`

対象材料系はデフォルトで両入力の共通集合です。`--target-groups` で明示指定でき、`--max-groups` と `--output-suffix` は小規模テストに利用できます。

## 小規模テスト

```powershell
python experiments/exp006/build_step9b_25k_vs_step31_ct_overlay.py `
  --sigma0ref25k experiments/exp006/data/processed/step9a_25k_bin_broad_family/step5b_train_reference_curve_bins.parquet `
  --step31-script experiments/exp005/plot_reclassified_material_system_ct_curves_step31.py `
  --step31-report data/output/starrydata2_step31_reclassified_ct_curves/step31_ct_curve_report.txt `
  --step31-output-dir data/output/starrydata2_step31_reclassified_ct_curves `
  --output experiments/exp006/data/processed/step9b_25k_vs_step31_ct_overlay `
  --figures experiments/exp006/figures/step9b_25k_vs_step31_ct_overlay `
  --report experiments/exp006/reports/step9b_25k_vs_step31_ct_overlay/step9b_25k_vs_step31_ct_overlay_report_test.md `
  --max-groups 3 `
  --output-suffix _test

python experiments/exp006/check_step9b_25k_vs_step31_ct_overlay_outputs.py `
  --summary experiments/exp006/data/processed/step9b_25k_vs_step31_ct_overlay/step9b_summary_by_group_carrier_test.csv `
  --figure-index experiments/exp006/data/processed/step9b_25k_vs_step31_ct_overlay/step9b_figure_index_test.csv `
  --parse-summary experiments/exp006/data/processed/step9b_25k_vs_step31_ct_overlay/step9b_step31_ct_parse_summary_test.csv `
  --report experiments/exp006/reports/step9b_25k_vs_step31_ct_overlay/step9b_25k_vs_step31_ct_overlay_report_test.md
```

## 全件実行

小規模テストが通った後、上のbuildコマンドから `--max-groups 3` と `--output-suffix _test` を外し、report名を `step9b_25k_vs_step31_ct_overlay_report.md` に変更します。続いてcheckコマンドの各 `_test` を外して検査します。

スクリプトは同名のStep9B成果物が既にある場合、上書きせずエラーで停止します。

## `step9b_ct_vs_pred_25k_np_split` と同じ形式の補助図

次のコマンドは、carrier別の色付き点と黒いC(T)線という既存Step9Bと同じ形式で補助図を作成します。p型は橙、n型は青で、Step9Aで保存済みのholdout/test `sigma_pred` を点、Step31のn/p別C(T)を黒線として表示します。`sigma_pred` は再計算しません。

```powershell
python experiments/exp006/build_step9b_25k_vs_step31_ct_style_matched_figures.py --overwrite
```

出力先は `experiments/exp006/figures/step9b_25k_vs_step31_ct_overlay/ct_vs_pred_style/` です。全図のx軸は少なくとも0～1300 Kを含み、1300 Kを超えるデータも保持します。元の `sigma0_ref vs Step31 C(T)` の解析CSVとoverlay図は別の比較として保持します。
