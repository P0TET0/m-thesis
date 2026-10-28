# exp008：DF・ML・FML法の温度別比較

全材料を重ねた電気伝導率σ・ZTの予測–実測散布図について、各点を測定温度で色分けし、温度と予測のずれの関係を確認するためのフォルダです。横軸は実測値、縦軸は予測値です。

## 図の見方と保存先

- `figures/sigma_DF_ML_FML_temperature_fixed.png`：電気伝導率の3手法比較（両対数軸、10²～10⁸ S/m）。
- `figures/zt_DF_ML_FML_temperature_fixed.png`：ZTの3手法比較（線形軸、0～3）。
- `figures/DF/`、`figures/ML/`、`figures/FML/`：手法別の図。
- 各図にPDF版と、外れ値や負の実測ZTも含む `full_range` 版を用意しています。

点の色は温度（K）を表し、全温度版では共通のカラースケール（0～1400 K）を使います。材料を統合して全点を描画し、点数を間引きません。図中のNは試料数ではなく温度点数です。固定軸の範囲外にある点数は図内に表示します。実線は予測と実測の一致、破線と点線はそれぞれ2倍・10倍のずれの目安です。

## 300 K以下だけを色分けした図

`temperature_le_300K/figures/` に、300 K以下の点を0～300 Kの共通カラースケールで色分けした図を保存しています。300 Kちょうどの点も含み、300 Kを超える点は背景に灰色で表示します。軸範囲・点の座標は全温度版と同じです。

比較図のファイル名は全温度版と同じで、手法別・全点範囲・PDF版もあります。`temperature_le_300K/figure_counts.csv` には色付き点数、灰色の点数、軸範囲外の色付き点数も記録します。

`temperature_le_300K_single_color/figures/` には、300 K以下の点を青一色、300 K超の点を灰色で表示した図を保存しています。温度のグラデーションとカラーバーを使わず、2区分の凡例を表示します。元の色分け版と点数・座標・軸範囲は同じです。

## 300 K以下を除外した比較

`temperature_gt_300K/` に、`temperature_K > 300` を満たす点だけを使ったDF・ML・FML法の比較を追加しています。300 Kちょうども除外し、全体図・材料別図・点数・誤差指標に同じ条件を適用しています。

- [図の一覧](temperature_gt_300K/index.html)：全体図と材料別比較へのリンク。
- `temperature_gt_300K/figures/`：全材料を重ねた3手法比較。既存図のp型・n型による色分けと、exp008の温度による色分けの両方があります。
- `temperature_gt_300K/comparisons/sigma/`、`comparisons/zt/`：16材料分類それぞれの3手法横並び比較。
- `temperature_gt_300K/sigma/{DF,ML,FML}/`、`zt/{DF,ML,FML}/`：手法別の全体図・4×4材料一覧・材料ごとの個別図。
- `temperature_gt_300K/filter_summary.csv`：除外前後の点数、温度範囲、除外後のRMSE・MAE。log10指標は実測値と予測値がともに正の点だけで計算しています。

各図に固定軸版・全点範囲版のPNGとPDFがあります。固定軸版の範囲は全温度版と同じです。全点範囲版では除外後の点から範囲を決め、同じ材料の3手法で軸をそろえています。元の予測値をそのまま使い、再学習・再フィッティングは行っていません。

| 物性 | 手法 | 除外前の点数 | 300 K以下の除外点数 | 除外後の点数 |
| --- | --- | ---: | ---: | ---: |
| 電気伝導率 σ | DF | 55,272 | 13,772 | 41,500 |
| 電気伝導率 σ | ML | 12,010 | 1,687 | 10,323 |
| 電気伝導率 σ | FML | 18,968 | 8,313 | 10,655 |
| ZT | DF | 6,381 | 750 | 5,631 |
| ZT | ML | 1,401 | 77 | 1,324 |
| ZT | FML | 2,279 | 338 | 1,941 |

この点数は有限な実測値・予測値の組を数えた温度点数です。σでは両値が正の点に限ります。FMLの温度条件には、既存図と同じσの評価温度を使います。

## ZTの指標の除外前後比較

`temperature_gt_300K/zt_metrics_before_after.csv` に、全温度（before）、300 K超（after）、除外した300 K以下（excluded）の指標を保存しています。`zt_metrics_changes.csv` は手法ごとの変化量、`zt_metrics_manifest.json` は計算条件と入力ファイルのハッシュです。

3手法とも除外後のlog10 RMSE・MAEは低下し、予測値が実測値の1/2～2倍に収まる割合は上昇しました。一方、ZT値そのもののRMSE・MAEは増加しています。除外した低温点では、残った点に比べて実測ZTの平均と絶対誤差が小さく、log10で測った相対的なずれが大きいためです。予測値は同じで、変化は評価対象の温度点を変更した結果です。

通常のRMSE・MAEは有限なZTの組すべてを、log10と倍率の指標は実測値・予測値がともに正の組だけを使っています。固定軸の範囲外の点も計算に含みます。再計算は次のコマンドです。

```powershell
python experiments/exp008/compare_zt_temperature_filter_metrics.py
```

## 元データと比較条件

`data/output/df_ml_fml_material_parity_20261025/plot_source_{DF,ML,FML}.csv` を使用し、既存の全材料散布図の座標を保ったまま、`temperature_K` で色分けしています。σは正の有限値の組、ZTは有限値の組を描画対象とします。点数・温度範囲・軸範囲は `figure_counts.csv`、入力ファイルのパスとSHA-256は `source_manifest.json` に記録します。

手法間で評価対象の試料・温度点・分割が異なるため、同一条件での性能順位を決める比較ではありません。DFは対象試料へのフィッティング結果です。ZTの予測値は実測S・κと予測σから計算した既存値（ZT = S²σT/κ）を使用します。FMLのκ・実測ZTには元処理で最大±2 Kの近傍温度対応があり、色はσの評価温度を表します。

## 再作成

リポジトリのルートで実行します（Python、NumPy、pandas、Matplotlibが必要）。

```powershell
python experiments/exp008/build_temperature_parity.py
```

300 K以下の色分け版は次のコマンドで作成します。

```powershell
python experiments/exp008/build_temperature_parity.py --color-max-temperature 300
```

300 K以下を青一色で表示する場合は、`--single-color` を追加します。

```powershell
python experiments/exp008/build_temperature_parity.py --color-max-temperature 300 --single-color
```

300 K以下を除外した全体図・材料別図は、次のコマンドで作成します。

```powershell
python experiments/exp008/build_temperature_filtered_parity.py
```

`--min-temperature` で下限を変更できます。指定した温度を含まず、それを超える点だけを残します。
