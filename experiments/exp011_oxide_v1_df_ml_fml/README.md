# exp011：酸化物の細分類によるDF・ML・FML法の比較

exp007の組成分類 `oxide_v1` をDF法・ML法にも適用し、再計算した電気伝導率σとZTをFML法と比較するフォルダです。酸化物以外の分類は維持し、細分類前（broad）と細分類後（fine）の結果を保存します。

- `index.html`：全図へのリンク。
- `figures/sigma/`、`figures/zt/`：27分類（酸化物12分類＋その他15分類）の3手法比較、酸化物全体、全材料を重ねた図（PNG・PDF）。p型はオレンジ、n型は青で、exp007と同じ色です。
- `figures/common/`：3手法に共通する評価点だけの比較。
- `data/metrics.csv`：分類別・全体の点数、試料数、RMSE・MAE。
- `data/classification_effect.csv`：同一評価点での細分類前後の差。
- `data/predictions_{broad,fine}.csv`：予測値と観測値。
- `data/plot_source_{broad,fine}.csv`：exp007に合わせた描画・集計対象。
- `data/exp007_alignment_counts.csv`：前回との分類別点数の対応。
- `reports/results.md`：計算条件と結果。

酸化物の図・集計は、exp007の比較図と同じ2,984温度点・552試料の行IDに限定します。要確認・除外候補を除き、酸化物以外の対象は維持しています。DF・MLにも同じ対象行を適用しますが、予測がない点は描きません。ZTには実測κ・ZTの対応も必要です。旧条件の図・集計は `archive/unfiltered_material_colors_20261026/` に保存しています。

DFは分類別の基準値から再フィットした結果、MLは基準値と学習用τを再計算して再学習した結果、FMLはexp007の保存済み結果を使います。今回の対象・色の統一では予測値を変更していません。DFは評価試料へのフィッティングであり、ML・FMLとは評価の意味が異なります。ZTは実測S・κと予測σから計算します。

再実行はリポジトリのルートで行います。Python、NumPy、pandas、Matplotlib、scikit-learn、joblib、openpyxlが必要です。

```powershell
python experiments/exp011_oxide_v1_df_ml_fml/build_oxide_comparison.py --plots-only
python experiments/exp011_oxide_v1_df_ml_fml/check_oxide_comparison.py
```

`.runtime/python-packages` があれば自動参照します。予測の計算からやり直す場合は `--plots-only` を外します。
