# Focus broad families: unified-axis scatter plots

`focus_broad_families` と、同じフォルダへ移動済みの `focus_sige_like` に対応する、
全15材料分類の実測電気伝導率－予測電気伝導率プロットを共通軸で再出力する。

- x軸・y軸: `10^2`～`10^8 S/m`（対数軸）
- 出力区分: `all`、`p`、`n`（該当データがない区分は省略）
- 各図: PNG、PDF
- 共通範囲外の点が過半数の区分: 全データを含む自動範囲版も追加
- 範囲内判定: 実測値と予測値の両方が共通範囲内の点を「範囲内」とする

## Build

```powershell
python experiments/exp006/build_focus_broad_families_uniform_axis_scatter.py
```

## Outputs

- Figures: `experiments/exp006/figures/focus_broad_families_uniform_axes/`
- Figure index: `experiments/exp006/data/processed/focus_broad_families_uniform_axes/focus_broad_families_uniform_axes_figure_index.csv`
- Report: `experiments/exp006/reports/focus_broad_families_uniform_axes/focus_broad_families_uniform_axes_report.md`

図のファイル名末尾が `_fixed_1e2_1e8` のものが共通軸版である。
追加条件に該当した場合は、末尾が `_full_data_range` の図も作成される。

## 15材料系の一覧図

15材料系の `all` データを重ねず、3行×5列の独立したパネルとして1枚に配置する。
各パネルの軸範囲は `10^2`～`10^8 S/m` で統一し、軸ラベルは表示しない。

```powershell
python experiments/exp006/build_focus_broad_families_uniform_axis_overview.py
```

出力先は
`experiments/exp006/figures/focus_broad_families_uniform_axes/overview/` である。

## y=x への近接度指標

全15材料系について、`all`、`p`、`n` ごとに log10 座標上の y=x への
直交距離、倍率誤差、MAE/RMSE、factor 2/5/10 内率、予測バイアス、
Lin's CCCを計算する。

```powershell
python experiments/exp006/build_focus_broad_families_y_equals_x_metrics.py
```

主な出力は以下である。

- `experiments/exp006/data/processed/focus_broad_families_uniform_axes/focus_broad_families_y_equals_x_metrics.csv`
- `experiments/exp006/data/processed/focus_broad_families_uniform_axes/focus_broad_families_y_equals_x_ranking_all.csv`
- `experiments/exp006/reports/focus_broad_families_uniform_axes/focus_broad_families_y_equals_x_metrics_report.md`

## 材料系別・RMSE分離・外側透過版

各材料系の `all` データを1枚ずつ、軸ラベルなし、目盛値のみで出力する。
プロット領域は白背景、それ以外のキャンバスは透過とする。図内の指標は
表示しない。実測値と予測値の両方が固定軸範囲内に入る点へ先に絞り、
その点だけでRMSEを計算して別CSV・Markdown表へ出力する。

```powershell
python experiments/exp006/build_focus_broad_families_clean_rmse_scatter.py
```

- Figures: `experiments/exp006/figures/focus_broad_families_uniform_axes/material_panels_transparent/`
- RMSE: `experiments/exp006/data/processed/focus_broad_families_uniform_axes/focus_broad_families_rmse_fixed_axes_1e2_1e8.csv`
- Report: `experiments/exp006/reports/focus_broad_families_uniform_axes/focus_broad_families_rmse_fixed_axes_1e2_1e8.md`
