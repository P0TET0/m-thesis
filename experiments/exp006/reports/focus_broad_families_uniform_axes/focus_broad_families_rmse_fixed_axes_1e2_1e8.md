# 材料系別散布図のRMSE（固定軸範囲）

実測値と予測値の両方が10^2～10^8 S/mに入る点へ先に絞り、
その点について`log10(予測値/実測値)`の二乗平均平方根を計算した。0に近いほどy=xに近い。
RMSE値は散布図内には記載せず、この表とCSVに分離した。

CSV: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\focus_broad_families_uniform_axes\focus_broad_families_rmse_fixed_axes_1e2_1e8.csv`

| 材料系 | 使用点数 | RMSE (log10) |
|---|---:|---:|
| Si-Ge系 | 197 | 0.928 |
| Sn-Te系 | 370 | 0.371 |
| Pb-Te系 | 472 | 0.460 |
| Bi-Te系 | 1764 | 0.490 |
| Sb-Te系 | 974 | 0.634 |
| Bi-Sb-Te系 | 616 | 0.611 |
| Ge-Te系 | 160 | 0.156 |
| Mg-Si/Sn系 | 260 | 0.971 |
| Co-Sb系・スクッテルダイト系 | 1318 | 0.885 |
| セレン化物系 | 1744 | 0.636 |
| テルル化物系 | 403 | 0.577 |
| 硫化物系 | 696 | 0.734 |
| 酸化物系 | 3049 | 0.956 |
| その他の化学式系 | 5968 | 0.903 |
| 分類不能 | 71 | 0.643 |
