# 材料系別散布図のRMSE

RMSEは軸範囲外を含む全有効点について、`log10(予測値/実測値)` の二乗平均平方根として計算した。
RMSE計算後、散布図の表示範囲だけをx/yともに10^2～10^8 S/mへ絞った。0に近いほどy=xに近い。
RMSE値は散布図内には記載せず、この表とCSVに分離した。

CSV: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\focus_broad_families_uniform_axes\focus_broad_families_rmse_only_all.csv`

| 材料系 | RMSE (log10) |
|---|---:|
| Si-Ge系 | 0.928 |
| Sn-Te系 | 0.371 |
| Pb-Te系 | 0.460 |
| Bi-Te系 | 0.566 |
| Sb-Te系 | 0.634 |
| Bi-Sb-Te系 | 1.053 |
| Ge-Te系 | 2.043 |
| Mg-Si/Sn系 | 1.062 |
| Co-Sb系・スクッテルダイト系 | 1.203 |
| セレン化物系 | 0.847 |
| テルル化物系 | 0.585 |
| 硫化物系 | 2.498 |
| 酸化物系 | 1.587 |
| その他の化学式系 | 1.196 |
| 分類不能 | 1.605 |
