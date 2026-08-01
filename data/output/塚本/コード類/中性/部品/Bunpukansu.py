#グラフと表を出力



import mpmath
import numpy as np
from scipy.constants import h, k, e, pi
import matplotlib.pyplot as plt
import matplotlib.table as tbl

# 物理定数の定義
T = mpmath.mpf(300)  # 温度 [K]
kb = mpmath.mpf(k)  # ボルツマン定数 [J/K]
hbar = mpmath.mpf(h) / (2 * mpmath.pi)
q = mpmath.mpf(e)  # 電気素量 [C]
N_D = mpmath.mpf(1e26)  # ドナー濃度 [1/m^3]
E_D = mpmath.mpf(0.026) * q  # ドナー準位 [eV]を[J]に変換
g_c = mpmath.mpf(2)  # 縮退係数
g_v = mpmath.mpf(4)  # 縮退係数
E_g = mpmath.mpf(0.8) * q  # バンドギャップ [eV]を[J]に変換
m_e = mpmath.mpf(9.11e-31)  # 電子の有効質量 [kg]
m_h = mpmath.mpf(9.11e-31)  # 正孔の有効質量 [kg]
mpmath.mp.dps = 30  # 精度を小数点以下30桁に設定

# N_C, N_Vの計算
N_C = 2 * (((m_e * kb * T) / (2 * mpmath.pi * hbar**2))**(3/2))
N_V = 2 * (((m_h * kb * T) / (2 * mpmath.pi * hbar**2))**(3/2))
xi_G = E_g / (kb * T)  # ξ_Gの計算

def fermi_dirac_half(xi):
    def integrand(x):
        return mpmath.sqrt(x) / (mpmath.exp(x - xi) + 1)
    return mpmath.quad(integrand, [0, mpmath.inf])

def neutrality_and_carrier_densities(xi_F):
    f_D = 1 / (1 + (1 / g_c) * mpmath.exp((E_D / (kb * T))- xi_F))
    n_D = N_D * f_D
    n = (2 / mpmath.sqrt(mpmath.pi)) * N_C * fermi_dirac_half(xi_F)
    p = (2 / mpmath.sqrt(mpmath.pi)) * N_V * fermi_dirac_half(-xi_F - xi_G)
    neutrality = n - (N_D - n_D) - p
    return (xi_F, n, p, n_D, neutrality) 

# データの計算と収集
xi_F_vals = np.arange(-20, 20, 1)
table_data = []
neutrality_vals = []

for xi_F in xi_F_vals:
    xi_F_val, n, p, n_D, neutrality = neutrality_and_carrier_densities(xi_F)
    table_data.append([xi_F_val, mpmath.nstr(neutrality, 10)])
    neutrality_vals.append(float(neutrality))

# グラフと表を描画
fig, ax = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 2]})

# 表を描画
ax[0].axis('tight')
ax[0].axis('off')
table = tbl.table(ax[0], cellText=table_data, colLabels=['ξ_F', 'neutrality'], loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# グラフを描画
ax[1].plot(xi_F_vals, neutrality_vals, 'bo-', label='Neutrality Condition')
ax[1].set_title('Fermi Level vs. Neutrality Condition')
ax[1].set_xlabel('ξ_F (Fermi Level)')
ax[1].set_ylabel('n - (N_D - n_D) - p')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()
