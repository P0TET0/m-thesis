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
N_D = mpmath.mpf(2.15e26)  # ドナー濃度 [1/m^3]
E_D = mpmath.mpf(0.026) * q  # ドナー準位 [eV]を[J]に変換
g_c = mpmath.mpf(2)  # 縮退係数
g_v = mpmath.mpf(4)  # 縮退係数
E_g = mpmath.mpf(0.13) * q  # バンドギャップ [eV]を[J]に変換
m_e = mpmath.mpf(9.11e-31)  # 電子の有効質量 [kg]
m_h = mpmath.mpf(9.11e-31)  # 正孔の有効質量 [kg]
mpmath.mp.dps = 50  # 精度を小数点以下50桁に設定

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

# xi_Fの初期探索
xi_F_vals = np.arange(-20, 20, 1)
previous_neutrality = None
found = False

for xi_F in xi_F_vals:
    xi_F_val, n, p, n_D, neutrality = neutrality_and_carrier_densities(xi_F)
    print(f"xi_F = {xi_F_val}, n = {n}, p = {p}, n_D = {n_D}, neutrality = {neutrality}")
    if previous_neutrality is not None:
        # 符号が変わったことを検出
        if previous_neutrality * neutrality < 0:
            # 符号が変わった範囲をより細かく探索
            lower_bound = xi_F - 1
            upper_bound = xi_F
            interval = 0.1
            while True:
                refined_xi_F_vals = np.linspace(lower_bound, upper_bound, 100)
                for refined_xi_F in refined_xi_F_vals:
                    xi_F_val, n, p, n_D, refined_neutrality = neutrality_and_carrier_densities(refined_xi_F)
                    print(f"Refined: xi_F = {refined_xi_F}, n = {n}, p = {p}, n_D = {n_D}, neutrality = {refined_neutrality}")
                    if abs(refined_neutrality) < 1e15:  
                        print(f"Neutrality condition met: xi_F = {refined_xi_F}")
                        found = True
                        break
                if found:
                    break
                # 次のループでさらに細かく探索
                for i in range(1, len(refined_xi_F_vals)):
                    if (neutrality_and_carrier_densities(refined_xi_F_vals[i - 1])[4] *
                            neutrality_and_carrier_densities(refined_xi_F_vals[i])[4]) < 0:
                        lower_bound = refined_xi_F_vals[i - 1]
                        upper_bound = refined_xi_F_vals[i]
                        interval *= 0.1
                        break
                else:
                    break
            if found:
                break
    previous_neutrality = neutrality

# グラフと表の描画
if not found:
    print("Neutrality condition not found in the given range.")

# グラフの描画
fig, ax = plt.subplots(figsize=(10, 6))
neutrality_vals = [neutrality_and_carrier_densities(xi_F)[4] for xi_F in np.arange(-20, 20, 1)]
ax.plot(np.arange(-20, 20, 1), neutrality_vals, 'bo-', label='Neutrality Condition')
ax.set_title('Fermi Level vs. Neutrality Condition')
ax.set_xlabel('ξ_F (Fermi Level)')
ax.set_ylabel('n - (N_D - n_D) - p')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()