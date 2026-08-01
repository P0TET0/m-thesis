import numpy as np
from scipy.constants import h, k, pi
from scipy.integrate import quad

# 物理定数の定義
hbar = h / (2 * pi)  # 換算プランク定数 (J s)
m = 9.11e-31  # 電子の質量 (kg)
n = 1.0e29  # 電子密度 (電子/m^3)
kb = k  # ボルツマン定数 (J/K)
T = 293  # 温度 (K)

# フェルミエネルギーの計算
EF = (hbar**2 / (2 * m)) * (3 * pi**2 * n)**(2/3)

# ξ_Fの計算
xi_F = EF / (kb * T)

# フェルミ積分の関数定義
def fermi_integral(s, xi_F):
    integrand = lambda xi: xi**s / (np.exp(xi - xi_F) + 1)
    result, _ = quad(integrand, 0, np.inf)
    return result

# S = 3/2 および S = -1/2 のフェルミ積分の計算
fermi_integral_32 = fermi_integral(3/2, xi_F)
fermi_integral_m12 = fermi_integral(-1/2, xi_F)

print("F_3/2(xi_F):", fermi_integral_32)
print("F_-1/2(xi_F):", fermi_integral_m12)
