import numpy as np
from scipy.constants import h, k, pi
from scipy.integrate import quad

# 物理定数の定義
hbar = h / (2 * pi)  # 換算プランク定数 (J s)
m = 9.11e-31  # 電子の質量 (kg)
kb = k  # ボルツマン定数 (J/K)
T = 300  # 温度 (K)

# ξ_Fの値を0に設定
xi_F = 20

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

#numpyの浮動少数点数は固定されたビット数(通常は64ビットの倍精度浮動小数点数)を使用しているため、表現できる数値の範囲が制限されている。一方mpmahはビット数が固定されておらず、任意の精度設定が可能なライブラリ。
#mpmathは高精度計算を必要とする状況で優れており、NumPyは大量のデータに対する高速な演算が必要な場合に適している。



