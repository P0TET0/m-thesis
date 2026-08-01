import mpmath
from scipy.constants import h, k, pi

# mpmathの精度を設定
mpmath.mp.dps = 50  # 50桁の精度で計算

# 物理定数の定義
hbar = mpmath.mpf(h) / (2 * mpmath.pi)  # 換算プランク定数 (Js)
m = mpmath.mpf(9.11e-31)  # 電子の質量 (kg)
kb = mpmath.mpf(k)  # ボルツマン定数 (J/K)
T = mpmath.mpf(300)  # 温度 (K)

# ξ_Fの計算
xi_F = 0  # ξ_Fを20として定義

# フェルミ積分の関数定義
def fermi_integral(s, xi_F):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - xi_F
        return xi**s / (mpmath.exp(exponent) + 1)
    
    # mpmathの数値積分関数を使用
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# δ(ξ_F)の計算関数
def calculate_delta(s, xi_F):
    # フェルミ積分の計算
    F_s_plus_32 = fermi_integral(s + 3/2, xi_F)
    F_s_plus_12 = fermi_integral(s + 1/2, xi_F)
    
    # σ(ξ_F)の計算式
    small_delta = ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_plus_12)
    return small_delta

# s = 3/2 の場合のδ(ξ_F)の計算
small_delta_32 = calculate_delta(3/2, xi_F)
print("σ(3/2, ξ_F):", small_delta_32)

# s = -1/2 の場合のδ(ξ_F)の計算
small_delta_m12 = calculate_delta(-1/2, xi_F)
print("σ(-1/2, ξ_F):", small_delta_m12)

