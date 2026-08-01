import mpmath
from scipy.constants import h, k, e, pi

# mpmathの精度を設定
mpmath.mp.dps = 30  # 30桁の精度で計算

# 物理定数の定義
hbar = mpmath.mpf(h) / (2 * mpmath.pi)  # 換算プランク定数 (Js)
m = mpmath.mpf(9.11e-31)  # 電子の質量 (kg)
kb = mpmath.mpf(k)  # ボルツマン定数 (J/K)
q = mpmath.mpf(e)  # 電子の電荷 (C)
T = mpmath.mpf(300)  # 温度 (K)

# ξ_Fの計算
xi_F = 0  # ξ_Fを0として定義

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
    F_s_plus_32 = fermi_integral(s + 3/2, xi_F)
    F_s_plus_12 = fermi_integral(s + 1/2, xi_F)
    
    # δ(ξ_F)の計算式
    small_delta = ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_plus_12)
    return small_delta

# ゼーベック係数αの計算
def calculate_alpha(s, xi_F):
    small_delta = calculate_delta(s, xi_F)
    alpha = (kb/q) * (small_delta - xi_F)
    return alpha

# 各s値に対するゼーベック係数を計算して出力
alpha_32 = calculate_alpha(3/2, xi_F)
alpha_m12 = calculate_alpha(-1/2, xi_F)

print("ξ_F:", xi_F)
print("ゼーベック係数 α (s = 3/2):", alpha_32 * 1000, "[mV/K]")
print("ゼーベック係数 α (s = -1/2):", alpha_m12 * 1000, "[mV/K]")
