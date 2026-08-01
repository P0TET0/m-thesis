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
E_g = mpmath.mpf(0.8) * q  # バンドギャップ [eV]を[J]に変換

# ξ_Fの計算
xi_F = 0  # ξ_Fを0として定義

# ξ_gの計算
xi_g = E_g / (kb * T)  # ξ_Gの計算

# フェルミ積分_hの関数定義
def fermi_integral_h(s, xi_F, xi_g):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = -xi_F - xi_g + xi
        return xi**s / (mpmath.exp(exponent) + 1)
    
    # mpmathの数値積分関数を使用
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# δ(-ξ_F-ξ_g)の計算関数
def calculate_delta_h(s, xi_F, xi_g):
    # フェルミ積分を -ξ_F - ξ_g の形で計算
    F_s_plus_52 = fermi_integral_h(s + 5/2, -xi_F, -xi_g)
    F_s_plus_12 = fermi_integral_h(s + 1/2, -xi_F, -xi_g)
    
    # δ(-ξ_F - ξ_g)の計算式
    small_delta_h = ((s + 5/2) * F_s_plus_52) / ((s + 3/2) * F_s_plus_12)
    return small_delta_h

# ゼーベック係数 α_h の計算
def calculate_alpha_h(s, xi_F, xi_g):
    small_delta_h = calculate_delta_h(s, xi_F, xi_g)
    alpha_h = -((kb/q) * (small_delta_h - ( - xi_F - xi_g)))
    return alpha_h

# 各s値に対するゼーベック係数を計算して出力
alpha_h_32 = calculate_alpha_h(3/2, xi_F, xi_g)
alpha_h_m12 = calculate_alpha_h(-1/2, xi_F, xi_g)

print("ξ_F:", xi_F)
print("ゼーベック係数 α_h (s = 3/2):", alpha_h_32 * 1000, "[mV/K]")
print("ゼーベック係数 α_h (s = -1/2):", alpha_h_m12 * 1000, "[mV/K]")
