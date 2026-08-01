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

# Δ_h(-ξ_F - ξ_g)の計算関数
def calculate_big_delta_h(s, xi_F, xi_g):
    # フェルミ積分を -ξ_F - ξ_g の形で計算
    F_s_plus_52 = fermi_integral_h(s + 5/2, -xi_F, -xi_g)
    F_s_plus_12 = fermi_integral_h(s + 1/2, -xi_F, -xi_g)
    
    # δ_h(-ξ_F - ξ_g)の計算
    small_delta_h = calculate_small_delta_h(s, -xi_F, -xi_g)

    # Δ_h(-ξ_F - ξ_g)の計算式
    big_delta_h = (((s + 7/2) * F_s_plus_52) / ((s + 3/2) * F_s_plus_12)) - small_delta_h**2
    return big_delta_h

# δ_h(-ξ_F - ξ_g)の計算関数
def calculate_small_delta_h(s, xi_F, xi_g):
    F_s_plus_32 = fermi_integral_h(s + 3/2, xi_F, xi_g)
    F_s_minus_12 = fermi_integral_h(s + 1/2, xi_F, xi_g)
    return ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_minus_12)

# L_hの計算
def calculate_lorentz_number_h(s, xi_F, xi_g):
    big_delta_h = calculate_big_delta_h(s, xi_F, xi_g)
    L_h = (kb / q) ** 2 * big_delta_h
    return L_h

# 各s値に対するL_hを計算して出力
L_h_32 = calculate_lorentz_number_h(3/2, xi_F, xi_g)
L_h_m12 = calculate_lorentz_number_h(-1/2, xi_F, xi_g)

print("ξ_F:", xi_F)
print("ローレンツ数 L_h (s = 3/2):", L_h_32 * 10**8, "[×10^-8 WΩK^-2]")
print("ローレンツ数 L_h (s = -1/2):", L_h_m12 * 10**8, "[×10^-8 WΩK^-2]")
