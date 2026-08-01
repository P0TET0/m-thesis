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

# フェルミ積分_eの関数定義
def fermi_integral_e(s, xi_F):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - xi_F
        return xi**s / (mpmath.exp(exponent) + 1)
    
    # mpmathの数値積分関数を使用
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# δ_e(ξ_F)の計算関数
def calculate_delta_e(s, xi_F):
    F_s_plus_32 = fermi_integral_e(s + 3/2, xi_F)
    F_s_plus_12 = fermi_integral_e(s + 1/2, xi_F)
    
    # δ_e(ξ_F)の計算式
    small_delta_e = ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_plus_12)
    return small_delta_e

# ゼーベック係数α_eの計算
def calculate_alpha_e(s, xi_F):
    small_delta_e = calculate_delta_e(s, xi_F)
    alpha_e = -((kb/q) * (small_delta_e - xi_F))
    return alpha_e

# 各s値に対するゼーベック係数を計算して出力
alpha_e_32 = calculate_alpha_e(3/2, xi_F)
alpha_e_m12 = calculate_alpha_e(-1/2, xi_F)

print("ξ_F:", xi_F)
print("ゼーベック係数 α_e (s = 3/2):", alpha_e_32 * 1000, "[mV/K]")
print("ゼーベック係数 α_e (s = -1/2):", alpha_e_m12 * 1000, "[mV/K]")
