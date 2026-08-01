# ZT=α^2×σ×T/ κ


import mpmath
from scipy.constants import h, k, e, pi

# mpmathの精度を設定
mpmath.mp.dps = 30  # 30桁の精度で計算

# 物理定数の定義
hbar = mpmath.mpf(h) / (2 * mpmath.pi)  # 換算プランク定数 (Js)
m = mpmath.mpf(9.11e-31)  # 電子の質量 (kg)
kb = mpmath.mpf(k)  # ボルツマン定数 (J/K)
q = mpmath.mpf(e)  # 電子の電荷 (C)
T = mpmath.mpf(300)  # 絶対温度 (K)
tau_0 = mpmath.mpf(1e-14)  # 緩和時間 (s)

# ξ_Fの計算
xi_F = 0  # ξ_Fを0として定義
print("ξ_F:", xi_F)

# フェルミ積分の関数定義
def fermi_integral(s, xi_F):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - xi_F
        return xi**s / (mpmath.exp(exponent) + 1)
    
    # mpmathの数値積分関数を使用
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# 電気伝導率 σ の計算関数
def calculate_conductivity(s, xi_F):
    F_s_plus_half = fermi_integral(s + 0.5, xi_F)
    N_B = 2 * ((m * kb * T) / (2 * pi * hbar**2))**(3/2)
    sigma = (4 * q**2 * N_B * tau_0) / (3 * mpmath.sqrt(pi) * m) * (s + 1.5) * F_s_plus_half
    return sigma

# ゼーベック係数αの計算関数
def calculate_alpha(s, xi_F):
    F_s_plus_32 = fermi_integral(s + 3/2, xi_F)
    F_s_plus_12 = fermi_integral(s + 1/2, xi_F)
    small_delta = ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_plus_12)
    alpha = (kb / q) * (small_delta - xi_F)
    return alpha

# ローレンツ数Lの計算関数
def calculate_lorentz_number(s, xi_F):
    F_s_plus_52 = fermi_integral(s + 5/2, xi_F)
    F_s_plus_12 = fermi_integral(s + 1/2, xi_F)
    big_delta = (((s + 7/2) * F_s_plus_52) / ((s + 3/2) * F_s_plus_12))
    L = (kb / q) ** 2 * big_delta
    return L

# 無次元性能指数ZTの計算
def calculate_ZT(s, xi_F):
    sigma = calculate_conductivity(s, xi_F)
    alpha = calculate_alpha(s, xi_F)
    L = calculate_lorentz_number(s, xi_F)
    kappa_e = L * T * sigma
    kappa_L = mpmath.mpf(1)  # 定数
    kappa = kappa_e + kappa_L
    ZT = (alpha ** 2 * sigma * T) / kappa
    return ZT

# s = 3/2 の場合の無次元性能指数ZTの計算
ZT_32 = calculate_ZT(3/2, xi_F)
print("ZT (s = 3/2):", ZT_32)

# s = -1/2 の場合の無次元性能指数ZTの計算
ZT_m12 = calculate_ZT(-1/2, xi_F)
print("ZT (s = -1/2):", ZT_m12)
