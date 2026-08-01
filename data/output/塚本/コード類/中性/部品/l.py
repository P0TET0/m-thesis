import mpmath
from scipy.constants import h, k, e, pi

# mpmathの精度を設定
mpmath.mp.dps = 30  # 30桁の精度で計算

# 物理定数の定義
hbar = mpmath.mpf(h) / (2 * mpmath.pi)  # 換算プランク定数 (Js)
m_star_e = mpmath.mpf(9.11e-31)  # 電子の有効質量_e (kg)
m_star_h = mpmath.mpf(9.11e-31)  # 電子の有効質量_h (kg)
kb = mpmath.mpf(k)  # ボルツマン定数 (J/K)
q = mpmath.mpf(e)  # 電子の電荷 (C)
T = mpmath.mpf(300)  # 温度 (K)
tau_e = mpmath.mpf(1e-14)  # 緩和時間_e (s)
tau_h = mpmath.mpf(1e-14)  # 緩和時間_h (s)
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

# フェルミ積分_hの関数定義
def fermi_integral_h(s, xi_F, xi_g):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = -xi_F - xi_g + xi
        return xi**s / (mpmath.exp(exponent) + 1)
    
    # mpmathの数値積分関数を使用
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# 電気伝導率 σ_e の計算関数
def calculate_conductivity_e(s, xi_F):
    # フェルミ積分の計算
    F_s_plus_half = fermi_integral_e(s + 0.5, xi_F)
    
    # N_C の計算
    N_C = 2 * ((m_star_e * kb * T) / (2 * pi * hbar**2))**(3/2)
    
    # σ_e の計算式
    sigma_e = (4 * q**2 * N_C * tau_e) / (3 * mpmath.sqrt(pi) * m_star_e) * (s + 1.5) * F_s_plus_half
    return sigma_e

# 電気伝導率 σ_h の計算関数
def calculate_conductivity_h(s, xi_F, xi_g):
    # フェルミ積分の計算
    F_s_plus_half = fermi_integral_h(s + 0.5, xi_F, xi_g)
    
    # キャリア濃度 N_V の計算
    N_V = 2 * ((m_star_h * kb * T) / (2 * pi * hbar**2))**(3/2)
    
    # σ_h の計算式
    sigma_h = (4 * q**2 * N_V * tau_h) / (3 * mpmath.sqrt(pi) * m_star_h) * (s + 1.5) * F_s_plus_half
    return sigma_h

# 電気伝導率 σ の計算関数
def calculate_conductivity(s, xi_F, xi_g):  
    # σ_e, σ_h の計算
    sigma_e = calculate_conductivity_e(s, xi_F)
    sigma_h = calculate_conductivity_h(s, xi_F, xi_g)
    
    # 電気伝導率 σ の計算式
    sigma = sigma_e + sigma_h
    return sigma

# ゼーベック係数 α_e の計算関数
def calculate_alpha_e(s, xi_F):
    F_s_plus_32 = fermi_integral_e(s + 3/2, xi_F)
    F_s_plus_12 = fermi_integral_e(s + 1/2, xi_F)
    small_delta_e = ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_plus_12)
    alpha_e = -((kb / q) * (small_delta_e - xi_F))
    return alpha_e

# ゼーベック係数 α_h の計算関数
def calculate_alpha_h(s, xi_F, xi_g):
    F_s_plus_52 = fermi_integral_h(s + 5/2, xi_F, xi_g)
    F_s_plus_12 = fermi_integral_h(s + 1/2, xi_F, xi_g)
    small_delta_h = ((s + 5/2) * F_s_plus_52) / ((s + 3/2) * F_s_plus_12)
    alpha_h = -((kb / q) * (small_delta_h - (-xi_F - xi_g)))
    return alpha_h

# ゼーベック係数 α の計算関数
def calculate_alpha(s, xi_F, xi_g):
    sigma_e = calculate_conductivity_e(s, xi_F)
    sigma_h = calculate_conductivity_h(s, xi_F, xi_g)
    sigma = calculate_conductivity(s, xi_F, xi_g)

    alpha_e = calculate_alpha_e(s, xi_F)
    alpha_h = calculate_alpha_h(s, xi_F, xi_g)
    
    alpha = ((alpha_e * sigma_e) + (alpha_h * sigma_h)) / sigma
    return alpha

# Δ_e(ξ_F)_の計算関数
def calculate_big_delta_e(s, xi_F):
    # フェルミ積分の計算
    F_s_plus_52 = fermi_integral_e(s + 5/2, xi_F)
    F_s_plus_12 = fermi_integral_e(s + 1/2, xi_F)
    
    # δ_e(ξ_F)の計算
    small_delta_e = calculate_small_delta_e(s, xi_F)

    # Δ_e(ξ_F)の計算式
    big_delta_e = (((s + 7/2) * F_s_plus_52) / ((s + 3/2) * F_s_plus_12)) - small_delta_e**2
    return big_delta_e

# Δ_h(-ξ_F - ξ_g)の計算関数
def calculate_big_delta_h(s, xi_F, xi_g):
    # フェルミ積分を -ξ_F - ξ_g の形で計算
    F_s_plus_52 = fermi_integral_h(s + 5/2, -xi_F, -xi_g)
    F_s_plus_12 = fermi_integral_h(s + 1/2, -xi_F, -xi_g)

    # δ_h(-ξ_F - ξ_g)の計算
    small_delta_h = calculate_small_delta_h(s, xi_F, xi_g)

    # Δ_h(-ξ_F - ξ_g)の計算式
    big_delta_h = (((s + 7/2) * F_s_plus_52) / ((s + 3/2) * F_s_plus_12)) - small_delta_h**2
    return big_delta_h

# δ_e(ξ_F)の計算関数
def calculate_small_delta_e(s, xi_F):
    F_s_plus_32 = fermi_integral_e(s + 3/2, xi_F)
    F_s_minus_12 = fermi_integral_e(s + 1/2, xi_F)
    return ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_minus_12)

# δ_h(-ξ_F - ξ_g)の計算関数
def calculate_small_delta_h(s, xi_F, xi_g):
    F_s_plus_32 = fermi_integral_h(s + 3/2, xi_F, xi_g)
    F_s_minus_12 = fermi_integral_h(s + 1/2, xi_F, xi_g)
    return ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_minus_12)

# L_e の計算
def calculate_lorentz_number_e(s, xi_F):
    big_delta_e = calculate_big_delta_e(s, xi_F)
    L_e = (kb / q) ** 2 * big_delta_e
    return L_e

# L_h の計算
def calculate_lorentz_number_h(s, xi_F, xi_g):
    big_delta_h = calculate_big_delta_h(s, xi_F, xi_g)
    L_h = (kb / q) ** 2 * big_delta_h
    return L_h

# L の計算
def calculate_lorentz_number(s, xi_F, xi_g):
    # 各物理量の計算
    sigma_e = calculate_conductivity_e(s, xi_F)
    sigma_h = calculate_conductivity_h(s, xi_F, xi_g)
    sigma = calculate_conductivity(s, xi_F, xi_g)
    alpha_e = calculate_alpha_e(s, xi_F)
    alpha_h = calculate_alpha_h(s, xi_F, xi_g)

    big_delta_e = calculate_big_delta_e(s, xi_F)
    big_delta_h = calculate_big_delta_h(s, xi_F, xi_g)
    L_e = (kb / q) ** 2 * big_delta_e
    L_h = (kb / q) ** 2 * big_delta_h

    L = (((L_e * sigma_e) + (L_h * sigma_h)) / sigma) + ((sigma_e * sigma_h * (alpha_e - alpha_h) ** 2) / (sigma ** 2))
    return L

# 各s値に対する L_e を計算して出力
L_e_32 = calculate_lorentz_number_e(3/2, xi_F)
L_e_m12 = calculate_lorentz_number_e(-1/2, xi_F)

print("ξ_F:", xi_F)
print("ローレンツ数 L_e (s = 3/2):", L_e_32 * 10**8, "[×10^-8 WΩK^-2]")
print("ローレンツ数 L_e (s = -1/2):", L_e_m12 * 10**8, "[×10^-8 WΩK^-2]")

# 各s値に対する L_h を計算して出力
L_h_32 = calculate_lorentz_number_h(3/2, xi_F, xi_g)
L_h_m12 = calculate_lorentz_number_h(-1/2, xi_F, xi_g)

print("ローレンツ数 L_h (s = 3/2):", L_h_32 * 10**8, "[×10^-8 WΩK^-2]")
print("ローレンツ数 L_h (s = -1/2):", L_h_m12 * 10**8, "[×10^-8 WΩK^-2]")

# 各s値に対するローレンツ数 L を計算して出力
L_32 = calculate_lorentz_number(3/2, xi_F, xi_g)
L_m12 = calculate_lorentz_number(-1/2, xi_F, xi_g)

print("ローレンツ数 L (s = 3/2):", L_32 * 10**8, "[×10^-8 WΩK^-2]")
print("ローレンツ数 L (s = -1/2):", L_m12 * 10**8, "[×10^-8 WΩK^-2]")