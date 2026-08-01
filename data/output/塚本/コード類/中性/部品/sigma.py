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
print("ξ_F:", xi_F)

# ξ_gの計算
xi_g = E_g / (kb * T)  # ξ_Gの計算


# フェルミ積分(e)の関数定義
def fermi_integral_e(s, xi_F):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - xi_F
        return xi**s / (mpmath.exp(exponent) + 1)
    
    # mpmathの数値積分関数を使用
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# フェルミ積分(h)の関数定義
def fermi_integral_h(s, xi_F, xi_g):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = -xi_F - xi_g + xi
        return xi**s / (mpmath.exp(exponent) + 1)
    
    # mpmathの数値積分関数を使用
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# σ_e の計算関数
def calculate_conductivity_e(s, xi_F):
    # フェルミ積分の計算
    F_s_plus_half = fermi_integral_e(s + 0.5, xi_F)
    
    # N_C の計算
    N_C = 2 * ((m_star_e * kb * T) / (2 * pi * hbar**2))**(3/2)
    
    # σ_e の計算式
    sigma_e = (4 * q**2 * N_C * tau_e) / (3 * mpmath.sqrt(pi) * m_star_e) * (s + 1.5) * F_s_plus_half
    return sigma_e

# σ_h の計算関数
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

# s = 3/2 の場合の σ_e の計算
sigma_e_32 = calculate_conductivity_e(3/2, xi_F)
print("σ_e (s = 3/2):", sigma_e_32)

# s = -1/2 の場合の σ_e の計算
sigma_e_m12 = calculate_conductivity_e(-1/2, xi_F)
print("σ_e (s = -1/2):", sigma_e_m12)

# s = 3/2 の場合の σ_h の計算
sigma_h_32 = calculate_conductivity_h(3/2, xi_F, xi_g)
print("σ_h (s = 3/2):", sigma_h_32)

# s = -1/2 の場合の σ_h の計算
sigma_h_m12 = calculate_conductivity_h(-1/2, xi_F, xi_g)
print("σ_h (s = -1/2):", sigma_h_m12)

# s = 3/2 の場合の電気伝導率 σ の計算
sigma_32 = calculate_conductivity(3/2, xi_F, xi_g)
print("σ (s = 3/2):", sigma_32)

# s = -1/2 の場合の電気伝導率 σ の計算
sigma_m12 = calculate_conductivity(-1/2, xi_F, xi_g)
print("σ (s = -1/2):", sigma_m12)