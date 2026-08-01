import mpmath
from scipy.constants import h, k, e, pi

# mpmathの精度を設定
mpmath.mp.dps = 30  # 30桁の精度で計算

# 物理定数の定義
hbar = mpmath.mpf(h) / (2 * mpmath.pi)  # 換算プランク定数 (Js)
m_star = mpmath.mpf(9.11e-31)  # 電子の有効質量 (kg)
kb = mpmath.mpf(k)  # ボルツマン定数 (J/K)
q = mpmath.mpf(e)  # 電子の電荷 (C)
T = mpmath.mpf(300)  # 温度 (K)
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
    # フェルミ積分の計算
    F_s_plus_half = fermi_integral(s + 0.5, xi_F)
    
    # キャリア濃度 N_B の計算
    N_B = 2 * ((m_star * kb * T) / (2 * pi * hbar**2))**(3/2)
    
    # 電気伝導率 σ の計算式
    sigma = (4 * q**2 * N_B * tau_0) / (3 * mpmath.sqrt(pi) * m_star) * (s + 1.5) * F_s_plus_half
    return sigma

# s = 3/2 の場合の電気伝導率の計算
sigma_32 = calculate_conductivity(3/2, xi_F)
print("σ (s = 3/2):", sigma_32)

# s = -1/2 の場合の電気伝導率の計算
sigma_m12 = calculate_conductivity(-1/2, xi_F)
print("σ (s = -1/2):", sigma_m12)
