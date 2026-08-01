import mpmath
import numpy as np
from scipy.constants import h, k, e, pi, N_A
from scipy.integrate import quad
import matplotlib.pyplot as plt

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
E_g = mpmath.mpf(0.13) * q  # バンドギャップ [eV]を[J]に変換
N_A = N_A  # アボガドロ数 (1/mol)


# ξ_Fの初期値（例）
xi_F = 0.8726856
print("ξ_F:", xi_F)

# ξ_gの計算
xi_g = E_g / (kb * T)  # ξ_Gの計算
print("ξ_g:", xi_g)

# フェルミ積分_eの関数定義
def fermi_integral_e(s, xi_F):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - xi_F
        return xi**s / (mpmath.exp(exponent) + 1)
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# フェルミ積分_hの関数定義
def fermi_integral_h(s, xi_F, xi_g):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - (-xi_F - xi_g)
        return xi**s / (mpmath.exp(exponent) + 1)
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# ゼーベック係数 α_e の計算関数
def calculate_alpha_e(s, xi_F):
    F_s_plus_32 = fermi_integral_e(s + 3/2, xi_F)
    F_s_plus_12 = fermi_integral_e(s + 1/2, xi_F)
    small_delta_e = ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_plus_12)
    alpha_e = -((kb / q) * (small_delta_e - xi_F))
    return alpha_e

# ゼーベック係数 α_h の計算関数
def calculate_alpha_h(s, xi_F, xi_g):
    F_s_plus_32 = fermi_integral_h(s + 3/2, xi_F, xi_g)
    F_s_plus_12 = fermi_integral_h(s + 1/2, xi_F, xi_g)
    small_delta_h = ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_plus_12)
    alpha_h = ((kb / q) * (small_delta_h - (-xi_F - xi_g)))
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

# グラフ作成
def confirm_conductivity_coefficients(s, xi_g, xi_F_range):
    results = {"xi_F": [], "alpha_e": [], "alpha_h": [], "alpha": []}
    for xi_F in xi_F_range:
        alpha_e = calculate_alpha_e(s, xi_F)
        alpha_h = calculate_alpha_h(s, xi_F, xi_g)
        alpha = calculate_alpha(s, xi_F, xi_g)

        results["xi_F"].append(float(xi_F))  # mpf を float に変換
        results["alpha_e"].append(float(alpha_e))   # mpf を float に変換
        results["alpha_h"].append(float(alpha_h))   # mpf を float に変換
        results["alpha"].append(float(alpha))   # mpf を float に変換

    # 数値リストの出力
    for i in range(len(results["xi_F"])):
        print(f"xi_F: {results['xi_F'][i]:.3f}, alpha_e: {results['alpha_e'][i]:.3e} [mV/K], alpha_h: {results['alpha_h'][i]:.3e} [mV/K], alpha: {results['alpha'][i]:.3e} [mV/K]")

    # グラフの作成
    plt.figure(figsize=(10, 6))
    plt.plot(results["xi_F"], [s * 1e3 for s in results["alpha_e"]], label="α_e", linestyle='--')
    plt.plot(results["xi_F"], [s * 1e3 for s in results["alpha_h"]], label="α_h", linestyle='-.')
    plt.plot(results["xi_F"], [s * 1e3 for s in results["alpha"]], label="α", linestyle='-')
    plt.axhline(0, color="gray", linestyle=":")
    plt.xlabel("xi_F")
    plt.ylabel("α [mV/K]")
    plt.legend()
    plt.grid(True)
    plt.show()

# xi_F の範囲を指定してゼーベック係数を確認
xi_F_range = [mpmath.mpf(val) for val in np.linspace(-20, 20, 40)]
s = -1/2  # 状態密度の有効質量の次元数
confirm_conductivity_coefficients(s, xi_g, xi_F_range)
