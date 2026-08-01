import numpy as np
import matplotlib.pyplot as plt
import mpmath
from scipy.constants import h, k, e, pi
from scipy.special import gamma

# 物理定数の定義
hbar = mpmath.mpf(h) / (2 * mpmath.pi)
m = mpmath.mpf(9.11e-31)
kb = mpmath.mpf(k)
q = mpmath.mpf(e)
T = mpmath.mpf(300)
tau_0 = mpmath.mpf(1e-14)

# mpmathの精度を設定
mpmath.mp.dps = 30

# ξ_Fの範囲を定義 (-5 から +5)
xi_F_range = np.linspace(-5, 5, 100)

# 計算結果を格納するリスト
conductivity_values_32 = []
conductivity_values_m12 = []

# フェルミ積分の関数
def fermi_integral(s, xi_F):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - xi_F
        return xi**s / (mpmath.exp(exponent) + 1)
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# 電気伝導率の計算
def calculate_conductivity(s, xi_F):
    F_s_plus_half = fermi_integral(s + 0.5, xi_F)
    N_B = 2 * ((m * kb * T) / (2 * pi * hbar**2))**(3/2)
    sigma = (4 * q**2 * N_B * tau_0) / (3 * mpmath.sqrt(pi) * m) * (s + 1.5) * F_s_plus_half
    return sigma

# ξ_Fの値に対して電気伝導率を計算
for xi_F in xi_F_range:
    sigma_32 = calculate_conductivity(3/2, xi_F)
    sigma_m12 = calculate_conductivity(-1/2, xi_F)
    conductivity_values_32.append(sigma_32)
    conductivity_values_m12.append(sigma_m12)

# 最大値で正規化
max_sigma_32 = max(conductivity_values_32)
max_sigma_m12 = max(conductivity_values_m12)
normalized_sigma_32 = [sigma / max_sigma_32 for sigma in conductivity_values_32]
normalized_sigma_m12 = [sigma / max_sigma_m12 for sigma in conductivity_values_m12]

# グラフを描画
plt.figure(figsize=(12, 10))
plt.semilogy(xi_F_range, normalized_sigma_32, 'k-', label='σ (s = 3/2) normalized')
plt.semilogy(xi_F_range, normalized_sigma_m12, 'k--', label='σ (s = -1/2) normalized')
plt.ylabel('Normalized σ (log scale)')
plt.xlabel('ξ_F (Reduced Fermi Energy)')
plt.legend()
plt.xlim(-5, 5)
plt.ylim(1e-3, 1)  # y軸の範囲を動的に設定
plt.tight_layout()
plt.show()

