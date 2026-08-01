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
seebeck_values_32 = []
seebeck_values_m12 = []
lorentz_values_32 = []
lorentz_values_m12 = []
ZT_values_32 = []
ZT_values_m12 = []
ZT_var2_values_32 = []
ZT_var2_values_m12 = []

# フェルミ積分の関数
def fermi_integral(s, xi_F):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - xi_F
        return xi**s / (mpmath.exp(exponent) + 1)
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# ガンマ関数を使用してε(ξ_F)を計算
def calculate_epsilon(s, xi_F):
    F_s_plus_12 = fermi_integral(s + 1/2, xi_F)
    gamma_s_plus_32 = gamma(s + 3/2)
    epsilon = F_s_plus_12 / gamma_s_plus_32
    return epsilon 

# 電気伝導率、ゼーベック係数、ローレンツ数、無次元性能指数の計算関数
def calculate_properties(s, xi_F, B):
    F_s_plus_12 = fermi_integral(s + 1/2, xi_F)
    F_s_plus_32 = fermi_integral(s + 3/2, xi_F)
    F_s_plus_52 = fermi_integral(s + 5/2, xi_F)
    
    N_B = 2 * ((m * kb * T) / (2 * pi * hbar**2))**(3/2)
    sigma = (4 * q**2 * N_B * tau_0) / (3 * mpmath.sqrt(pi) * m) * (s + 1.5) * F_s_plus_12
    
    small_delta = ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_plus_12)
    alpha = (kb/q) * (small_delta - xi_F)
    
    big_delta = (((s + 7/2) * F_s_plus_52) / ((s + 3/2) * F_s_plus_12)) - small_delta**2
    L = (kb / q) ** 2 * big_delta
    
    epsilon_value = calculate_epsilon(s, xi_F)
    ZT = ((small_delta - xi_F) ** 2) / (big_delta + (1 / (B * epsilon_value)))

    return sigma, alpha * 1000, L * 10**8, ZT

# 新たに追加する関数
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

# ZT_var2の計算関数
def calculate_ZT_var2(s, xi_F):
    sigma = calculate_conductivity(s, xi_F)
    alpha = calculate_alpha(s, xi_F)
    L = calculate_lorentz_number(s, xi_F)
    kappa_e = L * T * sigma
    kappa_L = mpmath.mpf(1)  # 定数として仮定
    kappa = kappa_e + kappa_L
    ZT_var2 = (alpha ** 2 * sigma * T) / kappa
    return ZT_var2

# パラメータ B を入力してください
B = float(input("パラメータ B の値を入力してください: "))

# ξ_Fの値に対して各物理量を計算
for xi_F in xi_F_range:
    sigma_32, alpha_32, L_32, ZT_32 = calculate_properties(3/2, xi_F, B)
    sigma_m12, alpha_m12, L_m12, ZT_m12 = calculate_properties(-1/2, xi_F, B)
    
    ZT_var2_32 = calculate_ZT_var2(3/2, xi_F)
    ZT_var2_m12 = calculate_ZT_var2(-1/2, xi_F)
    
    conductivity_values_32.append(float(sigma_32))
    conductivity_values_m12.append(float(sigma_m12))
    seebeck_values_32.append(float(alpha_32))
    seebeck_values_m12.append(float(alpha_m12))
    lorentz_values_32.append(float(L_32))
    lorentz_values_m12.append(float(L_m12))
    ZT_values_32.append(float(ZT_32))
    ZT_values_m12.append(float(ZT_m12))
    ZT_var2_values_32.append(float(ZT_var2_32))
    ZT_var2_values_m12.append(float(ZT_var2_m12))

# グラフを描画
plt.figure(figsize=(12, 12))

# 電気伝導率
plt.subplot(5, 1, 1)
plt.plot(xi_F_range, conductivity_values_32, 'k-', label='σ (s = 3/2)')
plt.plot(xi_F_range, conductivity_values_m12, 'k--', label='σ (s = -1/2)')
plt.ylabel('σ (S/m)')
plt.legend()
plt.xlim(-5, 5)  # x軸の範囲を設定
plt.ylim(min(conductivity_values_m12 + conductivity_values_32), 
    max(conductivity_values_m12 + conductivity_values_32))  # y軸の範囲を動的に設定

# ゼーベック係数
plt.subplot(5, 1, 2)
plt.plot(xi_F_range, seebeck_values_32, 'k-', label='α (s = 3/2)')
plt.plot(xi_F_range, seebeck_values_m12, 'k--', label='α (s = -1/2)')
plt.ylabel('α (mV/K)')
plt.legend()
plt.xlim(-5, 5)
plt.ylim(min(seebeck_values_m12 + seebeck_values_32),
     max(seebeck_values_m12 + seebeck_values_32))

# ローレンツ数
plt.subplot(5, 1, 3)
plt.plot(xi_F_range, lorentz_values_32, 'k-', label='L (s = 3/2)')
plt.plot(xi_F_range, lorentz_values_m12, 'k--', label='L (s = -1/2)')
plt.ylabel('L (×10^-8 WΩK^-2)')
plt.xlabel('ξ_F (還元フェルミエネルギー)')
plt.legend()
plt.xlim(-5, 5)
plt.ylim(min(lorentz_values_m12 + lorentz_values_32), 
    max(lorentz_values_m12 + lorentz_values_32))

# 無次元性能指数 ZT
plt.subplot(5, 1, 4)
plt.plot(xi_F_range, ZT_values_32, 'k-', label='ZT (s = 3/2)')
plt.plot(xi_F_range, ZT_values_m12, 'k--', label='ZT (s = -1/2)')
plt.ylabel('ZT')
plt.xlabel('ξ_F (還元フェルミエネルギー)')
plt.legend()
plt.xlim(-5, 5)
plt.ylim(min(ZT_values_m12 + ZT_values_32), max(ZT_values_m12 + ZT_values_32))

# ZT_var2
plt.subplot(5, 1, 5)
plt.plot(xi_F_range, ZT_var2_values_32, 'k-', label='ZT_var2 (s = 3/2)')
plt.plot(xi_F_range, ZT_var2_values_m12, 'k--', label='ZT_var2 (s = -1/2)')
plt.ylabel('ZT_var2')
plt.xlabel('ξ_F (還元フェルミエネルギー)')
plt.legend()
plt.xlim(-5, 5)
plt.ylim(min(ZT_var2_values_m12 + ZT_var2_values_32), 
    max(ZT_var2_values_m12 + ZT_var2_values_32))

plt.tight_layout()
plt.show()
