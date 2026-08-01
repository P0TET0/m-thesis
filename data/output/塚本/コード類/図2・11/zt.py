import mpmath
import numpy as np
from scipy.constants import h, k, e, pi
from scipy.special import gamma

# mpmathの精度を設定
mpmath.mp.dps = 30  # 30桁の精度で計算

# 物理定数の定義
hbar = mpmath.mpf(h) / (2 * mpmath.pi)  # 換算プランク定数 (Js)
m = mpmath.mpf(9.11e-31)  # 電子の質量 (kg)
kb = mpmath.mpf(k)  # ボルツマン定数 (J/K)
q = mpmath.mpf(e)  # 電子の電荷 (C)
T = mpmath.mpf(300)  # 温度 (K)
B = 0.4  #パラメータ

# ξ_Fの計算
xi_F = 0  # ξ_Fを0として定義

# フェルミ積分の関数定義
def fermi_integral(s, xi_F):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - xi_F
        return xi**s / (mpmath.exp(exponent) + 1)
    
    # mpmathの数値積分関数を使用
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# Δ(ξ_F)の計算関数
def calculate_big_delta(s, xi_F):
    # フェルミ積分の計算
    F_s_plus_52 = fermi_integral(s + 5/2, xi_F)
    F_s_plus_12 = fermi_integral(s + 1/2, xi_F)
    
    # δ(ξ_F)の計算
    small_delta = calculate_small_delta(s, xi_F)

    # Δ(ξ_F)の計算式
    big_delta = (((s + 7/2) * F_s_plus_52) / ((s + 3/2) * F_s_plus_12)) - small_delta**2
    return big_delta

# δ(ξ_F)の計算関数
def calculate_small_delta(s, xi_F):
    F_s_plus_32 = fermi_integral(s + 3/2, xi_F)
    F_s_minus_12 = fermi_integral(s + 1/2, xi_F)
    return ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_minus_12)

# ガンマ関数を使用してε(ξ_F)を計算
def calculate_epsilon(s, xi_F):
    F_s_plus_12 = fermi_integral(s + 1/2, xi_F)
    gamma_s_plus_32 = gamma(s + 3/2)
    epsilon = F_s_plus_12 / gamma_s_plus_32
    return epsilon 

# ZT の計算
def calculate_figure_of_merit(s, xi_F):
    small_delta = calculate_small_delta(s, xi_F)
    big_delta = calculate_big_delta(s, xi_F)
    epsilon_value = calculate_epsilon(s, xi_F)
    ZT = ((small_delta - xi_F) ** 2) / (big_delta + (1 / (B * epsilon_value)))
    return ZT

# 各s値に対する無次元性能指数を計算して出力
ZT_32 = calculate_figure_of_merit(3/2, xi_F)
ZT_m12 = calculate_figure_of_merit(-1/2, xi_F)

print("ξ_F:", xi_F)
print("無次元性能指数 ZT (s = 3/2):", ZT_32)
print("無次元性能指数 ZT (s = -1/2):", ZT_m12)