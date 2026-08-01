# 1：ZT={δ(ξ_F)-ξ_F}^2 / Δ(ξ_F)+(1/Bε(ξ_F))
# 2：ZT=α^2×σ×T/ κ

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
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

# ξ_Fの範囲を定義
xi_F_range = np.linspace(-5, 5, 100)

# フェルミ積分の関数
def fermi_integral(s, xi_F):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - xi_F
        return xi**s / (mpmath.exp(exponent) + 1)
    return mpmath.quad(integrand, [0, mpmath.inf])

# ZTの計算関数
def calculate_ZT_formula_1(s, xi_F, B):
    F_s_plus_12 = fermi_integral(s + 1/2, xi_F)
    F_s_plus_32 = fermi_integral(s + 3/2, xi_F)
    F_s_plus_52 = fermi_integral(s + 5/2, xi_F)

    small_delta = ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_plus_12)
    big_delta = (((s + 7/2) * F_s_plus_52) / ((s + 3/2) * F_s_plus_12)) - small_delta**2
    epsilon_value = F_s_plus_12 / gamma(s + 3/2)

    ZT = ((small_delta - xi_F) ** 2) / (big_delta + (1 / (B * epsilon_value)))
    return float(ZT)

# Bの値のリスト
B_values = [0.1, 0.2, 0.5, 1.0]
colors = ['b', 'g', 'r', 'c']  # グラフの色

# グラフの準備
fig, axs = plt.subplots(2, 1, figsize=(12, 12))
all_ZT_values = []

# 各Bの値に対して計算とプロット
for s_value, ax in zip([-1/2, 3/2], axs):
    for B, color in zip(B_values, colors):
        ZT_values = [calculate_ZT_formula_1(s_value, xi_F, B) for xi_F in xi_F_range]
        ax.plot(xi_F_range, ZT_values, color+'-', label=f'B={B}')
        all_ZT_values.extend(ZT_values)

    ax.set_title(f'ZT (s={s_value})')
    ax.set_xlabel('還元フェルミエネルギー ξ_F', fontsize=14)
    ax.set_ylabel('無次元性能指数 ZT', fontsize=14)
    ax.legend(fontsize=14)  # 例: 文字サイズを14に設定
    ax.grid(True)

# 軸範囲の設定
min_ZT, max_ZT = min(all_ZT_values), max(all_ZT_values)
for ax in axs:
    ax.set_xlim([-5, 5])
    ax.set_ylim([0, 5])

plt.tight_layout()
plt.show()
