import mpmath
from scipy.constants import h, k, e, pi
import scipy.constants as const
import scipy.integrate as integrate
import numpy as np

# mpmathの精度を設定
mpmath.mp.dps = 30  # 30桁の精度で計算

# 定数定義
T = mpmath.mpf(300)  # 絶対温度 (K)
T = 300  # 温度 [K]
hbar = mpmath.mpf(h) / (2 * mpmath.pi)  # 換算プランク定数 (Js)
E_D = 0.026 * const.elementary_charge  # ドナー準位 [J]
E_g = 0.8 * const.elementary_charge  # バンドギャップ [J]
m = mpmath.mpf(9.11e-31)  # 電子の質量 (kg)
m_e = 9.11e-31  # 電子の有効質量 [kg]
m_h = 9.11e-31  # 正孔の有効質量 [kg]
kb = mpmath.mpf(k)  # ボルツマン定数 (J/K)
k_B = const.Boltzmann  # ボルツマン定数 [J/K]
h = const.h  # プランク定数 [Js]
h_bar = const.hbar  # ℏ [Js]
q = const.elementary_charge  # 電気素量 [C]
q = mpmath.mpf(e)  # 電子の電荷 (C)
g_c = 2  # 縮退係数 (電子)
g_v = 4  # 縮退係数 (正孔)
tau_0 = mpmath.mpf(1e-14)  # 緩和時間 (s)

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


# N_C, N_V
def calc_N_C(m_e, T):
    return 2 * ((m_e * k_B * T) / (2 * np.pi * h_bar**2))**(3 / 2)

N_C = calc_N_C(m_e, T)
N_V = calc_N_C(m_h, T)


#  ξ_G
xi_G = E_g / (k_B * T)


# F_(1/2)
def F_half(xi_F):
    def integrand(xi, xi_F):
        return xi**(1/2) / (np.exp(xi - xi_F) + 1)
    return integrate.quad(integrand, 0, np.inf, args=(xi_F,), epsabs=1e-12, epsrel=1e-12)[0]


def neutrality_condition(xi_F):
    n = (2 / np.sqrt(np.pi)) * N_C * F_half(xi_F)
    p = (2 / np.sqrt(np.pi)) * N_V * F_half(-xi_F - xi_G)
    return n, p


# メイン処理部分
if __name__ == "__main__":
    # ユーザーがξ_Fの値を指定
    xi_F = float(input("ξ_Fの値を入力してください: "))
    s = 1.5  # フェルミ分布のsパラメータ（例として1.5を設定）

    # 各種計算の実行
    conductivity = calculate_conductivity(s, xi_F)
    alpha = calculate_alpha(s, xi_F)
    lorentz_number = calculate_lorentz_number(s, xi_F)
    n, p = neutrality_condition(xi_F)

    # 結果の出力
    print(f"ξ_F: {xi_F}")
    print(f"電子密度 n: {n}")
    print(f"正孔密度 p: {p}")
    print(f"ゼーベック係数 α: {alpha}")
    print(f"電気伝導率 σ: {conductivity}")
    print(f"ローレンツ数 L: {lorentz_number}")
