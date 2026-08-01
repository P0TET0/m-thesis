#途中まで作りかけの計算する分布関数のコードを途中まで復元しようとしたやつ
#復元終わってない

import mpmath
from scipy.constants import h, k, e, pi
import matplotlib.pyplot as plt
import re

# 物理定数の定義
T = mpmath.mpf(300)  # 温度 [K]
kb = mpmath.mpf(k)  # ボルツマン定数 [J/K]
hbar = mpmath.mpf(h) / (2 * mpmath.pi)
q = mpmath.mpf(e)  # 電気素量 [C]
N_D = mpmath.mpf(1e26)  # ドナー濃度 [1/m^3]
E_D = mpmath.mpf(0.026) * q  # ドナー準位 [eV]を[J]に変換
g_c = mpmath.mpf(2)  # 縮退係数
g_v = mpmath.mpf(4)  # 縮退係数
E_g = mpmath.mpf(0.8) * q  # バンドギャップ [eV]を[J]に変換
m_e = mpmath.mpf(9.11e-31)  # 電子の有効質量 [kg]
m_h = mpmath.mpf(9.11e-31)  # 正孔の有効質量 [kg]

# mpmathを使った物理定数の高精度定義
mpmath.mp.dps = 30  # 精度を小数点以下30桁に設定

# N_C, N_Vの計算
N_C = 2 * (((m_e * kb * T) / (2 * mpmath.pi * hbar**2))**(3/2))
N_V = 2 * (((m_h * kb * T) / (2 * mpmath.pi * hbar**2))**(3/2))

# ξ_Gの計算
xi_G = E_g / (kb * T)

def fermi_dirac_half(xi):
    def integrand(x):
        return mpmath.sqrt(x) / (mpmath.exp(x - xi) + 1)
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

def neutrality_condition(xi_F):
    f_D = 1 / (1 + (1 / g_c) * mpmath.exp((E_D / (kb * T))- xi_F))
    n_D = N_D * f_D
    n = (2 / mpmath.sqrt(mpmath.pi)) * N_C * fermi_dirac_half(xi_F)
    p = (2 / mpmath.sqrt(mpmath.pi)) * N_V * fermi_dirac_half(-xi_F - xi_G)
    return n - (N_D - n_D) - p

def find_fermi_level(start, end, step, tolerance):
    saved_xi_F = None
    saved_value = None
    initial_guesses = []
    values = []

    for initial_guess in mpmath.arange(start, end + step, step):
        print(f"Trying ξ_F =  {initial_guess}") # 追加
        try:
            # solver を 'newton' に変更し、maxstep を増やす
            xi_F_solution = mpmath.findroot(neutrality_condition, initial_guess, solver='newton', tol=mpmath.mpf(tolerance), maxstep=3000)
            value = neutrality_condition(xi_F_solution)

            # 絶対値が10^-10以下のとき、結果を保存して処理を終了
            if mpmath.fabs(value) < mpmath.mpf('1e-10'):
                saved_xi_F = xi_F_solution
                saved_value = value
                print(f"Solution found: ξ_F = {saved_xi_F}, n - (N_D - n_D) - p = {saved_value}")
                break

            formatted_value = mpmath.ntsr(value, 10)
        except Exception as e:
            print(f"Error at ξ_F = {initial_guess}: {e}") # エラーの中間出力
            match = re.search(r'(\d+\.\d+e[+\-]\d+)', str(e))
            

