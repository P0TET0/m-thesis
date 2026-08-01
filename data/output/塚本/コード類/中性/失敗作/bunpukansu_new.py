import scipy.constants as const
import scipy.integrate as integrate
import numpy as np

# 定数定義
T = 300  # 温度 [K]
N_D = 1e26  # ドナー濃度 [1/m^3]
E_D = 0.026 * const.elementary_charge  # ドナー準位 [J]
E_g = 0.8 * const.elementary_charge  # バンドギャップ [J]
m_e = 9.11e-31  # 電子の有効質量 [kg]
m_h = 9.11e-31  # 正孔の有効質量 [kg]
k_B = const.Boltzmann  # ボルツマン定数 [J/K]
h = const.h  # プランク定数 [Js]
h_bar = const.hbar  # ℏ [Js]
q = const.elementary_charge  # 電気素量 [C]
g_c = 2  # 縮退係数 (電子)
g_v = 4  # 縮退係数 (正孔)

# 縮退密度 N_C, N_V の計算
def calc_N_C(m_e, T):
    return 2 * ((m_e * k_B * T) / (2 * np.pi * h_bar**2))**(3 / 2)

N_C = calc_N_C(m_e, T)
N_V = calc_N_C(m_h, T)

# バンドギャップ ξ_G の計算
xi_G = E_g / (k_B * T)

# フェルミ分布関数 f_D の計算
def f_D(E_D, xi_F, g_c):
    return 1 / (1 + (1 / g_c) * np.exp((E_D / (k_B * T)) - xi_F))

# Fermi-Dirac 分布 F_(1/2) の計算
def F_half(xi_F):
    def integrand(xi, xi_F):
        return xi**(1/2) / (np.exp(xi - xi_F) + 1)
    return integrate.quad(integrand, 0, np.inf, args=(xi_F,), epsabs=1e-12, epsrel=1e-12)[0]

# 中性条件を満たす ξ_F の探索
from scipy.optimize import root

def neutrality_condition(xi_F):
    # n_D, n, p の計算
    n_D = N_D * f_D(E_D, xi_F, g_c)
    n = (2 / np.sqrt(np.pi)) * N_C * F_half(xi_F)
    p = (2 / np.sqrt(np.pi)) * N_V * F_half(-xi_F - xi_G)
    result = n - (N_D - n_D) - p
    return result

# 初期推定値と探索範囲を与えて収束確認
xi_F_min, xi_F_max = 0, 1  # ξ_Fの探索範囲
xi_F_current = xi_F_min
max_iterations = 1000
step_size = 0.1  # 初期の大きい探索間隔
final_tolerance = 1e-10
found_solution = False

for i in range(max_iterations):
    result = neutrality_condition(xi_F_current)
    print(f"Iteration {i+1}: xi_F = {xi_F_current:.4f}, n - (N_D - n_D) - p = {result:.4e}")  # デバッグ用出力
    if abs(result) < final_tolerance:
        found_solution = True
        break
    # 大きい探索間隔でξ_Fの範囲内を探索
    if xi_F_min <= xi_F_current <= xi_F_max:
        xi_F_current += step_size
    else:
        # ξ_Fが範囲外に出た場合は範囲と間隔を狭める
        xi_F_min = max(xi_F_min, xi_F_current - step_size / 2)
        xi_F_max = min(xi_F_max, xi_F_current + step_size / 2)
        step_size /= 2
        xi_F_current = xi_F_min
        print(f"探索範囲を狭めます: 新しい範囲は [{xi_F_min:.4f}, {xi_F_max:.4f}], 間隔 = {step_size:.4f}")
    
    # 許容範囲内のξ_Fが見つかった場合、間隔を狭めて精度を上げる
    if abs(result) < 1e-2:
        step_size = 0.01
    elif abs(result) < 1e-5:
        step_size = 0.001

if not found_solution:
    print("中性条件を満たす解が見つかりませんでした。許容誤差を満たす範囲で再度探索を行います。")
    xi_F_min, xi_F_max = 0, 1
    step_size = 0.001  # より小さな間隔で再探索
    for i in range(max_iterations):
        result = neutrality_condition(xi_F_current)
        print(f"Iteration {i+1 + max_iterations}: xi_F = {xi_F_current:.4f}, n - (N_D - n_D) - p = {result:.4e}")  # デバッグ用出力
        if abs(result) < final_tolerance:
            found_solution = True
            break
        if xi_F_min <= xi_F_current <= xi_F_max:
            xi_F_current += step_size
        else:
            # ξ_Fが範囲外に出た場合は範囲と間隔を狭める
            xi_F_min = max(xi_F_min, xi_F_current - step_size / 2)
            xi_F_max = min(xi_F_max, xi_F_current + step_size / 2)
            step_size /= 2
            xi_F_current = xi_F_min
            print(f"探索範囲を狭めます: 新しい範囲は [{xi_F_min:.4f}, {xi_F_max:.4f}], 間隔 = {step_size:.4f}")

xi_F_solution = xi_F_current

# 中性条件を満たすξ_Fの値と、その時の n - (N_D - n_D) - p の値を出力
neutrality_value = neutrality_condition(xi_F_solution)

print(f"中性条件を満たすξ_Fの値: {xi_F_solution:.4f}")
print(f"中性条件を満たすときの n - (N_D - n_D) - p の値: {neutrality_value:.4e}")
