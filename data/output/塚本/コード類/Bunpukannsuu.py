import mpmath
from scipy.constants import h, k, e, pi
import matplotlib.pyplot as plt
import matplotlib.table as tbl
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

mpmath.mp.dps = 30  # 精度を小数点以下30桁に設定

# N_C, N_Vの計算
N_C = 2 * (((m_e * kb * T) / (2 * mpmath.pi * hbar**2))**(3/2))
N_V = 2 * (((m_h * kb * T) / (2 * mpmath.pi * hbar**2))**(3/2))

# ξ_Gの計算
xi_G = E_g / (kb * T)

def fermi_dirac_half(xi):
    def integrand(x):
        return mpmath.sqrt(x) / (mpmath.exp(x - xi) + 1)
    result = mpmath.quad(integrand, [0, mpmath.inf])  # 無限大を保持
    return result

def neutrality_condition(xi_F):
    f_D = 1 / (1 + (1 / g_c) * mpmath.exp((E_D / (kb * T)) - xi_F))
    n_D = N_D * f_D
    n = (2 / mpmath.sqrt(mpmath.pi)) * N_C * fermi_dirac_half(xi_F)
    p = (2 / mpmath.sqrt(mpmath.pi)) * N_V * fermi_dirac_half(-xi_F - xi_G)
    return n - (N_D - n_D) - p

def find_fermi_level(start, end, step, tolerance):
    results = []
    initial_guesses = []
    values = []
    for initial_guess in mpmath.arange(start, end + step, step):
        try:
            xi_F_solution = mpmath.findroot(neutrality_condition, initial_guess, solver='secant', tol=mpmath.mpf(tolerance), maxsteps=500)
            value = neutrality_condition(xi_F_solution)
            formatted_value = mpmath.nstr(value, 10)  # 小数点以下10桁まで表示
            result = (float(initial_guess), formatted_value)
        except Exception as e:
            match = re.search(r'(\d+\.\d+e[+\-]\d+)', str(e))
            formatted_value = match.group(1) if match else "Not found"
            result = (float(initial_guess), formatted_value)
        
        # 解を一度だけ results に記録
        results.append(result)
        initial_guesses.append(float(initial_guess))
        values.append(float(formatted_value) if formatted_value != "Not found" else None)

    return results, initial_guesses, values

def display_results_graphically(results, initial_guesses, values):
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    
    # 表の描画をカスタマイズ
    ax[0].axis('tight')
    ax[0].axis('off')
    table_data = [["ξ_F", "n - (N_D - n_D) - p"]] + results
    table = tbl.table(ax[0], cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)  # 表を拡大

    # グラフの描画
    ax[1].plot(initial_guesses, values, 'bo-', label='n - (N_D - n_D) - p')
    ax[1].set_title('ξ_F vs n - (N_D - n_D) - p')
    ax[1].set_xlabel('ξ_F')
    ax[1].set_ylabel('n - (N_D - n_D) - p')
    ax[1].grid(True)
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

# 結果を求める範囲と間隔を設定
start = -20
end = 20
step = 1
tolerance = '1e-10'

# 解を求める
results, initial_guesses, values = find_fermi_level(start, end, step, tolerance)

# 結果をグラフに表示
display_results_graphically(results, initial_guesses, values)
