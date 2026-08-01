#表を出力


import tkinter as tk
from tkinter import ttk
import mpmath
from scipy.constants import h, k, e, pi

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
xi_G = E_g / (kb * T)  # ξ_Gの計算

def fermi_dirac_half(xi):
    def integrand(x):
        return mpmath.sqrt(x) / (mpmath.exp(x - xi) + 1)
    return mpmath.quad(integrand, [0, mpmath.inf])

def neutrality_and_carrier_densities(xi_F):
    f_D = 1 / (1 + (1 / g_c) * mpmath.exp((E_D / (kb * T))- xi_F))
    n_D = N_D * f_D
    n = (2 / mpmath.sqrt(mpmath.pi)) * N_C * fermi_dirac_half(xi_F)
    p = (2 / mpmath.sqrt(mpmath.pi)) * N_V * fermi_dirac_half(-xi_F - xi_G)
    neutrality = n - (N_D - n_D) - p
    return (xi_F, n, p, n_D, neutrality)  # 追加: ξ_Fも戻り値に含む

def create_table_window():
    window = tk.Tk()
    window.title("Carrier Densities and Neutrality")
    window.geometry('1000x400')

    columns = ('ξ_F', 'n', 'p', 'n_D', 'neutrality')
    tree = ttk.Treeview(window, columns=columns, show='headings', height='20')
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150)

    # ξ_Fの範囲でデータを計算しテーブルに挿入
    for xi_F in range(-20, 21):
        xi_F_val, n, p, n_D, neutrality = neutrality_and_carrier_densities(xi_F)  # 修正部分
        tree.insert('', 'end', values=(xi_F_val, mpmath.nstr(n, 10), mpmath.nstr(p, 10), mpmath.nstr(n_D, 10), mpmath.nstr(neutrality, 10)))

    tree.pack(expand=True, fill='both')
    window.mainloop()


create_table_window()
