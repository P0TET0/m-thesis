import mpmath
import numpy as np
from scipy.constants import h, k, e, pi, N_A
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pickle

# mpmathの精度を設定
mpmath.mp.dps = 50  # 小数点以下50桁の精度で計算

# 物理定数の定義
hbar = mpmath.mpf(h) / (2 * mpmath.pi)  # 換等プランク定数 [Js]
m_star = mpmath.mpf(1.4 * 9.11e-31)  # 電子の有効質量 [kg]
m_star_e = mpmath.mpf(1.4 * 9.11e-31)  # 電子の有効質量_e [kg]
m_star_h = mpmath.mpf(1.4 * 9.11e-31)  # 正孔の有効質量_h [kg]
kb = mpmath.mpf(k)  # ボルツマン定数 [J/K]
q = mpmath.mpf(e)  # 電子の電荷 [C]
tau_e = mpmath.mpf(1e-14)  # 緩和時間_e [s]
tau_h = mpmath.mpf(1e-14)  # 緩和時間_h [s]
E_g = mpmath.mpf(0.910022) * q  # バンドギャップ [eV]を[J]に変換
N_A = N_A  # アボガドロ数 [1/mol]

with open('T_range.pkl', 'rb') as f:
    T_range = pickle.load(f)
with open('N_D_values.pkl', 'rb') as f:
    N_D_values = pickle.load(f)
with open('xi_F_vals.pkl', 'rb') as f:
    xi_F_vals = pickle.load(f)

# 入力パラメータ y の更新
y=mpmath.mpf(0.8) # 入力パラメータ yの例

# parameters 定数
beta=mpmath.mpf(2.0)   # 正常過程とウムクラップ過程の緩和時間の比
gamma=mpmath.mpf(0.91) # グリュナイゼン定数
E=mpmath.mpf(2.94*e)   # 音響フォノン変形ポテンシャル定数
epsilon_S=mpmath.mpf(39) # 点欠陥散乱における歪みパラメータ
L=mpmath.mpf(1e-6)     # 粒形 [m]

delta_M=72.59-28.086   # 質量の差 [g]
delta_a=(2.8288-2.7155)*(1e-10) # 原子体積の差 [m]

def a_cubed(y): return (2.7155e-10)**3*(1-y)+(2.8288e-10)**3*y # 平均原子体積
def a(y): return a_cubed(y)**(1/3)  # 平均原子の３乗根
def M_g(y): return (28.086)*(1-y)+(72.59)*y  # 原子の平均質量 [g]
def M_kg(y): return M_g(y)*(1e-3)   # 原子の平均質量 [kg]
def G(y): return (1.033)*(1-y)+(1.017)*y*(1e-3) # 弾性定数
def Theta(y): return (1.48e-8)*a(y)**(-3/2)*(M_g(y)**(-1/2))*G(y) # デバイ温度 [K]
def v_s(y): return (kb/hbar)*(6*np.pi**2)**(-1/3)*Theta(y)*a(y)       # 平均音速 [m/s]
def rho_d(y): return M_kg(y)/(a_cubed(y)**(3/2)) # 物質の密度 [kg/m^3]
def phi(y,T): return (m_star * v_s(y=y))/(2*kb*T) # 音速で運動するキャリアの還元フェルミエネルギー

# 1/τ_N の定義
def tau_N_inv(xi, T):
    # xi は積分中で変動する値
    term1 = ((20 * np.pi) / 3) * hbar * N_A * ((6 * np.pi ** 2) / 4) ** (1 / 3) * (beta * (1 + (5 / 9) * beta) / (1 + beta)) * (gamma ** 2) / (M_kg(y) * a(y) ** 2) * (T / Theta(y)) ** 3 * xi ** 2
    return term1

# 1/τ_U の定義
def tau_U_inv(tau_N_inv_val):
    return (1 / beta) * tau_N_inv_val

# 1/τ_ep の定義
def tau_ep_inv(xi, xi_F):
    term1 = (E ** 2 * m_star ** 3 * v_s(y)) / (4 * np.pi * hbar ** 4 * rho_d(y) * phi(y,T))
    term2 = mpmath.log((1 + mpmath.exp(phi - xi_F + (xi ** 2 / (16 * phi(y,T) ** 2)) + xi / 2)) / (1 + mpmath.exp(phi(y,T) - xi_F + (xi ** 2 / (16 * phi(y,T) ** 2) - xi / 2))))
    return term1 * (xi - term2)

# 1/τ_pd の定義
def tau_pd_inv(xi, T, y):
    term1 = (1 / (4 * np.pi)) * ((a(y) / v_s(y))**3) * ((kb * T / hbar)**4) * y * (1 - y)
    term2 = ((delta_M / M_g(y)) ** 2 + epsilon_S * (delta_a / a(y)) ** 2)
    return term1 * term2 * xi ** 4

# 1/τ_gb の定義
def tau_gb_inv():
    return v_s(y) / L

# 1/τ_C の計算
def tau_C_inv(xi, xi_F, T, y):
    # τ_N と τ_U と τ_ep と τ_pd と τ_gb を考慮
    τ_N_inv_val = tau_N_inv(xi, T)
    τ_U_inv_val = tau_U_inv(τ_N_inv_val,)
    #τ_ep_inv_val = tau_ep_inv(xi, xi_F)
    #τ_pd_inv_val = tau_pd_inv(xi, v_s, T, y)
    τ_gb_inv_val = tau_gb_inv()
    #return τ_N_inv_val + τ_U_inv_val + τ_ep_inv_val + τ_pd_inv_val + τ_gb_inv_val
    return τ_N_inv_val + τ_U_inv_val + τ_gb_inv_val

# τ_C を積分中で動的に計算
def tau_C(xi, xi_F, T, y):
    τ_C_val = 1 / tau_C_inv(xi, xi_F, T, y)
    return τ_C_val

# I1の積分
def I1_integrand(xi, xi_F,  T, y):
    τ_C_val = tau_C(xi, xi_F, T, y)  # xi に依存する τ_C の計算
    return xi ** 4 * np.exp(-xi) / (1 - np.exp(-xi)) ** 2

def I1(xi_F, T, y):
    return quad(I1_integrand, 0, Theta(y) / T, args=(xi_F, T, y))[0]

# I2の積分
def I2_integrand(xi, xi_F, T, y):
    τ_C_val = tau_C(xi, xi_F, T, y)  # xi に依存する τ_C の計算
    τ_N_val = 1 / tau_N_inv(xi, T)  # xi に依存する τ_N の計算
    return (τ_C_val / τ_N_val) * (xi ** 4 * np.exp(xi)) / ((np.exp(xi) - 1) ** 2)

def I2(xi_F, T, y):
    return quad(I2_integrand, 0, Theta(y) / T, args=(xi_F, T, y))[0]

# I3の積分
def I3_integrand(xi, xi_F, T, y):
    τ_C_val = tau_C(xi, xi_F, T, y)  # xi に依存する τ_C の計算
    τ_N_val = 1 / tau_N_inv(xi, T)  # xi に依存する τ_N の計算
    return (1 / τ_N_val) * (1 - (τ_C_val / τ_N_val)) * (xi ** 4 * np.exp(xi)) / ((np.exp(xi) - 1) ** 2)

def I3(xi_F, T, y):
    return quad(I3_integrand, 0, Theta(y) / T, args=(xi_F, T, y))[0]

# 格子熱伝導率 κ_L の計算
def lattice_thermal_conductivity(xi_F, T, y):
    return (k / (2 * np.pi ** 2 * v_s(y))) * ((k * T) / hbar) ** 3 * (I1(xi_F, T, y) + (I2(xi_F, T, y) ** 2) / I3(xi_F, T, y))

def confirm_conductivity_coefficients(T_range):
    results = {"T": T_range}  # T のリスト
    for j, N_D in enumerate(N_D_values):  # すべての N_D を処理
        κ_L_list = []
        for i, T in enumerate(T_range):
            xi_F = xi_F_vals[j][i]  # j 番目の N_D に対応する xi_F を取得
            κ_L = lattice_thermal_conductivity(xi_F, T, y)
            κ_L_list.append(float(κ_L))
        results[f"κ_L (N_D={N_D:.2e})"] = κ_L_list  # カラム名に N_D を含める
    return results

results = confirm_conductivity_coefficients(T_range)

plt.figure(figsize=(10, 6))

for key in results:
    if "κ_L" in key:  # すべての κ_L を含むキーを取得
        plt.plot(results["T"], results[key], label=key)  # 各 N_D ごとにプロット

plt.xlabel("T [K]")
plt.ylabel("κ_L [W/mK]")
plt.legend()
plt.grid(True)
plt.show()