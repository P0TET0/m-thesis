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
#T = mpmath.mpf(300)  # 温度 [K] T を変化させるので定数ではない。
tau_e = mpmath.mpf(1e-14)  # 緩和時間_e [s]
tau_h = mpmath.mpf(1e-14)  # 緩和時間_h [s]
E_g = mpmath.mpf(0.910022) * q  # バンドギャップ [eV]を[J]に変換
N_A = N_A  # アボガドロ数 [1/mol] # この行はなくても動作する。

# N_C, N_Vの計算（探索コードに基づく追加部分）
def N_C(T): return 2 * ((m_star_e * kb * T) / (2 * mpmath.pi * hbar**2))**(3/2)
def N_V(T): return 2 * ((m_star_h * kb * T) / (2 * mpmath.pi * hbar**2))**(3/2)
def xi_g(T): return E_g / (kb * T)  # ξ_Gの計算
T_sample=mpmath.mpf(300)
N_D_sample = mpmath.mpf(5e26)  # ドナー濃度 [1/m^3]
E_D = mpmath.mpf(0.026) * q  # ドナー準位 [eV]を[J]に変換
def xi_D(T): return E_D / (kb *T) # ξ_Dの計算
g_c = mpmath.mpf(2)  # 縮逆係数
g_v = mpmath.mpf(4)  # 縮逆係数

# フェルミ統計の関数定義
# 中性条件とキャリア濃度の計算
def fermi_dirac_half(xi):
  def integrand(x):
    return mpmath.sqrt(x)/ (mpmath.exp(x-xi)+1)
  return mpmath.quad(integrand,[0,mpmath.inf])
def f_D(xi_F,T): return (1 + (1/g_c) * mpmath.exp(xi_D(T) - xi_F))**(-1)
def n_D(xi_F,N_D,T): return N_D*f_D(xi_F,T)
def n(xi_F,T): return (2 / mpmath.sqrt(mpmath.pi)) * N_C(T) * fermi_dirac_half(xi_F)
def p(xi_F,T): return (2 / mpmath.sqrt(mpmath.pi)) * N_V(T) * fermi_dirac_half(-xi_F - xi_g(T))
def neutrality(N_D,T):
  def func(xi_F): return n(xi_F,T) - N_D*(1 - f_D(xi_F,T)) - p(xi_F,T)
  return func
def neutral_xi_F(N_D,T,**kwarg): return mpmath.findroot(neutrality(N_D,T),0,**kwarg)

N_D_values = [mpmath.mpf(2.15e25), mpmath.mpf(4.64e25), mpmath.mpf(1e26), mpmath.mpf(2.15e26), mpmath.mpf(4.64e26)]
N_D_values = [2.15e25, 4.64e25, 1e26, 2.15e26, 4.64e26]
T_range = np.linspace(10,1300,100) # 温度範囲 (10K～1300K)

xi_F_vals = [[neutral_xi_F(N_D,T,tol=1e-50) for T in T_range] for N_D in N_D_values]

plt.figure(figsize=(10, 6))
for i,N_D in enumerate(N_D_values):
  plt.plot(T_range, xi_F_vals[i], label=f'N_D = {N_D}')
plt.xlabel('T')
plt.ylabel('xi_F')
plt.legend()
plt.show()

with open('xi_F_vals.pkl', 'wb') as f:
    pickle.dump(xi_F_vals, f)
with open('N_D_values.pkl', 'wb') as f:
    pickle.dump(N_D_values, f)
with open('T_range.pkl', 'wb') as f:
    pickle.dump(T_range, f)