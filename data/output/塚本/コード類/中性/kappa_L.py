import numpy as np
from scipy.integrate import quad
from scipy.constants import k, hbar, N_A

# 定数の定義
k_B = k  # ボルツマン定数 (J/K)
ħ = hbar  # プランク定数のディラック値 (J s)
N_A = N_A  # アボガドロ数 (1/mol)

# 入力パラメータ y
y = float(input("y の値を入力して下さい (0 ～ 1): "))


# パラメータの更新
a_cubed = (((2.7155e-10) ** 3) * (1 - y)) + (((2.8288e-10) ** 3) * y)  # 平均原子体積
a = a_cubed ** (1/3)  # 三乗根

M_g = (28.086 * (1 - y) + 72.59 * y)  # 原子の平均質量 [g]
M_kg = (28.086 * (1 - y) + 72.59 * y) * 1e-3  # 原子の平均質量[kg]

G = (1.033 * (1 - y) + 1.017 * y) * 1e-3

print("a:", a)
print("M:", M_g, "[g]")
print("G:", G)

# デバイ温度の計算
Θ = 1.48e-8 * (a ** (-3/2)) * (M_g ** (-1/2)) * G  # デバイ温度 (K)  
print("Θ:", Θ)

# 平均音速の計算
v_s = (k_B / ħ) * ((6 * (np.pi ** 2)) ** (-1/3)) * Θ * a  # 平均音速 (m/s)
print("v_s:", v_s)

# 与えられたパラメータ
β = 2.0  # 正常過程とウムクラップ過程の緩和時間の比
γ = 0.91  # グリュナイゼン定数
T = 300  # 温度 (K)

# 1/τ_N の定義
def tau_N_inv(xi, T, Θ, β, γ, M_kg, a):
    # xi は積分中で変動する値
    factor = ((20 * np.pi) / 3) * ħ * N_A * ((6 * np.pi ** 2) / 4) ** (1 / 3) * (β * (1 + (5 / 9) * β) / (1 + β)) * (γ ** 2) / (M_kg * a ** 2) * (T / Θ) ** 3 * xi ** 2
    return factor

# 1/τ_U の定義
def tau_U_inv(tau_N_inv_val, β):
    return (1 / β) * tau_N_inv_val

# 1/τ_C の計算
def tau_C_inv(xi, T, Θ, β, γ, M_kg, a):
    # τ_N と τ_U のみを考慮
    τ_N_inv_val = tau_N_inv(xi, T, Θ, β, γ, M_kg, a)
    τ_U_inv_val = tau_U_inv(τ_N_inv_val, β)
    return τ_N_inv_val + τ_U_inv_val

# τ_C を積分中で動的に計算
def tau_C(xi, T, Θ, β, γ, M_kg, a):
    τ_C_val = 1 / tau_C_inv(xi, T, Θ, β, γ, M_kg, a)
    return τ_C_val

# I1の積分
def I1_integrand(xi, T, Θ, β, γ, M_kg, a):
    τ_C_val = tau_C(xi, T, Θ, β, γ, M_kg, a)  # xi に依存する τ_C の計算
    return τ_C_val * (xi ** 4 * np.exp(xi)) / ((np.exp(xi) - 1) ** 2)

def I1(T, Θ, β, γ, M_kg, a):
    return quad(I1_integrand, 0, Θ / T, args=(T, Θ, β, γ, M_kg, a))[0]

# I2の積分
def I2_integrand(xi, T, Θ, β, γ, M_kg, a):
    τ_C_val = tau_C(xi, T, Θ, β, γ, M_kg, a)  # xi に依存する τ_C の計算
    τ_N_val = 1 / tau_N_inv(xi, T, Θ, β, γ, M_kg, a)  # xi に依存する τ_N の計算
    return (τ_C_val / τ_N_val) * (xi ** 4 * np.exp(xi)) / ((np.exp(xi) - 1) ** 2)

def I2(T, Θ, β, γ, M_kg, a):
    return quad(I2_integrand, 0, Θ / T, args=(T, Θ, β, γ, M_kg, a))[0]

# I3の積分
def I3_integrand(xi, T, Θ, β, γ, M_kg, a):
    τ_C_val = tau_C(xi, T, Θ, β, γ, M_kg, a)  # xi に依存する τ_C の計算
    τ_N_val = 1 / tau_N_inv(xi, T, Θ, β, γ, M_kg, a)  # xi に依存する τ_N の計算
    return (1 / τ_N_val) * (1 - (τ_C_val / τ_N_val)) * (xi ** 4 * np.exp(xi)) / ((np.exp(xi) - 1) ** 2)

def I3(T, Θ, β, γ, M_kg, a):
    return quad(I3_integrand, 0, Θ / T, args=(T, Θ, β, γ, M_kg, a))[0]

# 格子熱伝導率 κ_L の計算
def lattice_thermal_conductivity(T, Θ, v_s, k_B, ħ, β, γ, M_kg, a):
    I1_val = I1(T, Θ, β, γ, M_kg, a)
    I2_val = I2(T, Θ, β, γ, M_kg, a)
    I3_val = I3(T, Θ, β, γ, M_kg, a)

    print(f"I1: {I1_val}")
    print(f"I2: {I2_val}")
    print(f"I3: {I3_val}")
    
    κ_L = (k_B / (2 * np.pi ** 2 * v_s)) * ((k_B * T) / ħ) ** 3 * (I1_val + (I2_val ** 2) / I3_val)
    return κ_L

# 格子熱伝導率の計算と表示
κ_L = lattice_thermal_conductivity(T, Θ, v_s, k_B, ħ, β, γ, M_kg, a)
print(f"格子熱伝導率 κ_L: {κ_L:.3e} W/(m·K)")