import mpmath
import numpy as np
from scipy.constants import h, k, e, pi, N_A
import matplotlib.pyplot as plt
from scipy.integrate import quad

# mpmathの精度を設定
mpmath.mp.dps = 50  # 小数点以下50桁の精度で計算

# 物理定数の定義
hbar = mpmath.mpf(h) / (2 * mpmath.pi)  # 換等プランク定数 (Js)
m_star = mpmath.mpf(1.4 * 9.11e-31)  # 電子の有効質量 (kg)
m_star_e = mpmath.mpf(1.4 * 9.11e-31)  # 電子の有効質量_e (kg)
m_star_h = mpmath.mpf(1.4 * 9.11e-31)  # 正孔の有効質量_h (kg)
kb = mpmath.mpf(k)  # ボルツマン定数 (J/K)
q = mpmath.mpf(e)  # 電子の電荷 (C)
T = mpmath.mpf(300)  # 温度 (K)
tau_e = mpmath.mpf(1e-14)  # 緩和時間_e (s)
tau_h = mpmath.mpf(1e-14)  # 緩和時間_h (s)
E_g = mpmath.mpf(0.000001) * q  # バンドギャップ [eV]を[J]に変換
N_A = N_A  # アボガドロ数 (1/mol)

# N_C, N_Vの計算（探索コードに基づく追加部分）
N_C = 2 * ((m_star_e * kb * T) / (2 * mpmath.pi * hbar**2))**(3/2)
N_V = 2 * ((m_star_h * kb * T) / (2 * mpmath.pi * hbar**2))**(3/2)
xi_g = E_g / (kb * T)  # ξ_Gの計算
N_D = mpmath.mpf(5e26)  # ドナー濃度 [1/m^3]
E_D = mpmath.mpf(0.026) * q  # ドナー準位 [eV]を[J]に変換
g_c = mpmath.mpf(2)  # 縮逆係数
g_v = mpmath.mpf(4)  # 縮逆係数

# フェルミ統計の関数定義
def fermi_dirac_half(xi):
    def integrand(x):
        return mpmath.sqrt(x) / (mpmath.exp(x - xi) + 1)
    return mpmath.quad(integrand, [0, mpmath.inf])

# 中性条件とキャリア濃度の計算
def neutrality_and_carrier_densities(xi_F):
    f_D = 1 / (1 + (1 / g_c) * mpmath.exp((E_D / (kb * T)) - xi_F))
    n_D = N_D * f_D
    n = (2 / mpmath.sqrt(mpmath.pi)) * N_C * fermi_dirac_half(xi_F)
    p = (2 / mpmath.sqrt(mpmath.pi)) * N_V * fermi_dirac_half(-xi_F - xi_g)
    neutrality = n - (N_D - n_D) - p
    return (xi_F, n, p, n_D, neutrality)

# 温度範囲の定義を冒頭で行う
T_range = [mpmath.mpf(val) for val in np.linspace(300, 1300, 10)]  # 温度範囲 (300K～1300K)

# xi_Fの初期探索
xi_F_results = []  # 各温度ごとの xi_F を格納するリスト

for T in T_range:
    xi_g = E_g / (kb * T)
    xi_F_vals = np.arange(-20, 20, 1)  # xi_F の探索範囲を定義
    previous_neutrality = None
    found = False

    print(f"Debug: Starting search for T = {float(T)}")  # 温度の開始を表示

    for xi_F in xi_F_vals:
        _, _, _, _, neutrality = neutrality_and_carrier_densities(xi_F)
        if previous_neutrality is not None:
            # 符号が変わったことを検出
            if previous_neutrality * neutrality < 0:
                print(f"Debug: Sign change detected for T = {float(T)} at xi_F = {xi_F}")  # 符号変化の検出を表示
                # 符号が変わった範囲をより精度高く探索
                lower_bound = xi_F - 1
                upper_bound = xi_F
                interval = 0.01
                while True:
                    refined_xi_F_vals = np.linspace(lower_bound, upper_bound, 100)
                    for refined_xi_F in refined_xi_F_vals:
                        _, _, _, _, refined_neutrality = neutrality_and_carrier_densities(refined_xi_F)
                        if abs(refined_neutrality) < 1e15:
                            xi_F_results.append(refined_xi_F)  # 各温度ごとの xi_F を保存
                            found = True
                            print(f"Debug: Found xi_F = {refined_xi_F} for T = {float(T)}")  # 探索成功時の表示
                            break
                    if found:
                        break
                    # 次のループでさらに精度高く探索
                    for i in range(1, len(refined_xi_F_vals)):
                        if (neutrality_and_carrier_densities(refined_xi_F_vals[i - 1])[4] *
                                neutrality_and_carrier_densities(refined_xi_F_vals[i])[4]) < 0:
                            lower_bound = refined_xi_F_vals[i - 1]
                            upper_bound = refined_xi_F_vals[i]
                            interval *= 0.1
                            break
                    else:
                        break
                if found:
                    break
        previous_neutrality = neutrality

    if not found:
        print(f"Debug Info: T = {float(T)}, Last Neutrality = {float(previous_neutrality)}")  # デバッグ出力
        raise ValueError(f"Neutrality condition not found for T = {float(T)}")


# 入力パラメータ y
y = 0.8

# パラメータの更新
β = 2.0  # 正常過程とウムクラップ過程の緩和時間の比
γ = 0.91  # グリュナイゼン定数
T = 300  # 温度 (K)
E = 2.94 * e  # 音響フォノン変形ポテンシャル定数 (J)
ε_S = 39  # 点欠陥散乱における歪みパラメータ
L = 1e-6  # 粒形 (m)

a_cubed = (((2.7155e-10) ** 3) * (1 - y)) + (((2.8288e-10) ** 3) * y)  # 平均原子体積
a = a_cubed ** (1/3)  # 三乗根
M_g = (28.086 * (1 - y) + 72.59 * y)  # 原子の平均質量 [g]
M_kg = (28.086 * (1 - y) + 72.59 * y) * 1e-3  # 原子の平均質量[kg]
G = (1.033 * (1 - y) + 1.017 * y) * 1e-3
delta_M = 72.59 - 28.086 # 質量の差 (g)
delta_a = (2.8288 - 2.7155) * 1e-10  # 原子体積の差 (m)

Θ = 1.48e-8 * (a ** (-3/2)) * (M_g ** (-1/2)) * G  # デバイ温度 (K)
v_s = (kb / hbar) * ((6 * (np.pi ** 2)) ** (-1/3)) * Θ * a  # 平均音速 (m/s)
d = M_kg / a_cubed # 物質の密度 (kg/m^3)
phi = (m_star * v_s) / (2 * kb * T) # 音速で運動するキャリアの還元フェルミエネルギー

# 1/τ_N の定義
def tau_N_inv(xi, T, Θ, β, γ, M_kg, a):
    # xi は積分中で変動する値
    term1 = ((20 * np.pi) / 3) * hbar * N_A * ((6 * np.pi ** 2) / 4) ** (1 / 3) * (β * (1 + (5 / 9) * β) / (1 + β)) * (γ ** 2) / (M_kg * a ** 2) * (T / Θ) ** 3 * xi ** 2
    return term1

# 1/τ_U の定義
def tau_U_inv(tau_N_inv_val, β):
    return (1 / β) * tau_N_inv_val

# トンネル散乱
def l_TS_inv(omega, T):
    A = 1.38e4  # [M^-1K^-1]
    B = 1.5e-3  # [K^-2]
    term1 = A * (hbar * omega / kb) * mpmath.tanh(hbar * omega / (2 * kb * T))
    term2 = (A / 2) * ((kb / (hbar * omega)) + (B**-1) * (T**-3))**-1
    return term1 + term2

# 共鳴散乱
def l_res_inv(omega, T):
    C1 = 3.8e6  # [m^-1s^-3K^-2]
    C2 = 1.7e8  # [m^-1s^-3K^-2]
    omega1 = 1.7e12  # [Hz]
    omega2 = 6.3e12  # [Hz]
    gamma1 = gamma2 = 0.8  # 無次元
    term1 = (C1 * omega**2 * T**2) / ((omega1**2 - omega**2)**2 + (gamma1 * omega1**2 * omega**2))
    term2 = (C2 * omega**2 * T**2) / ((omega2**2 - omega**2)**2 + (gamma2 * omega2**2 * omega**2))
    return term1 + term2

# レーリー散乱
def l_R_inv(omega):
    D = 1  # [m^-1K^-4]
    return D * (hbar**4 * omega**4) / (kb**4)

# 最低項 l_min の定義
l_min = 3e-10  # 最小平均自由行程 [m]

# フォノン平均自由行程 l(ω) の計算
def phonon_mean_free_path(omega, T, v_s):
    l_TS = 1 / l_TS_inv(omega, T)
    l_res = 1 / l_res_inv(omega, T)
    l_R = 1 / l_R_inv(omega)
    l_combined = (1 / (l_TS**-1 + l_res**-1 + l_R**-1)) + l_min
    return l_combined

# フォノン散乱時間 τ(ω) の計算
def phonon_scattering_time(omega, T, v_s):
    l = phonon_mean_free_path(omega, T, v_s)
    return l / v_s

# 1/τ_C の計算
def tau_C_inv(xi, xi_F, omega, T, Θ, β, γ, E, y, L, m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a):
    # τ_N と τ_U を考慮
    τ_N_inv_val = tau_N_inv(xi, T, Θ, β, γ, M_kg, a)
    τ_U_inv_val = tau_U_inv(τ_N_inv_val, β)

    # 3散乱 を考慮
    l_TS_inv_val = l_TS_inv(omega, T)  # トンネル散乱の逆平均自由行程
    l_res_inv_val = l_res_inv(omega, T)  # 共鳴散乱の逆平均自由行程
    l_R_inv_val = l_R_inv(omega)  # レーリー散乱の逆平均自由行程

    # フォノン平均自由行程 l の計算
    l_inv_total = l_TS_inv_val + l_res_inv_val + l_R_inv_val
    l = 1 / l_inv_total + l_min  # 合成平均自由行程
    τ_scattering_inv = v_s / l  # 散乱時間の逆数

    return τ_N_inv_val + τ_U_inv_val + τ_scattering_inv

# τ_C を積分中で動的に計算
def tau_C(xi, xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a):
    τ_C_val = 1 / tau_C_inv(xi, xi_F, omega, T, Θ, β, γ, E, y, L, m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a)
    return τ_C_val

# I1の積分
def I1_integrand(xi, omega, T, Θ, β, γ, E, y, L, m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a):
    τ_C_val = tau_C(xi, xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a)  # xi に依存する τ_C の計算
    return τ_C_val * (xi ** 4 * np.exp(xi)) / ((np.exp(xi) - 1) ** 2)

def I1(omega, T, Θ, β, γ, E, y, L, m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a):
    return quad(I1_integrand, 0, Θ / T, args=(omega, T, Θ, β, γ, E, y, L, m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a))[0]

# I2の積分
def I2_integrand(xi, xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a):
    τ_C_val = tau_C(xi, xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a)  # xi に依存する τ_C の計算
    τ_N_val = 1 / tau_N_inv(xi, T, Θ, β, γ, M_kg, a)  # xi に依存する τ_N の計算
    return (τ_C_val / τ_N_val) * (xi ** 4 * np.exp(xi)) / ((np.exp(xi) - 1) ** 2)

def I2(xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a):
    return quad(I2_integrand, 0, Θ / T, args=(xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a))[0]

# I3の積分
def I3_integrand(xi, xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a):
    τ_C_val = tau_C(xi, xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a)  # xi に依存する τ_C の計算
    τ_N_val = 1 / tau_N_inv(xi, T, Θ, β, γ, M_kg, a)  # xi に依存する τ_N の計算
    return (1 / τ_N_val) * (1 - (τ_C_val / τ_N_val)) * (xi ** 4 * np.exp(xi)) / ((np.exp(xi) - 1) ** 2)

def I3(xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a):
    return quad(I3_integrand, 0, Θ / T, args=(xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a))[0]

# 格子熱伝導率 κ_L の計算
def lattice_thermal_conductivity(xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a):
    I1_val = I1(omega, T, Θ, β, γ, E, y, L, m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a)
    I2_val = I2(xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a)
    I3_val = I3(xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a)

    κ_L = (kb / (2 * np.pi ** 2 * v_s)) * ((kb * T) / hbar) ** 3 * (I1_val + (I2_val ** 2) / I3_val)
    return κ_L

# グラフ作成
def confirm_conductivity_coefficients(s, xi_g, T_range, omega):
    results = {"T": [], "κ_L": []}
    for T, xi_F in zip(T_range, xi_F_results):  # 温度と対応する xi_F を利用
        E_F = xi_F * (kb * T)  # E_F を計算
        xi_g = E_g / (kb * T)

        κ_L = lattice_thermal_conductivity(xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a)

        results["T"].append(float(T))  # mpf を float に変換
        results["κ_L"].append(float(κ_L))   # mpf を float に変換


# 周波数を設定
omega = 1e13  # サンプル周波数 [Hz]

# T の範囲を指定してゼーベック係数を確認
T_range = [mpmath.mpf(val) for val in np.linspace(1000, 2500, 15)]
s = -1/2  # 状態密度の有効質量の次元数
confirm_conductivity_coefficients(s, xi_g, T_range, omega)

# ドナー濃度 N_D のリスト
N_D_values = [mpmath.mpf(2.15e25), mpmath.mpf(4.64e25), mpmath.mpf(1e26),
              mpmath.mpf(2.15e26), mpmath.mpf(4.64e26)]

# 線のスタイルのリスト
line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]  # 実線、破線、点破線、点線など

# グラフデータの計算とプロット
plt.figure(figsize=(10, 6))

for i, N_D in enumerate(N_D_values):
    # N_D を更新
    current_N_D = N_D

    # xi_F を N_D ごとに計算
    xi_F_vals = np.arange(-20, 20, 1)
    previous_neutrality = None
    found = False

    for xi_F in xi_F_vals:
        _, _, _, _, neutrality = neutrality_and_carrier_densities(xi_F)
        if previous_neutrality is not None:
            # 符号が変わったことを検出
            if previous_neutrality * neutrality < 0:
                # 符号が変わった範囲をより精度高く探索
                lower_bound = xi_F - 1
                upper_bound = xi_F
                while True:
                    refined_xi_F_vals = np.linspace(lower_bound, upper_bound, 100)
                    for refined_xi_F in refined_xi_F_vals:
                        _, _, _, _, refined_neutrality = neutrality_and_carrier_densities(refined_xi_F)
                        if abs(refined_neutrality) < 1e15:
                            xi_F = refined_xi_F  # 探索された ξ_F の保存
                            found = True
                            break
                    if found:
                        break
                    # 次のループでさらに精度高く探索
                    for i in range(1, len(refined_xi_F_vals)):
                        if (neutrality_and_carrier_densities(refined_xi_F_vals[i - 1])[4] *
                                neutrality_and_carrier_densities(refined_xi_F_vals[i])[4]) < 0:
                            lower_bound = refined_xi_F_vals[i - 1]
                            upper_bound = refined_xi_F_vals[i]
                            break
                    else:
                        break
                if found:
                    break
        previous_neutrality = neutrality

    if not found:
        raise ValueError(f"Neutrality condition not found for N_D = {current_N_D}")

    # ξ_F の値として探索された値を使用
    xi_F = refined_xi_F
    E_F = xi_F * (kb * T)
    xi_g = E_g / (kb * T)

    # 結果を格納するリスト
    results = {"T": [], "κ_L": []}

    for T in T_range:
        # xi_F と xi_g の更新
        xi_F = E_F / (kb * T)
        xi_g = E_g / (kb * T)

        # κ_L の計算
        κ_L = lattice_thermal_conductivity(xi_F, omega, T, Θ, β, γ, E, y, L,  m_star, v_s, d, phi, delta_M, M_kg, ε_S, delta_a, a)

        # 結果を格納
        results["T"].append(float(T))  # mpf を float に変換
        results["κ_L"].append(float(κ_L))  # mpf を float に変換

    # プロット
    label_N_D = f"N_D = {mpmath.nstr(N_D, 3)}"  # mpmath.nstr を使用
    plt.plot(
        results["T"],
        [a for a in results["κ_L"]],
        label=label_N_D,
        linestyle=line_styles[i % len(line_styles)]  # スタイルを循環させる
    )

# グラフ設定
plt.axhline(0, color="gray", linestyle=":")
plt.xlabel("T [K]")
plt.ylabel("κ_L")
plt.legend()
plt.grid(True)
plt.show()