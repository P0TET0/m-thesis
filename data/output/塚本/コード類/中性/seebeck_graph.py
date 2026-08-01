import mpmath
import numpy as np
from scipy.constants import h, k, e, pi, N_A
import matplotlib.pyplot as plt
from scipy.integrate import quad

# mpmathの精度を設定
mpmath.mp.dps = 50  # 小数点以下50桁の精度で計算

# 物理定数の定義
hbar = mpmath.mpf(h) / (2 * mpmath.pi)  # 換等プランク定数 (Js)
m_star_e = mpmath.mpf(1.4 * 9.11e-31)  # 電子の有効質量_e (kg)
m_star_h = mpmath.mpf(1.4 * 9.11e-31)  # 正孔の有効質量_h (kg)
kb = mpmath.mpf(k)  # ボルツマン定数 (J/K)
q = mpmath.mpf(e)  # 電子の電荷 (C)
T = mpmath.mpf(300)  # 温度 (K)
tau_e = mpmath.mpf(1e-14)  # 緩和時間_e (s)
tau_h = mpmath.mpf(1e-14)  # 緩和時間_h (s)
E_g = mpmath.mpf(0.7552) * q  # バンドギャップ [eV]を[J]に変換
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

# xi_Fの初期探索
xi_F_vals = np.arange(-20, 20, 1)
previous_neutrality = None
found = False

for xi_F in xi_F_vals:
    xi_F_val, n, p, n_D, neutrality = neutrality_and_carrier_densities(xi_F)
    if previous_neutrality is not None:
        # 符号が変わったことを検出
        if previous_neutrality * neutrality < 0:
            # 符号が変わった範囲をより精度高く探索
            lower_bound = xi_F - 1
            upper_bound = xi_F
            interval = 0.1
            while True:
                refined_xi_F_vals = np.linspace(lower_bound, upper_bound, 100)
                for refined_xi_F in refined_xi_F_vals:
                    xi_F_val, n, p, n_D, refined_neutrality = neutrality_and_carrier_densities(refined_xi_F)
                    if abs(refined_neutrality) < 1e15:
                        refined_xi_F_found = refined_xi_F  # 探索されたξ_Fの保存
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
                        interval *= 0.1
                        break
                else:
                    break
            if found:
                break
    previous_neutrality = neutrality

if not found:
    raise ValueError("Neutrality condition not found in the given range.")

# ξ_F の値として探索された値を使用
xi_F = refined_xi_F_found

# E_Fの定義
E_F = xi_F * (kb * T)

# フェルミ統計_eの関数定義
def fermi_integral_e(s, xi_F):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - xi_F
        return xi**s / (mpmath.exp(exponent) + 1)
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# フェルミ統計_hの関数定義
def fermi_integral_h(s, xi_F, xi_g):
    def integrand(xi):
        xi = mpmath.mpf(xi)
        exponent = xi - (-xi_F - xi_g)
        return xi**s / (mpmath.exp(exponent) + 1)
    result = mpmath.quad(integrand, [0, mpmath.inf])
    return result

# ゼーベック係数 α_e の計算関数
def calculate_alpha_e(s, xi_F):
    F_s_plus_32 = fermi_integral_e(s + 3/2, xi_F)
    F_s_plus_12 = fermi_integral_e(s + 1/2, xi_F)
    small_delta_e = ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_plus_12)
    alpha_e = -((kb / q) * (small_delta_e - xi_F))
    return alpha_e

# ゼーベック係数 α_h の計算関数
def calculate_alpha_h(s, xi_F, xi_g):
    F_s_plus_32 = fermi_integral_h(s + 3/2, xi_F, xi_g)
    F_s_plus_12 = fermi_integral_h(s + 1/2, xi_F, xi_g)
    small_delta_h = ((s + 5/2) * F_s_plus_32) / ((s + 3/2) * F_s_plus_12)
    alpha_h = ((kb / q) * (small_delta_h - (-xi_F - xi_g)))
    return alpha_h

# ゼーベック係数 α の計算関数
def calculate_alpha(s, xi_F, xi_g):
    sigma_e = calculate_conductivity_e(s, xi_F)
    sigma_h = calculate_conductivity_h(s, xi_F, xi_g)
    sigma = calculate_conductivity(s, xi_F, xi_g)

    alpha_e = calculate_alpha_e(s, xi_F)
    alpha_h = calculate_alpha_h(s, xi_F, xi_g)

    alpha = ((alpha_e * sigma_e) + (alpha_h * sigma_h)) / sigma
    return alpha

# 電気伝属率 σ_e の計算関数
def calculate_conductivity_e(s, xi_F):
    F_s_plus_half = fermi_integral_e(s + 0.5, xi_F)
    sigma_e = (4 * q**2 * N_C * tau_e) / (3 * mpmath.sqrt(pi) * m_star_e) * (s + 1.5) * F_s_plus_half
    return sigma_e

# 電気伝属率 σ_h の計算関数
def calculate_conductivity_h(s, xi_F, xi_g):
    F_s_plus_half = fermi_integral_h(s + 0.5, xi_F, xi_g)
    sigma_h = (4 * q**2 * N_V * tau_h) / (3 * mpmath.sqrt(pi) * m_star_h) * (s + 1.5) * F_s_plus_half
    return sigma_h

# 電気伝属率 σ の計算関数
def calculate_conductivity(s, xi_F, xi_g):
    sigma_e = calculate_conductivity_e(s, xi_F)
    sigma_h = calculate_conductivity_h(s, xi_F, xi_g)
    sigma = sigma_e + sigma_h
    return sigma

# グラフ作成
def confirm_conductivity_coefficients(s, xi_g, T_range):
    results = {"T": [], "alpha_e": [], "alpha_h": [], "alpha": []}
    for T in T_range:
        xi_F = E_F / (kb * T)
        xi_g = E_g / (kb * T)

        alpha_e = calculate_alpha_e(s, xi_F)
        alpha_h = calculate_alpha_h(s, xi_F, xi_g)
        alpha = calculate_alpha(s, xi_F, xi_g)

        results["T"].append(float(T))  # mpf を float に変換
        results["alpha_e"].append(float(alpha_e))   # mpf を float に変換
        results["alpha_h"].append(float(alpha_h))   # mpf を float に変換
        results["alpha"].append(float(alpha))   # mpf を float に変換


# T の範囲を指定してゼーベック係数を確認
T_range = [mpmath.mpf(val) for val in np.linspace(300, 1300, 10)]
s = 3/2  # 状態密度の有効質量の次元数
confirm_conductivity_coefficients(s, xi_g, T_range)

# T の範囲を指定
T_range = [mpmath.mpf(val) for val in np.linspace(300, 1300, 10)]
s = 3/2  # 状態密度の有効質量の次元数

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
    results = {"T": [], "alpha": []}

    for T in T_range:
        # xi_F と xi_g の更新
        xi_F = E_F / (kb * T)
        xi_g = E_g / (kb * T)
        
        # α の計算
        alpha = calculate_alpha(s, xi_F, xi_g)
        
        # 結果を格納
        results["T"].append(float(T))  # mpf を float に変換
        results["alpha"].append(float(alpha))  # mpf を float に変換
    
    # プロット
    label_N_D = f"N_D = {mpmath.nstr(N_D, 3)}"  # mpmath.nstr を使用
    plt.plot(
        results["T"],
        [a * 1e6 for a in results["alpha"]],
        label=label_N_D,
        linestyle=line_styles[i % len(line_styles)]  # スタイルを循環させる
    )

# グラフ設定
plt.axhline(0, color="gray", linestyle=":")
plt.xlabel("T [K]")
plt.ylabel("α [μV/K]")
plt.legend()
plt.grid(True)
plt.show()