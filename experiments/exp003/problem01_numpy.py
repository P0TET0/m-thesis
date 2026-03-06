# measure_seebeck_time_numpy_full.py
from __future__ import annotations

"""
T_range.pkl / N_D_values.pkl を読み込み、
指定温度・指定ドーピング点で

  (T -> xi_F を NumPy+Newton で解く) -> alpha2 を計算

までを1回として、repeats回計測するスクリプト。
"""

import argparse
import pickle
import statistics
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.constants import h, k, e, pi, N_A as N_A_SI
from scipy.integrate import quad
from scipy.optimize import newton


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", type=str, required=True, help="pklファイルがあるフォルダ")
    parser.add_argument("--t", type=float, required=True, help="入力温度 [K]")
    parser.add_argument("--nd-index", type=int, default=0, help="N_D_values の何番目を使うか")
    parser.add_argument(
        "--mode",
        type=str,
        default="nearest",
        choices=["nearest", "exact"],
        help="T_range からの取り方: nearest=最も近い温度, exact=完全一致のみ",
    )
    parser.add_argument("--repeats", type=int, default=10, help="計測回数")
    parser.add_argument("--warmup", type=int, default=1, help="ウォームアップ回数（計測に含めない）")
    parser.add_argument("--dps", type=int, default=100, help="mpmath精度(mp.dps)")
    parser.add_argument("--y", type=float, default=0.8, help="組成比 y")
    parser.add_argument("--eg-ev", type=float, default=0.910022, help="バンドギャップ [eV]")
    parser.add_argument("--s", type=float, default=1.5, help="散乱指数 s（例: 3/2=1.5）")
    args = parser.parse_args()

    # 入力データは計測前に読み込む（I/O時間を分離）
    pkl_dir = Path(args.pkl_dir)
    t_range_path = pkl_dir / "T_range.pkl"
    nd_values_path = pkl_dir / "N_D_values.pkl"

    if not t_range_path.exists():
        raise FileNotFoundError(f"見つかりません: {t_range_path}")
    if not nd_values_path.exists():
        raise FileNotFoundError(f"見つかりません: {nd_values_path}")

    T_range = load_pickle(t_range_path)
    N_D_values = load_pickle(nd_values_path)

    if not (0 <= args.nd_index < len(N_D_values)):
        raise IndexError(f"--nd-index が範囲外です: {args.nd_index} (0..{len(N_D_values)-1})")

    # 温度インデックスの決定（最近傍 or 完全一致）
    T_array = np.array(T_range, dtype=float)
    if args.mode == "nearest":
        j = int(np.argmin(np.abs(T_array - float(args.t))))
    else:
        hits = np.where(T_array == float(args.t))[0]
        if len(hits) == 0:
            raise ValueError(f"--mode exact ですが T_range に {args.t} が存在しません。")
        j = int(hits[0])

    T_used_float = float(T_array[j])
    N_D_used_float = float(N_D_values[args.nd_index])

    # ここから物理モデル定義
    mp.mp.dps = int(args.dps)

    # 定数（mpmath側）
    hbar = mp.mpf(h) / (2 * mp.pi)
    kb = mp.mpf(k)
    q = mp.mpf(e)
    N_A = mp.mpf(N_A_SI)

    # パラメータ
    y = mp.mpf(args.y)
    E_g = mp.mpf(args.eg_ev) * q  # [eV] -> [J]
    s = mp.mpf(args.s)

    # 材料パラメータ
    beta = mp.mpf("2.0")
    gamma = mp.mpf("0.91")
    E_def = mp.mpf("2.94") * q

    def xi_g(T):
        return E_g / (kb * T)

    def a_cubed(y_):
        return mp.power(mp.mpf("2.7155e-10"), 3) * (1 - y_) + mp.power(mp.mpf("2.8288e-10"), 3) * y_

    def a(y_):
        return mp.power(a_cubed(y_), mp.mpf(1) / 3)

    def M_g(y_):
        return mp.mpf("28.086") * (1 - y_) + mp.mpf("72.59") * y_

    def M_kg(y_):
        return M_g(y_) * mp.mpf("1e-3")

    def G(y_):
        return (mp.mpf("1.033") * (1 - y_) + mp.mpf("1.017") * y_) * mp.mpf("1e-3")

    def Theta(y_):
        return mp.mpf("1.48e-8") * a(y_) ** (-mp.mpf("3") / 2) * (M_g(y_) ** (-mp.mpf("1") / 2)) * G(y_)

    def v_s(y_):
        return (kb / hbar) * (6 * mp.pi**2) ** (-mp.mpf("1") / 3) * Theta(y_) * a(y_)

    def rho_d(y_):
        return M_kg(y_) / (a_cubed(y_) * N_A)

    # 有効質量
    m0 = mp.mpf("9.11e-31")
    m_star_e = mp.mpf("1.4") * m0
    m_star_h = mp.mpf("1.4") * m0

    # フェルミ・ディラック積分（alpha2側：重い）
    def fermi_dirac(power_s, xi_F):
        def integrand(x):
            return x**power_s / (mp.e ** (x - xi_F) + 1)

        return mp.quad(integrand, [0, mp.inf])

    def delta(power_s, xi_F):
        return (power_s + mp.mpf("2.5")) * fermi_dirac(power_s + mp.mpf("1.5"), xi_F) / (
            (power_s + mp.mpf("1.5")) * fermi_dirac(power_s + mp.mpf("0.5"), xi_F)
        )

    def alpha1(power_s, xi_F):
        return (kb / q) * (delta(power_s, xi_F) - xi_F)

    # 変形ポテンシャル散乱の緩和時間
    def tau_dp_e(T):
        term1 = mp.sqrt(2) * E_def**2 * (m_star_e * kb * T) ** (mp.mpf("1.5"))
        term2 = mp.pi * rho_d(y) * hbar**4 * v_s(y) ** 2
        return term2 / term1

    def tau_dp_h(T):
        term1 = mp.sqrt(2) * E_def**2 * (m_star_h * kb * T) ** (mp.mpf("1.5"))
        term2 = mp.pi * rho_d(y) * hbar**4 * v_s(y) ** 2
        return term2 / term1

    def N_B(m_star, T):
        return 2 * (m_star * kb * T / (2 * mp.pi * hbar**2)) ** (mp.mpf("1.5"))

    def N_C(T):
        return N_B(m_star_e, T)

    def N_V(T):
        return N_B(m_star_h, T)

    def sigma1(power_s, xi_F, N_band, m_star, tau_val):
        return ((4 * q**2 * N_band * tau_val) / (3 * mp.sqrt(mp.pi) * m_star)) * (power_s + mp.mpf("1.5")) * fermi_dirac(
            power_s + mp.mpf("0.5"), xi_F
        )

    def alpha2e(power_s, xi_F):
        return -alpha1(power_s, xi_F)

    def alpha2h(power_s, xi_F, T):
        return alpha1(power_s, -xi_F - xi_g(T))

    def sigma2e(power_s, xi_F, T):
        return sigma1(power_s, xi_F, N_C(T), m_star_e, tau_dp_e(T))

    def sigma2h(power_s, xi_F, T):
        return sigma1(power_s, -xi_F - xi_g(T), N_V(T), m_star_h, tau_dp_h(T))

    def alpha2(power_s, xi_F, T):
        sig_e = sigma2e(power_s, xi_F, T)
        sig_h = sigma2h(power_s, xi_F, T)
        denom = sig_e + sig_h
        return (alpha2e(power_s, xi_F) * sig_e + alpha2h(power_s, xi_F, T) * sig_h) / denom

    # -------------------------
    # ここが「xi_Fの差し替えポイント」（NumPy + Newton）
    # -------------------------
    hbar_np = h / (2.0 * pi)
    m_star_e_np = 1.4 * 9.11e-31
    m_star_h_np = 1.4 * 9.11e-31
    kb_np = k
    q_np = e
    E_g_np = float(args.eg_ev) * q_np
    E_D_np = 0.026 * q_np
    g_c_np = 2.0

    def N_C_np(T):
        return 2.0 * ((m_star_e_np * kb_np * T) / (2.0 * pi * hbar_np**2)) ** 1.5

    def N_V_np(T):
        return 2.0 * ((m_star_h_np * kb_np * T) / (2.0 * pi * hbar_np**2)) ** 1.5

    def xi_g_np(T):
        return E_g_np / (kb_np * T)

    def xi_D_np(T):
        return E_D_np / (kb_np * T)

    def fermi_dirac_half_np(xi):
        xi = float(xi)

        def integrand(x):
            diff = x - xi
            if diff > 40:
                return 0.0
            if diff < -40:
                return np.sqrt(x)
            ex = np.exp(diff)
            return np.sqrt(x) / (1.0 + ex)

        result, _ = quad(integrand, 0.0, 100.0, epsabs=1e-10, epsrel=1e-10, limit=100)
        return float(result)

    def f_D_np(xi_F, T):
        x = xi_D_np(T) - float(xi_F)
        if x >= 40:
            return 0.0
        return 1.0 / (1.0 + (1.0 / g_c_np) * np.exp(x))

    def n_np(xi_F, T):
        return (2.0 / np.sqrt(np.pi)) * N_C_np(T) * fermi_dirac_half_np(xi_F)

    def p_np(xi_F, T):
        return (2.0 / np.sqrt(np.pi)) * N_V_np(T) * fermi_dirac_half_np(-float(xi_F) - xi_g_np(T))

    def neutrality_np(N_D, T):
        return lambda xi_F: n_np(xi_F, T) - N_D * (1.0 - f_D_np(xi_F, T)) - p_np(xi_F, T)

    def solve_xiF_neutrality_np(N_D, T, x0):
        func = neutrality_np(N_D, T)

        def fprime(x):
            return (func(x + 1e-6) - func(x - 1e-6)) / (2e-6)

        return float(
            newton(
                func,
                x0=float(x0),
                fprime=fprime,
                tol=1e-10,
                maxiter=100,
            )
        )

    # -------------------------
    # 計測対象（T -> xi_F -> alpha2）
    # -------------------------
    T_used = mp.mpf(T_used_float)

    # 初期値（繰り返し同一条件を解くので、前回解を次回初期値にすると安定しやすい）
    xi_guess = 0.0

    def calc_one():
        nonlocal xi_guess
        xi_F_used_float = solve_xiF_neutrality_np(N_D_used_float, T_used_float, x0=xi_guess)
        xi_guess = xi_F_used_float
        xi_F_used = mp.mpf(xi_F_used_float)
        return xi_F_used_float, alpha2(s, xi_F_used, T_used)

    # ウォームアップ（xi_F解法込み）
    last_xi = None
    last_val = None
    for _ in range(int(args.warmup)):
        last_xi, last_val = calc_one()

    # 計測（xi_F解法込み）
    times = []
    for _ in range(int(args.repeats)):
        t0 = time.perf_counter()
        last_xi, last_val = calc_one()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # 出力
    print("---- input ----")
    print(f"T_input={args.t} K  mode={args.mode}")
    print(f"selected: T_used={T_used_float} K  (index j={j})")
    print(f"selected: N_D_used={N_D_used_float}  (nd_index i={args.nd_index})")
    print(f"selected: xi_F_used(last)={last_xi}")
    print("---- result ----")
    print("Seebeck(alpha2) =", last_val)
    print("---- timing (T -> xi_F -> alpha2) ----")
    print(f"repeats={args.repeats}, warmup={args.warmup}, mp.dps={args.dps}")
    print(f"mean   = {statistics.mean(times):.6f} s")
    print(f"median = {statistics.median(times):.6f} s")
    print(f"min    = {min(times):.6f} s")
    print(f"max    = {max(times):.6f} s")


if __name__ == "__main__":
    main()