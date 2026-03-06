# measure_seebeck_time_pkl_full.py
from __future__ import annotations

"""
T_range.pkl / N_D_values.pkl / xi_F_vals.pkl を読み込み、
指定温度・指定ドーピング点で

  (T_input -> インデックス決定 -> xi_F(pkl)取得) -> alpha2

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

    # --- I/O（計測に含めない） ---
    pkl_dir = Path(args.pkl_dir)
    t_range_path = pkl_dir / "T_range.pkl"
    nd_values_path = pkl_dir / "N_D_values.pkl"
    xi_f_vals_path = pkl_dir / "xi_F_vals.pkl"

    if not t_range_path.exists():
        raise FileNotFoundError(f"見つかりません: {t_range_path}")
    if not nd_values_path.exists():
        raise FileNotFoundError(f"見つかりません: {nd_values_path}")
    if not xi_f_vals_path.exists():
        raise FileNotFoundError(f"見つかりません: {xi_f_vals_path}")

    T_range = load_pickle(t_range_path)
    N_D_values = load_pickle(nd_values_path)
    xi_F_vals = load_pickle(xi_f_vals_path)

    # 整合性チェック
    if not (0 <= args.nd_index < len(N_D_values)):
        raise IndexError(f"--nd-index が範囲外です: {args.nd_index} (0..{len(N_D_values)-1})")
    if len(xi_F_vals) != len(N_D_values):
        raise ValueError("xi_F_vals の外側次元が N_D_values と一致していません。")
    if len(xi_F_vals[args.nd_index]) != len(T_range):
        raise ValueError("xi_F_vals[nd_index] の長さが T_range と一致していません。")

    # 温度配列（インデックス決定に使う）
    T_array = np.array(T_range, dtype=float)

    # --- ここから物理モデル定義 ---
    mp.mp.dps = int(args.dps)

    hbar = mp.mpf(h) / (2 * mp.pi)
    kb = mp.mpf(k)
    q = mp.mpf(e)
    N_A = mp.mpf(N_A_SI)

    y = mp.mpf(args.y)
    E_g = mp.mpf(args.eg_ev) * q
    s = mp.mpf(args.s)

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

    m0 = mp.mpf("9.11e-31")
    m_star_e = mp.mpf("1.4") * m0
    m_star_h = mp.mpf("1.4") * m0

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
    # 計測対象（T_input -> index決定 -> xi_F(pkl) -> alpha2）
    # -------------------------
    def calc_one():
        # 温度インデックスの決定（ここも計測に含める）
        if args.mode == "nearest":
            j_local = int(np.argmin(np.abs(T_array - float(args.t))))
        else:
            hits = np.where(T_array == float(args.t))[0]
            if len(hits) == 0:
                raise ValueError(f"--mode exact ですが T_range に {args.t} が存在しません。")
            j_local = int(hits[0])

        T_used_float = float(T_array[j_local])
        xi_F_used_float = float(xi_F_vals[args.nd_index][j_local])

        T_used = mp.mpf(T_used_float)
        xi_F_used = mp.mpf(xi_F_used_float)

        val = alpha2(s, xi_F_used, T_used)
        return j_local, T_used_float, xi_F_used_float, val

    # ウォームアップ
    last = None
    for _ in range(int(args.warmup)):
        last = calc_one()

    # 計測
    times = []
    for _ in range(int(args.repeats)):
        t0 = time.perf_counter()
        last = calc_one()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    j_last, T_used_last, xi_F_last, last_val = last

    # 出力
    print("---- input ----")
    print(f"T_input={args.t} K  mode={args.mode}")
    print(f"selected: T_used={T_used_last} K  (index j={j_last})")
    print(f"selected: xi_F_used={xi_F_last}  (nd_index i={args.nd_index})")
    print("---- result ----")
    print("Seebeck(alpha2) =", last_val)
    print("---- timing (T -> index -> xi_F(pkl) -> alpha2) ----")
    print(f"repeats={args.repeats}, warmup={args.warmup}, mp.dps={args.dps}")
    print(f"mean   = {statistics.mean(times):.6f} s")
    print(f"median = {statistics.median(times):.6f} s")
    print(f"min    = {min(times):.6f} s")
    print(f"max    = {max(times):.6f} s")


if __name__ == "__main__":
    main()