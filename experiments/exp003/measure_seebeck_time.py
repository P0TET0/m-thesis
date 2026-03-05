# measure_seebeck_time.py
from __future__ import annotations

"""
事前計算した `T_range.pkl` / `N_D_values.pkl` / `xi_F_vals.pkl` を読み込み、
指定温度・指定ドーピング点で Seebeck 係数 `alpha2` を計算して実行時間を測るスクリプト。
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
    # pklファイル読み込みの共通処理
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    # 実行オプション
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

    # 入力データは計測前に読み込む（計算時間とI/O時間を分離）
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

    # 配列サイズの整合性チェック
    if not (0 <= args.nd_index < len(N_D_values)):
        raise IndexError(f"--nd-index が範囲外です: {args.nd_index} (0..{len(N_D_values)-1})")
    if len(xi_F_vals) != len(N_D_values):
        raise ValueError("xi_F_vals の外側次元が N_D_values と一致していません。")
    if len(xi_F_vals[args.nd_index]) != len(T_range):
        raise ValueError("xi_F_vals[nd_index] の長さが T_range と一致していません。")

    # 温度インデックスの決定（最近傍 or 完全一致）
    T_array = np.array(T_range, dtype=float)
    if args.mode == "nearest":
        j = int(np.argmin(np.abs(T_array - float(args.t))))
    else:
        # exact
        hits = np.where(T_array == float(args.t))[0]
        if len(hits) == 0:
            raise ValueError(f"--mode exact ですが T_range に {args.t} が存在しません。")
        j = int(hits[0])

    T_used_float = float(T_array[j])
    xi_F_used_float = float(xi_F_vals[args.nd_index][j])

    # ここから物理モデル定義
    mp.mp.dps = int(args.dps)

    # 定数
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

    def xi_g(T):  # dimensionless
        return E_g / (kb * T)

    def a_cubed(y):
        return mp.power(mp.mpf("2.7155e-10"), 3) * (1 - y) + mp.power(mp.mpf("2.8288e-10"), 3) * y

    def a(y):
        return mp.power(a_cubed(y), mp.mpf(1) / 3)

    def M_g(y):
        return mp.mpf("28.086") * (1 - y) + mp.mpf("72.59") * y

    def M_kg(y):
        return M_g(y) * mp.mpf("1e-3")

    def G(y):
        return (mp.mpf("1.033") * (1 - y) + mp.mpf("1.017") * y) * mp.mpf("1e-3")

    def Theta(y):
        return mp.mpf("1.48e-8") * a(y) ** (-mp.mpf("3") / 2) * (M_g(y) ** (-mp.mpf("1") / 2)) * G(y)

    def v_s(y):
        return (kb / hbar) * (6 * mp.pi**2) ** (-mp.mpf("1") / 3) * Theta(y) * a(y)

    def rho_d(y):
        return M_kg(y) / (a_cubed(y) * N_A)

    # 有効質量
    m0 = mp.mpf("9.11e-31")
    m_star_e = mp.mpf("1.4") * m0
    m_star_h = mp.mpf("1.4") * m0

    # フェルミ・ディラック積分（計算コストが高い部分）
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

    def sigma2(power_s, xi_F, T):
        return sigma2e(power_s, xi_F, T) + sigma2h(power_s, xi_F, T)

    def alpha2(power_s, xi_F, T):
        sig_e = sigma2e(power_s, xi_F, T)
        sig_h = sigma2h(power_s, xi_F, T)
        denom = sig_e + sig_h
        return (alpha2e(power_s, xi_F) * sig_e + alpha2h(power_s, xi_F, T) * sig_h) / denom

    # 指定した1点の alpha2 を繰り返し計算して時間計測
    T_used = mp.mpf(T_used_float)
    xi_F_used = mp.mpf(xi_F_used_float)

    def calc_one():
        return alpha2(s, xi_F_used, T_used)

    # ウォームアップ
    for _ in range(int(args.warmup)):
        _ = calc_one()

    times = []
    last_val = None
    for _ in range(int(args.repeats)):
        t0 = time.perf_counter()
        last_val = calc_one()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # 出力
    print("---- input ----")
    print(f"T_input={args.t} K  mode={args.mode}")
    print(f"selected: T_used={T_used_float} K  (index j={j})")
    print(f"selected: xi_F_used={xi_F_used_float}  (nd_index i={args.nd_index})")
    print("---- result ----")
    print("Seebeck(alpha2) =", last_val)
    print("---- timing ----")
    print(f"repeats={args.repeats}, warmup={args.warmup}, mp.dps={args.dps}")
    print(f"mean   = {statistics.mean(times):.6f} s")
    print(f"median = {statistics.median(times):.6f} s")
    print(f"min    = {min(times):.6f} s")
    print(f"max    = {max(times):.6f} s")


if __name__ == "__main__":
    main()
