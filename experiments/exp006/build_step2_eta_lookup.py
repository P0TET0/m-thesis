import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed"
DEFAULT_REPORT = EXP_DIR / "reports" / "step2a_eta_lookup_report.md"

K_B_OVER_E_V_PER_K = 8.617333262145e-5
K_B_OVER_E_UV_PER_K = 86.17333262145
LOOKUP_CSV = "step2_eta_lookup_table.csv"
LOOKUP_PARQUET = "step2_eta_lookup_table.parquet"

REFERENCE_VALUES_UV = {
    0.0: 204.5,
    1.0: 150.9,
    2.0: 112.4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Step2A eta lookup table from Fermi-Dirac integrals."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output directory")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="report markdown path")
    parser.add_argument("--eta-min", type=float, default=-50.0)
    parser.add_argument("--eta-max", type=float, default=500.0)
    parser.add_argument("--d-eta", type=float, default=0.005)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step2a] {message}", flush=True)


def build_eta_grid(eta_min: float, eta_max: float, d_eta: float) -> np.ndarray:
    if not np.isfinite(eta_min) or not np.isfinite(eta_max) or not np.isfinite(d_eta):
        raise ValueError("eta-min, eta-max, and d-eta must be finite")
    if eta_max <= eta_min:
        raise ValueError("eta-max must be larger than eta-min")
    if d_eta <= 0:
        raise ValueError("d-eta must be positive")
    n_steps = int(np.floor((eta_max - eta_min) / d_eta + 0.5))
    eta = eta_min + np.arange(n_steps + 1, dtype=float) * d_eta
    if eta[-1] < eta_max - d_eta * 1e-6:
        eta = np.append(eta, eta_max)
    else:
        eta[-1] = eta_max
    return eta


def compute_f1_from_f0(eta: np.ndarray, f0: np.ndarray) -> np.ndarray:
    d_eta = np.diff(eta)
    trapezoids = 0.5 * (f0[:-1] + f0[1:]) * d_eta
    f1 = np.empty_like(f0)
    # For eta_min=-50 this is essentially zero, but it preserves the
    # non-degenerate asymptotic ratio F1/F0 and avoids an artificial edge bump.
    f1[0] = np.exp(eta[0]) if eta[0] < -20.0 else 0.0
    f1[1:] = f1[0] + np.cumsum(trapezoids)
    return f1


def build_lookup(eta_min: float, eta_max: float, d_eta: float) -> pd.DataFrame:
    log("building eta grid...")
    eta = build_eta_grid(eta_min, eta_max, d_eta)
    log("computing F0...")
    f0 = np.logaddexp(0.0, eta)
    log("computing F1 by cumulative trapezoid integration...")
    f1 = compute_f1_from_f0(eta, f0)
    log("computing S(eta) table...")
    s_model = 2.0 * f1 / f0 - eta
    return pd.DataFrame(
        {
            "eta": eta,
            "F0_eta": f0,
            "F1_eta": f1,
            "s_model": s_model,
            "S_abs_V_per_K": s_model * K_B_OVER_E_V_PER_K,
            "S_abs_uV_per_K": s_model * K_B_OVER_E_UV_PER_K,
        }
    )


def interpolate_s_uV(df: pd.DataFrame, eta_value: float) -> float:
    return float(np.interp(eta_value, df["eta"].to_numpy(), df["S_abs_uV_per_K"].to_numpy()))


def run_sanity_checks(df: pd.DataFrame, eta_min: float, eta_max: float) -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    eta = df["eta"].to_numpy(dtype=float)
    f0 = df["F0_eta"].to_numpy(dtype=float)
    f1 = df["F1_eta"].to_numpy(dtype=float)
    s_model = df["s_model"].to_numpy(dtype=float)
    s_uV = df["S_abs_uV_per_K"].to_numpy(dtype=float)

    checks["eta_monotonic_increasing"] = bool(np.all(np.diff(eta) > 0))
    checks["F0_eta_positive_finite"] = bool(np.isfinite(f0).all() and (f0 > 0).all())
    checks["F1_eta_nonnegative_finite"] = bool(np.isfinite(f1).all() and (f1 >= 0).all())
    checks["s_model_positive_finite"] = bool(np.isfinite(s_model).all() and (s_model > 0).all())
    checks["S_abs_uV_per_K_positive_finite"] = bool(np.isfinite(s_uV).all() and (s_uV > 0).all())
    checks["S_abs_uV_per_K_monotonic_decreasing"] = bool(np.all(np.diff(s_uV) <= 1e-9))
    for eta_ref, expected in REFERENCE_VALUES_UV.items():
        if eta_min <= eta_ref <= eta_max:
            actual = interpolate_s_uV(df, eta_ref)
            checks[f"S_abs_uV_at_eta_{eta_ref:g}_within_1_uV"] = bool(abs(actual - expected) <= 1.0)
        else:
            checks[f"S_abs_uV_at_eta_{eta_ref:g}_within_1_uV"] = False
    failures = [name for name, ok in checks.items() if not ok]
    return checks, failures


def save_parquet(df: pd.DataFrame, path: Path) -> tuple[bool, str]:
    try:
        df.to_parquet(path, index=False)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def write_report(
    report_path: Path,
    df: pd.DataFrame,
    eta_min: float,
    eta_max: float,
    d_eta: float,
    checks: dict[str, bool],
    parquet_status: str,
    elapsed_sec: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    eta_ge_1 = df[df["eta"] >= 1.0]
    lines = [
        "# Step2A Eta Lookup Report",
        "",
        "## Summary",
        "",
        f"- eta_min: {eta_min:g}",
        f"- eta_max: {eta_max:g}",
        f"- d_eta: {d_eta:g}",
        f"- grid points: {len(df)}",
        f"- S_abs_uV_per_K max: {df['S_abs_uV_per_K'].max():.6g}",
        f"- S_abs_uV_per_K min: {df['S_abs_uV_per_K'].min():.6g}",
        f"- S_abs_uV_per_K at eta = 0: {interpolate_s_uV(df, 0.0):.6g}",
        f"- S_abs_uV_per_K at eta = 1: {interpolate_s_uV(df, 1.0):.6g}",
        f"- S_abs_uV_per_K at eta = 2: {interpolate_s_uV(df, 2.0):.6g}",
        f"- S_abs_uV_per_K at eta = 5: {interpolate_s_uV(df, 5.0):.6g}",
        (
            "- eta >= 1 corresponds roughly to "
            f"S_abs_uV_per_K <= {eta_ge_1['S_abs_uV_per_K'].max():.6g}"
            if not eta_ge_1.empty
            else "- eta >= 1 is outside the grid"
        ),
        f"- parquet status: {parquet_status}",
        f"- elapsed_seconds: {elapsed_sec:.2f}",
        "",
        "## Monotonic Check",
        "",
        f"- eta monotonic increasing: {checks.get('eta_monotonic_increasing')}",
        f"- S_abs_uV_per_K monotonic decreasing: {checks.get('S_abs_uV_per_K_monotonic_decreasing')}",
        "",
        "## Sanity Check",
        "",
    ]
    for name, ok in checks.items():
        lines.append(f"- {name}: {ok}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This Step2A creates only the numerical lookup table.",
            "- Step1 data is not modified and eta is not assigned in this step.",
            "- F1 is computed by cumulative trapezoid integration of F0 over eta.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    df = build_lookup(args.eta_min, args.eta_max, args.d_eta)

    log("running sanity checks...")
    checks, failures = run_sanity_checks(df, args.eta_min, args.eta_max)
    if failures:
        for failure in failures:
            print(f"[step2a] FAIL: {failure}", flush=True)
        raise SystemExit(1)

    log("writing outputs...")
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / LOOKUP_CSV, index=False, encoding="utf-8-sig")
    parquet_ok, parquet_error = save_parquet(df, args.output / LOOKUP_PARQUET)
    parquet_status = "saved" if parquet_ok else f"not saved: {parquet_error}"
    write_report(
        args.report,
        df,
        args.eta_min,
        args.eta_max,
        args.d_eta,
        checks,
        parquet_status,
        time.time() - started,
    )
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
