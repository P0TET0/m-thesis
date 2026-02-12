import argparse
import ast
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd


def parse_numeric_list(raw_value: Any) -> list[float]:
    if isinstance(raw_value, (list, tuple)):
        return [float(v) for v in raw_value]
    if raw_value is None or pd.isna(raw_value):
        return []

    text = str(raw_value).strip()
    if not text:
        return []

    parsed = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            break
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue

    if not isinstance(parsed, (list, tuple)):
        raise ValueError("x/y is not list-like")

    values: list[float] = []
    for value in parsed:
        if value is None:
            continue
        values.append(float(value))
    return values


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe.strip("._") or "composition"


def iter_xy_pairs(x_values: Iterable[float], y_values: Iterable[float]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(x_values, y_values):
        if x is None or y is None:
            continue
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(xf) and math.isfinite(yf)):
            continue
        pairs.append((xf, yf))
    pairs.sort(key=lambda t: t[0])
    return pairs


def resolve_xy_columns(df: pd.DataFrame) -> tuple[str, str]:
    if {"x_list", "y_list"}.issubset(df.columns):
        return "x_list", "y_list"
    if {"x", "y"}.issubset(df.columns):
        return "x", "y"
    raise KeyError("missing x/y columns: expected x_list,y_list or x,y")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=r"C:\Users\miots\m-thesis\m-thesis\sige_ElectricalConductivity_curves.csv",
        help="input csv from SiGe_ElectricalConductivity.py",
    )
    parser.add_argument(
        "--outdir",
        default=r"C:\Users\miots\m-thesis\m-thesis\experiments\exp001\plots_sige_ec_by_composition",
        help="directory for output png files",
    )
    parser.add_argument(
        "--split-csv",
        action="store_true",
        help="also save one csv file per composition",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="dpi for saved figures",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="plot conductivity axis on log scale",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(args.csv)
    if "composition" not in df.columns:
        raise KeyError("missing column: composition")
    x_col, y_col = resolve_xy_columns(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_outdir = outdir / "split_csv"
    if args.split_csv:
        csv_outdir.mkdir(parents=True, exist_ok=True)

    grouped = df[df["composition"].notna()].groupby("composition", sort=True)
    total_groups = grouped.ngroups

    saved_figures = 0
    skipped_groups = 0
    for composition, group in grouped:
        fig, ax = plt.subplots(figsize=(8, 5))
        curve_count = 0

        for row in group.itertuples(index=False):
            try:
                x_values = parse_numeric_list(getattr(row, x_col))
                y_values = parse_numeric_list(getattr(row, y_col))
            except (ValueError, TypeError) as exc:
                logging.warning("skip row: composition=%s parse error: %s", composition, exc)
                continue

            pairs = iter_xy_pairs(x_values, y_values)
            if len(pairs) < 2:
                continue

            xs, ys = zip(*pairs)
            ax.plot(xs, ys, color="tab:blue", linewidth=1.0, alpha=0.35)
            curve_count += 1

        if curve_count == 0:
            plt.close(fig)
            skipped_groups += 1
            continue

        ax.set_title(f"{composition} ({curve_count} curves)")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Electrical conductivity [1/(ohm*m)]")
        if args.log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_name = sanitize_filename(str(composition))
        png_path = outdir / f"{safe_name}.png"
        fig.savefig(png_path, dpi=args.dpi)
        plt.close(fig)
        saved_figures += 1

        if args.split_csv:
            csv_path = csv_outdir / f"{safe_name}.csv"
            group.to_csv(csv_path, index=False)

    print(f"rows_total: {len(df)}")
    print(f"groups_total: {total_groups}")
    print(f"figures_saved: {saved_figures}")
    print(f"groups_skipped_no_curves: {skipped_groups}")
    print(f"outdir: {outdir}")
    if args.split_csv:
        print(f"split_csv_outdir: {csv_outdir}")


if __name__ == "__main__":
    main()
