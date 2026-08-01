# Starrydata から SiGe のゼーベック係数データを抽出し、組成別の CSV とプロットを作るスクリプト。
import argparse
import ast
import json
import logging
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGE_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "sige"
DEFAULT_INPUT_CSV = PROJECT_ROOT / "experiments" / "exp001" / "starrydata_curves.csv"
DEFAULT_OUTPUT_CSV = SIGE_OUTPUT_DIR / "sige_Seebeck_curves.csv"

RE_ELEM = re.compile(r"(Si|Ge)(\d*\.?\d*)")
# Extract any element symbols for composition validation.
RE_ELEM_ANY = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def extract_elements(composition: str) -> set[str]:
    # Pull element symbols like "Si", "Ge", "Ba" from the composition string.
    if composition is None or pd.isna(composition):
        return set()
    return {elem for elem, _ in RE_ELEM_ANY.findall(str(composition))}


def filter_sige(df: pd.DataFrame) -> pd.DataFrame:
    allowed_elems = {"Si", "Ge"}

    def is_sige_only(composition) -> bool:
        # Keep only compositions made exclusively of Si and Ge.
        elems = extract_elements(composition)
        if not elems:
            return False
        if not elems.issubset(allowed_elems):
            return False
        if "Si" not in elems or "Ge" not in elems:
            return False
        return True

    return df[df["composition"].apply(is_sige_only)]


def filter_seebeck(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["prop_x"] == "Temperature")
        & (df["prop_y"] == "Seebeck coefficient")
    ]


def parse_xy(raw_value):
    parsed = ast.literal_eval(raw_value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("x/y is not list-like")
    return [float(v) for v in parsed]


def format_ratio(value: float) -> str:
    formatted = f"{value:.6f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def parse_composition(composition: str) -> tuple[float, float]:
    """
    # Examples:
    # "Si0.8Ge0.2" -> (0.8, 0.2)
    # "SiGe"       -> (0.5, 0.5)
    # "Si2Ge"      -> (0.666..., 0.333...)
    """
    # Defensive guard: reject non-SiGe compositions.
    elems = extract_elements(composition)
    if not elems:
        raise ValueError("composition has no elements")
    if not elems.issubset({"Si", "Ge"}) or "Si" not in elems or "Ge" not in elems:
        raise ValueError("composition is not SiGe-only")
    si = 0.0
    ge = 0.0
    for elem, num in RE_ELEM.findall(composition):
        coeff = float(num) if num else 1.0
        if elem == "Si":
            si += coeff
        elif elem == "Ge":
            ge += coeff
    total = si + ge
    if total == 0:
        return 0.0, 0.0
    return si / total, ge / total


def normalize_composition_ratio(si_frac: float, ge_frac: float) -> str:
    return f"Si{format_ratio(si_frac)}Ge{format_ratio(ge_frac)}"


def seebeck_sign_from_list(y_list: list[float], eps: float = 1e-9) -> str:
    if not y_list:
        return "unknown"
    mean_val = sum(y_list) / len(y_list)
    if mean_val > eps:
        return "p"
    if mean_val < -eps:
        return "n"
    return "unknown"


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe.strip("._") or "composition"


def iter_xy_pairs(
    x_values: Iterable[float], y_values: Iterable[float]
) -> list[tuple[float, float]]:
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


def plot_by_composition_ratio(
    df: pd.DataFrame,
    outdir: Path,
    dpi: int = 150,
    split_csv: bool = False,
) -> tuple[int, int, int]:
    outdir.mkdir(parents=True, exist_ok=True)

    csv_outdir = outdir / "split_csv"
    if split_csv:
        csv_outdir.mkdir(parents=True, exist_ok=True)

    if "composition_ratio" not in df.columns:
        return 0, 0, 0

    grouped = df[df["composition_ratio"].notna()].groupby("composition_ratio", sort=True)
    total_groups = grouped.ngroups

    saved_figures = 0
    skipped_groups = 0

    for composition_ratio, group in grouped:
        fig, ax = plt.subplots(figsize=(8, 5))
        curve_count = 0

        for row in group.itertuples(index=False):
            x_values = getattr(row, "x_list", [])
            y_values = getattr(row, "y_list", [])
            pairs = iter_xy_pairs(x_values, y_values)
            if len(pairs) < 2:
                continue

            xs, ys = zip(*pairs)
            ax.plot(xs, ys, color="tab:orange", linewidth=1.0, alpha=0.35)
            curve_count += 1

        if curve_count == 0:
            plt.close(fig)
            skipped_groups += 1
            continue

        ax.set_title(f"{composition_ratio} ({curve_count} curves)")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Seebeck coefficient [V/K]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_name = sanitize_filename(str(composition_ratio))
        png_path = outdir / f"{safe_name}.png"
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)
        saved_figures += 1

        if split_csv:
            csv_path = csv_outdir / f"{safe_name}.csv"
            group.to_csv(csv_path, index=False)

    return total_groups, saved_figures, skipped_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=DEFAULT_INPUT_CSV,
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_CSV,
    )
    parser.add_argument(
        "--plot-outdir",
        default=r"C:\Users\miots\m-thesis\m-thesis\data\output\sige_Seebeck_by_composition",
        help="directory for output png files grouped by composition ratio",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="dpi for saved figures",
    )
    parser.add_argument(
        "--split-csv",
        action="store_true",
        help="also save one csv file per composition ratio",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    df = load_csv(args.csv)
    print(f"rows_total: {len(df)}")

    df_sige = filter_sige(df)
    print(f"rows_sige: {len(df_sige)}")

    df_sige = filter_seebeck(df_sige)
    print(f"rows_prop_filtered: {len(df_sige)}")

    records = []
    for row in df_sige.itertuples(index=True):
        comp_original = row.composition
        try:
            x_list = parse_xy(row.x)
            y_list = parse_xy(row.y)
        except (ValueError, SyntaxError, TypeError) as exc:
            logging.warning("skip index=%s: x/y parse error: %s", row.Index, exc)
            continue

        try:
            si_frac, ge_frac = parse_composition(comp_original)
        except (ValueError, TypeError) as exc:
            logging.warning("skip index=%s: composition parse error: %s", row.Index, exc)
            continue

        if si_frac + ge_frac == 0:
            logging.warning("skip index=%s: Si+Ge total is zero", row.Index)
            continue

        if not x_list or not y_list:
            logging.warning("skip index=%s: empty x/y list", row.Index)
            continue

        # Keep points above 100 K only.
        xy_filtered = [(x, y) for x, y in zip(x_list, y_list) if x > 100.0]
        if not xy_filtered:
            logging.warning("skip index=%s: no data above 100K", row.Index)
            continue
        xy_filtered.sort(key=lambda t: t[0])
        x_list, y_list = map(list, zip(*xy_filtered))

        composition_ratio = normalize_composition_ratio(si_frac, ge_frac)

        record = row._asdict()
        record["composition_original"] = comp_original
        record["composition_ratio"] = composition_ratio
        record["x_list"] = x_list
        record["y_list"] = y_list
        record["T_min"] = min(x_list)
        record["T_max"] = max(x_list)
        record["seebeck_sign"] = seebeck_sign_from_list(y_list)
        record["si_frac"] = si_frac
        record["ge_frac"] = ge_frac
        records.append(record)

    df_out = pd.DataFrame(records)

    preview_cols = [
        c
        for c in [
            "composition_ratio",
            "composition",
            "composition_original",
            "si_frac",
            "ge_frac",
            "T_min",
            "T_max",
            "seebeck_sign",
            "DOI",
        ]
        if c in df_out.columns
    ]
    print(df_out[preview_cols].head())

    df_save = df_out.copy()
    if "x_list" in df_save.columns:
        df_save["x_list"] = df_save["x_list"].apply(json.dumps)
    if "y_list" in df_save.columns:
        df_save["y_list"] = df_save["y_list"].apply(json.dumps)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_save.to_csv(out_path, index=False)

    group_total, figures_saved, groups_skipped = plot_by_composition_ratio(
        df=df_out,
        outdir=Path(args.plot_outdir),
        dpi=args.dpi,
        split_csv=args.split_csv,
    )
    print(f"plot_groups_total: {group_total}")
    print(f"figures_saved: {figures_saved}")
    print(f"groups_skipped_no_curves: {groups_skipped}")
    print(f"plot_outdir: {args.plot_outdir}")
    if args.split_csv:
        print(f"split_csv_outdir: {Path(args.plot_outdir) / 'split_csv'}")


if __name__ == "__main__":
    main()
