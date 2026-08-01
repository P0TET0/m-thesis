# Build point_master.csv by exploding curve_master.csv into one row per temperature point.
import argparse
import ast
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_DIR = PROJECT_ROOT / "data" / "output" / "master"
INPUT_CSV = MASTER_DIR / "curve_master.csv"
OUTPUT_CSV = MASTER_DIR / "point_master.csv"
META_CSV = MASTER_DIR / "paper_curve_meta.csv"


def safe_parse_list(raw_value: Any) -> list[Any]:
    """Parse a CSV cell into a list without failing hard on malformed values."""
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, tuple):
        return list(raw_value)
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return []
    if isinstance(raw_value, str):
        try:
            parsed = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    return []


def to_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def explode_curve_row(row: pd.Series) -> list[dict[str, Any]]:
    x_values = safe_parse_list(row.get("x_list"))
    y_values = safe_parse_list(row.get("y_list"))
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return []

    curve_n_points = len(x_values)
    point_weight = 1.0 / curve_n_points
    records: list[dict[str, Any]] = []

    for point_index, (x_raw, y_raw) in enumerate(zip(x_values, y_values)):
        temp_k = to_float(x_raw)
        sigma = to_float(y_raw)
        if temp_k is None or sigma is None or sigma <= 0.0:
            continue

        records.append(
            {
                "curve_id": row.get("curve_id"),
                "DOI": row.get("DOI"),
                "sample_id": row.get("sample_id"),
                "figure_id": row.get("figure_id"),
                "composition": row.get("composition"),
                "si_frac": row.get("si_frac"),
                "ge_frac": row.get("ge_frac"),
                "point_index_in_curve": point_index,
                "curve_n_points": curve_n_points,
                "point_weight": point_weight,
                "T_K": temp_k,
                "sigma": sigma,
                "log10_sigma": float(np.log10(sigma)),
            }
        )
    return records


def load_optional_meta(meta_csv: Path) -> pd.DataFrame | None:
    if not meta_csv.exists():
        return None
    try:
        df_meta = pd.read_csv(meta_csv)
    except Exception:
        return None
    required_columns = {"curve_id", "dopant_element", "carrier_conc_cm3"}
    if not required_columns.issubset(df_meta.columns):
        return None
    return df_meta[["curve_id", "dopant_element", "carrier_conc_cm3"]].drop_duplicates(subset=["curve_id"])


def build_point_master(input_csv: Path, output_csv: Path, meta_csv: Path) -> tuple[pd.DataFrame, bool]:
    df = pd.read_csv(input_csv)
    if "is_valid_curve" not in df.columns:
        raise KeyError("missing column: is_valid_curve")

    valid_df = df[df["is_valid_curve"].apply(truthy_flag)].copy()

    point_records: list[dict[str, Any]] = []
    for row in valid_df.to_dict(orient="records"):
        point_records.extend(explode_curve_row(pd.Series(row)))

    point_df = pd.DataFrame(
        point_records,
        columns=[
            "curve_id",
            "DOI",
            "sample_id",
            "figure_id",
            "composition",
            "si_frac",
            "ge_frac",
            "point_index_in_curve",
            "curve_n_points",
            "point_weight",
            "T_K",
            "sigma",
            "log10_sigma",
        ],
    )

    meta_joined = False
    df_meta = load_optional_meta(meta_csv)
    if df_meta is not None:
        point_df = point_df.merge(df_meta, on="curve_id", how="left")
        meta_joined = True

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    point_df.to_csv(output_csv, index=False)
    return point_df, meta_joined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point_master.csv from curve_master.csv")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV, help="input curve master csv")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV, help="output point master csv")
    parser.add_argument("--meta-csv", type=Path, default=META_CSV, help="optional curve metadata csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        point_df, meta_joined = build_point_master(args.input_csv, args.output_csv, args.meta_csv)
    except Exception as exc:
        raise SystemExit(f"failed to build point master: {exc}") from exc

    print(f"saved: {args.output_csv}")
    print(f"rows: {len(point_df)}")
    print(f"meta_joined: {meta_joined}")


if __name__ == "__main__":
    main()
