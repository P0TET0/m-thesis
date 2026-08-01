# Build curve_master.csv from conductivity curves for a target SiGe composition.
import argparse
import ast
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGE_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "sige"
INPUT_CSV = SIGE_OUTPUT_DIR / "sige_ElectricalConductivity_curves.csv"
MASTER_DIR = PROJECT_ROOT / "data" / "output" / "master"
OUTPUT_CSV = MASTER_DIR / "curve_master.csv"
TARGET_COMPOSITION = "Si0.8Ge0.2"


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


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def build_curve_id(row: pd.Series) -> str:
    parts = [
        normalize_text(row.get("DOI")),
        normalize_text(row.get("sample_id")),
        normalize_text(row.get("figure_id")),
        normalize_text(row.get("Index")),
    ]
    return "__".join(parts)


def infer_sigma_source(prop_y: Any) -> str:
    prop_y_text = normalize_text(prop_y).lower()
    if "conductivity" in prop_y_text:
        return "original_conductivity"
    if "resistivity" in prop_y_text:
        return "converted_from_resistivity"
    return "unknown"


def is_valid_curve(prop_x: Any, x_list: list[Any], y_list: list[Any]) -> bool:
    prop_x_text = normalize_text(prop_x).lower()
    return prop_x_text == "temperature" and len(x_list) == len(y_list) and len(x_list) >= 2


def filter_by_composition(df: pd.DataFrame, composition: str | None) -> pd.DataFrame:
    if not composition:
        return df.copy()
    if "composition" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["composition"].astype(str).str.strip() == composition].copy()


def build_curve_master(input_csv: Path, output_csv: Path, composition: str | None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df = filter_by_composition(df, composition)

    parsed_x = df["x_list"].apply(safe_parse_list) if "x_list" in df.columns else pd.Series([[]] * len(df), index=df.index)
    parsed_y = df["y_list"].apply(safe_parse_list) if "y_list" in df.columns else pd.Series([[]] * len(df), index=df.index)
    prop_x_values = df["prop_x"] if "prop_x" in df.columns else pd.Series([None] * len(df), index=df.index)
    prop_y_values = df["prop_y"] if "prop_y" in df.columns else pd.Series([None] * len(df), index=df.index)

    df_out = df.copy()
    df_out["curve_id"] = df_out.apply(build_curve_id, axis=1)
    df_out["n_points"] = parsed_x.apply(len)
    df_out["sigma_source"] = prop_y_values.apply(infer_sigma_source)
    df_out["is_valid_curve"] = [
        is_valid_curve(prop_x, x_vals, y_vals)
        for prop_x, x_vals, y_vals in zip(prop_x_values, parsed_x, parsed_y)
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    return df_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curve_master.csv from sige_ElectricalConductivity_curves.csv")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV, help="input conductivity curve csv")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV, help="output curve master csv")
    parser.add_argument(
        "--composition",
        default=TARGET_COMPOSITION,
        help="target composition to keep; use an empty string to keep all rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_composition = args.composition.strip() or None
    try:
        df_out = build_curve_master(args.input_csv, args.output_csv, target_composition)
    except Exception as exc:
        raise SystemExit(f"failed to build curve master: {exc}") from exc

    print(f"saved: {args.output_csv}")
    print(f"rows: {len(df_out)}")
    print(f"valid_curves: {int(df_out['is_valid_curve'].sum())}")
    print(f"composition_filter: {target_composition or 'ALL'}")


if __name__ == "__main__":
    main()
