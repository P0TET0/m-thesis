import argparse
import ast
import json
import logging
import re
from typing import Any, List, Optional, Tuple

import pandas as pd


RE_ELEM = re.compile(r"(Si|Ge)(\d*\.?\d*)")
RE_ELEM_ANY = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def format_ratio(value: float) -> str:
    formatted = f"{value:.6f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def extract_elements(composition: Any) -> set[str]:
    if composition is None or pd.isna(composition):
        return set()
    return {elem for elem, _ in RE_ELEM_ANY.findall(str(composition))}


def normalize_sige_composition(
    composition: Any,
) -> Optional[Tuple[str, Optional[float], Optional[float]]]:
    if composition is None or pd.isna(composition):
        return None
    comp_str = str(composition).strip()

    elems = extract_elements(comp_str)
    if elems != {"Si", "Ge"}:
        return None

    matches = RE_ELEM.findall(comp_str)
    if not matches:
        return None

    si_total = 0.0
    ge_total = 0.0
    has_missing_coeff = False
    for elem, num in matches:
        if not num:
            has_missing_coeff = True
            continue
        coeff = float(num)
        if elem == "Si":
            si_total += coeff
        else:
            ge_total += coeff

    # Keep rows like "SiGe" / "Si-Ge" without inferring coefficients.
    if has_missing_coeff:
        return comp_str, None, None

    total = si_total + ge_total
    if total == 0.0:
        return None

    si_frac = si_total / total
    ge_frac = ge_total / total
    normalized = f"Si{format_ratio(si_frac)}Ge{format_ratio(ge_frac)}"
    return normalized, si_frac, ge_frac


def parse_xy(raw_value: Any) -> list[float]:
    parsed = ast.literal_eval(raw_value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("x/y is not list-like")
    return [float(v) for v in parsed]


def filter_electrical(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["prop_x"] == "Temperature")
        & (
            (df["prop_y"] == "Electrical conductivity")
            | (df["prop_y"] == "Electrical resistivity")
        )
    ]


def invert_resistivity(prop_y: str, values: List[float]) -> List[Optional[float]]:
    if prop_y != "Electrical resistivity":
        return values

    inverted: List[Optional[float]] = []
    for val in values:
        if val == 0.0:
            inverted.append(None)
        else:
            inverted.append(1.0 / val)
    return inverted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=r"C:\Users\miots\m-thesis\m-thesis\experiments\exp001\starrydata_curves.csv",
    )
    parser.add_argument("--out", default="sige_EC_ER_curves.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    df = pd.read_csv(args.csv)

    required_cols = {"composition", "prop_x", "prop_y", "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    print(f"rows_total: {len(df)}")

    df = filter_electrical(df)
    print(f"rows_prop_filtered: {len(df)}")

    records = []
    for row in df.itertuples(index=True):
        comp_original = getattr(row, "composition")
        normalized = normalize_sige_composition(comp_original)
        if normalized is None:
            continue
        comp_norm, si_frac, ge_frac = normalized

        try:
            x_list = parse_xy(getattr(row, "x"))
            y_list = parse_xy(getattr(row, "y"))
        except (ValueError, SyntaxError, TypeError) as exc:
            logging.warning("skip index=%s: x/y parse error: %s", row.Index, exc)
            continue

        y_list = invert_resistivity(getattr(row, "prop_y"), y_list)

        if not x_list or not y_list:
            logging.warning("skip index=%s: empty x/y list", row.Index)
            continue

        xy_sorted = sorted(zip(x_list, y_list), key=lambda t: t[0])
        if not xy_sorted:
            continue
        x_list, y_list = map(list, zip(*xy_sorted))

        record = row._asdict()
        record["composition_original"] = comp_original
        record["composition"] = comp_norm
        record["x_list"] = x_list
        record["y_list"] = y_list
        record["si_frac"] = si_frac
        record["ge_frac"] = ge_frac
        records.append(record)

    df_out = pd.DataFrame(records)

    preview_cols = [
        c
        for c in [
            "composition",
            "composition_original",
            "si_frac",
            "ge_frac",
            "prop_x",
            "prop_y",
            "DOI",
        ]
        if c in df_out.columns
    ]
    if preview_cols:
        print(df_out[preview_cols].head())
    else:
        print(df_out.head())

    df_save = df_out.copy()
    if "x_list" in df_save.columns:
        df_save["x_list"] = df_save["x_list"].apply(json.dumps)
    if "y_list" in df_save.columns:
        df_save["y_list"] = df_save["y_list"].apply(json.dumps)
    df_save.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
