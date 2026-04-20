import argparse
import ast
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd


RE_ELEM = re.compile(r"(Si|Ge)(\d*\.?\d*)")
RE_ELEM_ANY = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
RE_ALLOWED_SEPARATOR = re.compile(r"[\s\-\(\)\[\]\{\}]+")


def format_ratio(value: float) -> str:
    formatted = f"{value:.6f}"
    formatted = formatted.rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def extract_elements(composition: Any) -> set[str]:
    if composition is None or pd.isna(composition):
        return set()
    return {elem for elem, _ in RE_ELEM_ANY.findall(str(composition))}


def is_supported_sige_syntax(composition: str) -> bool:
    leftover = RE_ELEM.sub("", composition)
    leftover = RE_ALLOWED_SEPARATOR.sub("", leftover)
    return leftover == ""


def normalize_sige_composition(
    composition: Any,
) -> Optional[tuple[str, float, float]]:
    if composition is None or pd.isna(composition):
        return None

    comp_str = str(composition).strip()
    if extract_elements(comp_str) != {"Si", "Ge"}:
        return None
    if not is_supported_sige_syntax(comp_str):
        return None

    matches = RE_ELEM.findall(comp_str)
    if not matches:
        return None

    si_total = 0.0
    ge_total = 0.0
    for elem, num in matches:
        coeff = float(num) if num else 1.0
        if elem == "Si":
            si_total += coeff
        else:
            ge_total += coeff

    total = si_total + ge_total
    if total == 0.0:
        return None

    si_frac = si_total / total
    ge_frac = ge_total / total
    normalized = f"Si{format_ratio(si_frac)}Ge{format_ratio(ge_frac)}"
    return normalized, si_frac, ge_frac


def parse_xy(raw_value: Any) -> list[float]:
    if isinstance(raw_value, (list, tuple)):
        parsed = raw_value
    else:
        parsed = ast.literal_eval(str(raw_value))

    if not isinstance(parsed, (list, tuple)):
        raise ValueError("x/y is not list-like")

    return [float(value) for value in parsed]


def filter_sige_candidates(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["composition"].apply(lambda value: extract_elements(value) == {"Si", "Ge"})]


def normalize_xy_pairs(
    x_values: list[float], y_values: list[float]
) -> Optional[tuple[list[float], list[float]]]:
    if len(x_values) != len(y_values):
        return None


    pairs: list[tuple[float, float]] = []
    for x_value, y_value in zip(x_values, y_values):
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            continue
        pairs.append((x_value, y_value))

    if not pairs:
        return None

    pairs.sort(key=lambda pair: pair[0])
    x_sorted, y_sorted = zip(*pairs)
    return list(x_sorted), list(y_sorted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all SiGe rows from Starrydata, normalize composition, and save to CSV."
    )
    parser.add_argument(
        "--csv",
        default=r"C:\Users\miots\m-thesis\m-thesis\experiments\exp001\starrydata_curves.csv",
    )
    parser.add_argument(
        "--out",
        default=r"C:\Users\miots\m-thesis\m-thesis\data\output\sige_all_curves.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    df = pd.read_csv(args.csv)

    required_cols = {"composition", "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    print(f"rows_total: {len(df)}")

    df_sige = filter_sige_candidates(df)
    print(f"rows_sige_candidates: {len(df_sige)}")

    skipped_bad_composition = 0
    skipped_bad_xy = 0
    records = []

    for row in df_sige.itertuples(index=True):
        comp_original = getattr(row, "composition")
        normalized = normalize_sige_composition(comp_original)
        if normalized is None:
            skipped_bad_composition += 1
            logging.warning(
                "skip index=%s: unsupported SiGe composition syntax: %r",
                row.Index,
                comp_original,
            )
            continue

        comp_norm, si_frac, ge_frac = normalized

        try:
            x_list = parse_xy(getattr(row, "x"))
            y_list = parse_xy(getattr(row, "y"))
        except (ValueError, SyntaxError, TypeError) as exc:
            skipped_bad_xy += 1
            logging.warning("skip index=%s: x/y parse error: %s", row.Index, exc)
            continue

        xy_normalized = normalize_xy_pairs(x_list, y_list)
        if xy_normalized is None:
            skipped_bad_xy += 1
            logging.warning("skip index=%s: invalid x/y pairs", row.Index)
            continue

        x_list, y_list = xy_normalized

        record = row._asdict()
        record.pop("Index", None)
        record["composition_original"] = comp_original
        record["composition"] = comp_norm
        record["x_list"] = x_list
        record["y_list"] = y_list
        record["si_frac"] = si_frac
        record["ge_frac"] = ge_frac
        records.append(record)

    df_out = pd.DataFrame(records)

    print(f"rows_skipped_bad_composition: {skipped_bad_composition}")
    print(f"rows_skipped_bad_xy: {skipped_bad_xy}")
    print(f"rows_written: {len(df_out)}")

    preview_cols = [
        column
        for column in [
            "composition",
            "composition_original",
            "si_frac",
            "ge_frac",
            "prop_x",
            "prop_y",
            "DOI",
        ]
        if column in df_out.columns
    ]
    if preview_cols:
        print(df_out[preview_cols].head())
    else:
        print(df_out.head())

    df_save = df_out.copy()
    df_save["x_list"] = df_save["x_list"].apply(json.dumps)
    df_save["y_list"] = df_save["y_list"].apply(json.dumps)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_save.to_csv(out_path, index=False)
    print(f"saved_csv: {out_path}")


if __name__ == "__main__":
    main()
