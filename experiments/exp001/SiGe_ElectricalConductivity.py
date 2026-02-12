import argparse
import ast
import json
import logging
import re
from typing import Any, List, Optional, Tuple

import pandas as pd


RE_ELEM = re.compile(r"(Si|Ge)(\d*\.?\d*)")
RE_ELEM_ANY = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")

OVERRIDES = {
    (9423, "10.1143/jjap.43.5978"): "Si0.8Ge0.2",
    (10229, "10.1063/1.347717"): "Si0.8Ge0.2",
    (23629, "10.1016/j.jallcom.2019.153182"): "Si0.8Ge0.2",
}

EXCLUDED_DOIS = {
    "10.1016/0306-2619(81)90049-0",
    "10.1063/1.1661622",
    "10.1063/1.1663600",
}


def normalize_sid(value: Any) -> Any:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return str(value).strip()


def normalize_doi(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


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
) -> Optional[Tuple[str, float, float]]:
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
    parser.add_argument(
        "--out",
        default=r"C:\Users\miots\m-thesis\m-thesis\sige_ElectricalConductivity_curves.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    df = pd.read_csv(args.csv)

    required_cols = {"composition", "SID", "DOI", "prop_x", "prop_y", "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    print(f"rows_total: {len(df)}")

    df = filter_electrical(df)
    print(f"rows_prop_filtered: {len(df)}")

    records = []
    for row in df.itertuples(index=True):
        comp_original = getattr(row, "composition")
        sid = getattr(row, "SID")
        doi = getattr(row, "DOI")
        doi_norm = normalize_doi(doi)
        if doi_norm in EXCLUDED_DOIS:
            logging.warning("skip index=%s: excluded DOI=%s", row.Index, doi_norm)
            continue

        key = (normalize_sid(sid), doi_norm)

        if key in OVERRIDES:
            normalized = normalize_sige_composition(OVERRIDES[key])
        else:
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

        xy_filtered = [(x, y) for x, y in zip(x_list, y_list) if x > 100.0]
        if not xy_filtered:
            logging.warning("skip index=%s: no data above 100K", row.Index)
            continue

        xy_filtered.sort(key=lambda t: t[0])
        x_list, y_list = map(list, zip(*xy_filtered))

        record = row._asdict()
        record["composition_original"] = comp_original
        record["composition"] = comp_norm
        record["x_list"] = x_list
        record["y_list"] = y_list
        record["T_min"] = min(x_list)
        record["T_max"] = max(x_list)
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
            "T_min",
            "T_max",
            "prop_y",
            "DOI",
        ]
        if c in df_out.columns
    ]
    print(df_out[preview_cols].head())

    df_save = df_out.copy()
    df_save["x_list"] = df_save["x_list"].apply(json.dumps)
    df_save["y_list"] = df_save["y_list"].apply(json.dumps)
    df_save.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
