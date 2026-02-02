import argparse
import ast
import json
import logging
import re

import pandas as pd


RE_ELEM = re.compile(r"(Si|Ge)(\d*\.?\d*)")
# 組成の妥当性確認のため、元素記号を抽出する。
RE_ELEM_ANY = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def extract_elements(composition: str) -> set[str]:
    # 組成文字列から "Si", "Ge", "Ba" のような元素記号を抽出する。
    if composition is None or pd.isna(composition):
        return set()
    return {elem for elem, _ in RE_ELEM_ANY.findall(str(composition))}


def filter_sige(df: pd.DataFrame) -> pd.DataFrame:
    allowed_elems = {"Si", "Ge"}

    def is_sige_only(composition) -> bool:
        # Si と Ge のみから成る組成だけを残す。
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
        & (df["prop_y"] == "Thermal conductivity")
    ]


def parse_xy(raw_value):
    parsed = ast.literal_eval(raw_value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("x/y is not list-like")
    return [float(v) for v in parsed]


def parse_composition(composition: str) -> tuple[float, float]:
    """
    # 例:
    # "Si0.8Ge0.2" -> (0.8, 0.2)
    # "SiGe"       -> (0.5, 0.5)
    # "Si2Ge"      -> (0.666..., 0.333...)
    """
    # 念のため Si/Ge 以外を含む組成は除外する。
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


def seebeck_sign_from_list(y_list: list[float], eps: float = 1e-9) -> str:
    if not y_list:
        return "unknown"
    mean_val = sum(y_list) / len(y_list)
    if mean_val > eps:
        return "p"
    if mean_val < -eps:
        return "n"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=r"C:\Users\miots\m-thesis\m-thesis\experiments\exp001\starrydata_curves.csv",
    )
    parser.add_argument("--out", default="sige_ThermalConductivity_curves.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    # 入力 CSV を読み込み、Si-Ge と熱伝導率データに絞る。
    df = load_csv(args.csv)
    print(f"rows_total: {len(df)}")

    df_sige = filter_sige(df)
    print(f"rows_sige: {len(df_sige)}")

    df_sige = filter_seebeck(df_sige)
    print(f"rows_prop_filtered: {len(df_sige)}")

    records = []
    for row in df_sige.itertuples(index=True):
        comp = row.composition
        # x/y を配列に変換（文字列として保存されているケースを想定）。
        try:
            x_list = parse_xy(row.x)
            y_list = parse_xy(row.y)
        except (ValueError, SyntaxError, TypeError) as exc:
            logging.warning("skip index=%s: x/y parse error: %s", row.Index, exc)
            continue

        # 組成から Si/Ge 比率を算出。
        try:
            si_frac, ge_frac = parse_composition(comp)
        except (ValueError, TypeError) as exc:
            logging.warning("skip index=%s: composition parse error: %s", row.Index, exc)
            continue

        if si_frac + ge_frac == 0:
            logging.warning("skip index=%s: Si+Ge total is zero", row.Index)
            continue

        if not x_list or not y_list:
            logging.warning("skip index=%s: empty x/y list", row.Index)
            continue

        # x>100K のみ残し、x 昇順で並べ直す
        xy_filtered = [(x, y) for x, y in zip(x_list, y_list) if x > 100.0]
        if not xy_filtered:
            logging.warning("skip index=%s: no data above 100K", row.Index)
            continue
        xy_filtered.sort(key=lambda t: t[0])
        x_list, y_list = map(list, zip(*xy_filtered))

        # 解析結果を行情報に追加して保存用のレコードにする。
        record = row._asdict()
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
            "composition",
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
    df_save["x_list"] = df_save["x_list"].apply(json.dumps)
    df_save["y_list"] = df_save["y_list"].apply(json.dumps)
    df_save.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
