import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PARQUET = EXP_DIR / "data" / "processed" / "step0_te_analysis_base.parquet"
DEFAULT_INPUT_CSV = EXP_DIR / "data" / "processed" / "step0_te_analysis_base.csv"
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed"
DEFAULT_REPORT = EXP_DIR / "reports" / "step1_carrier_report.md"

REQUIRED_STEP0_COLUMNS = [
    "row_id",
    "paper_id",
    "sample_id",
    "sample_key",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "T_K",
    "S_V_per_K",
    "S_uV_per_K",
    "sigma_S_per_m",
    "match_method",
    "sigma_source",
]

ADDED_COLUMNS = [
    "S_abs_uV_per_K",
    "carrier_type",
    "carrier_type_rule",
    "is_usable_for_eta",
    "sample_group_id",
    "n_points_sample",
    "n_p_points_sample",
    "n_n_points_sample",
    "n_unknown_points_sample",
    "sample_has_sign_change",
    "sample_carrier_behavior",
    "is_conservative_main_analysis",
]

CLASSIFIED_CSV = "step1_te_carrier_classified.csv"
CLASSIFIED_PARQUET = "step1_te_carrier_classified.parquet"
ETA_CANDIDATES_CSV = "step1_eta_input_candidates.csv"
ETA_CANDIDATES_PARQUET = "step1_eta_input_candidates.parquet"
CONSERVATIVE_CSV = "step1_conservative_main_candidates.csv"
CONSERVATIVE_PARQUET = "step1_conservative_main_candidates.parquet"
SAMPLE_SUMMARY_CSV = "step1_sample_sign_summary.csv"
FAMILY_COUNTS_CSV = "step1_carrier_counts_by_material_family.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify p/n carrier type from Step0 Seebeck sign."
    )
    parser.add_argument("--input", type=Path, default=None, help="Step0 parquet or CSV input")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output directory")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="report markdown path")
    parser.add_argument("--zero-threshold-uV", type=float, default=1.0)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step1] {message}", flush=True)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def resolve_input(explicit_input: Path | None) -> Path:
    if explicit_input is not None:
        if not explicit_input.exists():
            raise FileNotFoundError(
                f"Step0 input file not found: {explicit_input}. "
                "Run experiments/exp006/build_step0_table.py first, or pass an existing Step0 CSV/parquet."
            )
        return explicit_input
    if DEFAULT_INPUT_PARQUET.exists():
        return DEFAULT_INPUT_PARQUET
    if DEFAULT_INPUT_CSV.exists():
        return DEFAULT_INPUT_CSV
    raise FileNotFoundError(
        "Step0 input file not found. Expected experiments/exp006/data/processed/"
        "step0_te_analysis_base.parquet or experiments/exp006/data/processed/"
        "step0_te_analysis_base.csv. Run Step0 first."
    )


def read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported Step0 input extension: {path.suffix}")


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_STEP0_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Step0 input is missing required columns: {missing}")


def make_sample_group_id(row: pd.Series) -> str:
    paper_id = clean_text(row.get("paper_id"))
    sample_id = clean_text(row.get("sample_id"))
    sample_key = clean_text(row.get("sample_key"))
    row_id = clean_text(row.get("row_id"))
    if paper_id and sample_id:
        return f"{paper_id}::{sample_id}"
    if paper_id and sample_key:
        return f"{paper_id}::{sample_key}"
    if sample_key:
        return f"unknown_paper::{sample_key}"
    return f"row::{row_id or row.name}"


def assign_carrier_type(s_uV: pd.Series, threshold: float) -> pd.Series:
    values = pd.to_numeric(s_uV, errors="coerce")
    return pd.Series(
        np.select(
            [values > threshold, values < -threshold, values.abs() <= threshold],
            ["p", "n", "unknown_near_zero"],
            default="unknown_near_zero",
        ),
        index=s_uV.index,
    )


def behavior_from_counts(n_p: int, n_n: int, n_unknown: int) -> str:
    if n_p > 0 and n_n > 0:
        return "mixed_sign"
    if n_p > 0:
        return "p_only"
    if n_n > 0:
        return "n_only"
    return "unknown_only"


def build_sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "sample_group_id",
        "paper_id",
        "sample_id",
        "sample_key",
        "formula_raw",
        "material_name_raw",
        "material_family_raw",
    ]
    first = df.groupby("sample_group_id", sort=False)[base_cols[1:]].first().reset_index()
    counts = (
        df.pivot_table(
            index="sample_group_id",
            columns="carrier_type",
            values="row_id",
            aggfunc="count",
            fill_value=0,
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )
    for col in ["p", "n", "unknown_near_zero"]:
        if col not in counts.columns:
            counts[col] = 0
    total = df.groupby("sample_group_id", sort=False).size().rename("n_points_sample").reset_index()
    ranges = (
        df.groupby("sample_group_id", sort=False)
        .agg(
            T_min_K=("T_K", "min"),
            T_max_K=("T_K", "max"),
            S_min_uV_per_K=("S_uV_per_K", "min"),
            S_max_uV_per_K=("S_uV_per_K", "max"),
        )
        .reset_index()
    )
    summary = first.merge(total, on="sample_group_id", how="left")
    summary = summary.merge(counts[["sample_group_id", "p", "n", "unknown_near_zero"]], on="sample_group_id", how="left")
    summary = summary.merge(ranges, on="sample_group_id", how="left")
    summary = summary.rename(
        columns={
            "p": "n_p_points_sample",
            "n": "n_n_points_sample",
            "unknown_near_zero": "n_unknown_points_sample",
        }
    )
    for col in ["n_p_points_sample", "n_n_points_sample", "n_unknown_points_sample"]:
        summary[col] = summary[col].fillna(0).astype(int)
    summary["sample_has_sign_change"] = (
        (summary["n_p_points_sample"] > 0) & (summary["n_n_points_sample"] > 0)
    )
    summary["sample_carrier_behavior"] = [
        behavior_from_counts(int(p), int(n), int(u))
        for p, n, u in zip(
            summary["n_p_points_sample"],
            summary["n_n_points_sample"],
            summary["n_unknown_points_sample"],
        )
    ]
    ordered = [
        "sample_group_id",
        "paper_id",
        "sample_id",
        "sample_key",
        "formula_raw",
        "material_name_raw",
        "material_family_raw",
        "n_points_sample",
        "n_p_points_sample",
        "n_n_points_sample",
        "n_unknown_points_sample",
        "sample_has_sign_change",
        "sample_carrier_behavior",
        "T_min_K",
        "T_max_K",
        "S_min_uV_per_K",
        "S_max_uV_per_K",
    ]
    return summary[ordered]


def add_step1_columns(df: pd.DataFrame, zero_threshold_uV: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["S_V_per_K"] = pd.to_numeric(out["S_V_per_K"], errors="coerce")
    out["S_uV_per_K"] = pd.to_numeric(out["S_uV_per_K"], errors="coerce")
    out["T_K"] = pd.to_numeric(out["T_K"], errors="coerce")
    out["sigma_S_per_m"] = pd.to_numeric(out["sigma_S_per_m"], errors="coerce")
    out["S_abs_uV_per_K"] = out["S_uV_per_K"].abs()
    out["carrier_type"] = assign_carrier_type(out["S_uV_per_K"], zero_threshold_uV)
    out["carrier_type_rule"] = f"S_sign_threshold_{zero_threshold_uV:g}_uV"
    out["is_usable_for_eta"] = out["carrier_type"].isin(["p", "n"])
    out["sample_group_id"] = out.apply(make_sample_group_id, axis=1)

    sample_summary = build_sample_summary(out)
    merge_cols = [
        "sample_group_id",
        "n_points_sample",
        "n_p_points_sample",
        "n_n_points_sample",
        "n_unknown_points_sample",
        "sample_has_sign_change",
        "sample_carrier_behavior",
    ]
    out = out.merge(sample_summary[merge_cols], on="sample_group_id", how="left")
    out["is_conservative_main_analysis"] = (
        out["carrier_type"].isin(["p", "n"]) & ~out["sample_has_sign_change"].astype(bool)
    )
    return out, sample_summary


def build_family_counts(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["material_family_raw", "carrier_type"], dropna=False, sort=False)
        .agg(
            row_count=("row_id", "count"),
            sample_count=("sample_group_id", "nunique"),
            paper_count=("paper_id", "nunique"),
            T_min_K=("T_K", "min"),
            T_max_K=("T_K", "max"),
            S_min_uV_per_K=("S_uV_per_K", "min"),
            S_max_uV_per_K=("S_uV_per_K", "max"),
        )
        .reset_index()
    )
    return grouped


def save_parquet(df: pd.DataFrame, path: Path) -> tuple[bool, str]:
    try:
        df.to_parquet(path, index=False)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def write_outputs(
    df: pd.DataFrame,
    sample_summary: pd.DataFrame,
    family_counts: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eta_candidates = df[df["is_usable_for_eta"]].copy()
    conservative = df[df["is_conservative_main_analysis"]].copy()

    df.to_csv(output_dir / CLASSIFIED_CSV, index=False, encoding="utf-8-sig")
    eta_candidates.to_csv(output_dir / ETA_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    conservative.to_csv(output_dir / CONSERVATIVE_CSV, index=False, encoding="utf-8-sig")
    sample_summary.to_csv(output_dir / SAMPLE_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    family_counts.to_csv(output_dir / FAMILY_COUNTS_CSV, index=False, encoding="utf-8-sig")

    statuses: dict[str, str] = {}
    for frame, filename in [
        (df, CLASSIFIED_PARQUET),
        (eta_candidates, ETA_CANDIDATES_PARQUET),
        (conservative, CONSERVATIVE_PARQUET),
    ]:
        ok, error = save_parquet(frame, output_dir / filename)
        statuses[filename] = "saved" if ok else f"not saved: {error}"
    return statuses


def run_sanity_checks(
    input_rows: int,
    df: pd.DataFrame,
    eta_candidates: pd.DataFrame,
    conservative: pd.DataFrame,
    sample_summary: pd.DataFrame,
) -> tuple[dict[str, bool], list[str], list[str]]:
    checks: dict[str, bool] = {}
    failures: list[str] = []
    warnings: list[str] = []

    checks["input_rows_equal_output_rows"] = input_rows == len(df)
    checks["row_id_unique"] = df["row_id"].is_unique
    checks["S_V_per_K_finite"] = bool(np.isfinite(pd.to_numeric(df["S_V_per_K"], errors="coerce")).all())
    checks["S_uV_per_K_finite"] = bool(np.isfinite(pd.to_numeric(df["S_uV_per_K"], errors="coerce")).all())
    checks["S_uV_per_K_consistent"] = bool(
        np.allclose(
            pd.to_numeric(df["S_uV_per_K"], errors="coerce"),
            pd.to_numeric(df["S_V_per_K"], errors="coerce") * 1e6,
            rtol=1e-6,
            atol=1e-9,
        )
    )
    checks["carrier_type_not_missing"] = bool(df["carrier_type"].notna().all())
    checks["carrier_type_allowed"] = set(df["carrier_type"].dropna()).issubset({"p", "n", "unknown_near_zero"})
    checks["is_usable_for_eta_rule"] = bool((df["is_usable_for_eta"] == df["carrier_type"].isin(["p", "n"])).all())
    expected_sign_change = (
        sample_summary.set_index("sample_group_id")
        .apply(lambda row: bool(row["n_p_points_sample"] > 0 and row["n_n_points_sample"] > 0), axis=1)
    )
    actual_sign_change = sample_summary.set_index("sample_group_id")["sample_has_sign_change"].astype(bool)
    checks["sample_has_sign_change_rule"] = bool(expected_sign_change.equals(actual_sign_change))
    checks["is_conservative_main_analysis_rule"] = bool(
        (
            df["is_conservative_main_analysis"]
            == (df["carrier_type"].isin(["p", "n"]) & ~df["sample_has_sign_change"].astype(bool))
        ).all()
    )
    checks["eta_candidates_carrier_type_p_or_n_only"] = set(eta_candidates["carrier_type"].dropna()).issubset({"p", "n"})
    checks["conservative_candidates_no_sign_change"] = bool(
        conservative.empty or (~conservative["sample_has_sign_change"].astype(bool)).all()
    )
    checks["T_K_positive_finite"] = bool(
        np.isfinite(pd.to_numeric(df["T_K"], errors="coerce")).all()
        and (pd.to_numeric(df["T_K"], errors="coerce") > 0).all()
    )
    checks["sigma_S_per_m_positive_finite"] = bool(
        np.isfinite(pd.to_numeric(df["sigma_S_per_m"], errors="coerce")).all()
        and (pd.to_numeric(df["sigma_S_per_m"], errors="coerce") > 0).all()
    )
    for name, ok in checks.items():
        if not ok:
            failures.append(name)
    if df["sample_group_id"].astype(str).str.startswith("row::").any():
        warnings.append("sample_key と paper_id が両方欠損したため row_id 由来の sample_group_id があります。")
    return checks, failures, warnings


def format_counts(series: pd.Series) -> str:
    if series.empty:
        return "{}"
    return str(series.value_counts(dropna=False).to_dict())


def format_share(series: pd.Series) -> str:
    if series.empty:
        return "{}"
    return str((series.value_counts(dropna=False, normalize=True) * 100).round(3).to_dict())


def numeric_summary(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    return f"min={values.min():.6g}, max={values.max():.6g}, median={values.median():.6g}"


def crosstab_markdown(df: pd.DataFrame, index: str, columns: str) -> str:
    if df.empty:
        return "n/a"
    table = pd.crosstab(df[index].fillna(""), df[columns].fillna(""))
    return dataframe_to_markdown(table.reset_index())


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "n/a"
    text_df = df.copy()
    text_df.columns = [str(col) for col in text_df.columns]
    for column in text_df.columns:
        text_df[column] = text_df[column].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text_df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in text_df.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in text_df.columns) + " |"
        for _, row in text_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    report_path: Path,
    input_path: Path,
    input_rows: int,
    df: pd.DataFrame,
    eta_candidates: pd.DataFrame,
    conservative: pd.DataFrame,
    sample_summary: pd.DataFrame,
    family_counts: pd.DataFrame,
    parquet_statuses: dict[str, str],
    zero_threshold_uV: float,
    checks: dict[str, bool],
    warnings: list[str],
    elapsed_sec: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    carrier_counts = df["carrier_type"].value_counts(dropna=False)
    behavior_counts = sample_summary["sample_carrier_behavior"].value_counts(dropna=False)
    mixed_examples = sample_summary[sample_summary["sample_carrier_behavior"] == "mixed_sign"].head(20)

    lines = [
        "# Step1 Carrier Classification Report",
        "",
        "## Summary",
        "",
        f"- input_file: {input_path}",
        f"- input_rows: {input_rows}",
        f"- output_rows: {len(df)}",
        f"- zero_threshold_uV: {zero_threshold_uV:g}",
        f"- carrier_type row counts: {carrier_counts.to_dict()}",
        f"- carrier_type row shares percent: {format_share(df['carrier_type'])}",
        f"- p data points: {int(carrier_counts.get('p', 0))}",
        f"- n data points: {int(carrier_counts.get('n', 0))}",
        f"- unknown_near_zero data points: {int(carrier_counts.get('unknown_near_zero', 0))}",
        f"- is_usable_for_eta == True rows: {int(df['is_usable_for_eta'].sum())}",
        f"- is_conservative_main_analysis == True rows: {int(df['is_conservative_main_analysis'].sum())}",
        f"- sample_group_id count: {df['sample_group_id'].nunique()}",
        f"- p_only samples: {int(behavior_counts.get('p_only', 0))}",
        f"- n_only samples: {int(behavior_counts.get('n_only', 0))}",
        f"- mixed_sign samples: {int(behavior_counts.get('mixed_sign', 0))}",
        f"- unknown_only samples: {int(behavior_counts.get('unknown_only', 0))}",
        f"- S_uV_per_K summary: {numeric_summary(df['S_uV_per_K'])}",
        f"- S_abs_uV_per_K summary: {numeric_summary(df['S_abs_uV_per_K'])}",
        f"- elapsed_seconds: {elapsed_sec:.2f}",
        "",
        "## Parquet Status",
        "",
    ]
    for filename, status in parquet_statuses.items():
        lines.append(f"- {filename}: {status}")
    lines.extend(["", "## Mixed Sign Sample Examples"])
    if mixed_examples.empty:
        lines.append("")
        lines.append("- none")
    else:
        lines.append("")
        lines.append(
            dataframe_to_markdown(
                mixed_examples[
                [
                    "sample_group_id",
                    "paper_id",
                    "sample_id",
                    "sample_key",
                    "formula_raw",
                    "n_points_sample",
                    "n_p_points_sample",
                    "n_n_points_sample",
                    "n_unknown_points_sample",
                    "T_min_K",
                    "T_max_K",
                    "S_min_uV_per_K",
                    "S_max_uV_per_K",
                ]
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Material Family Carrier Overview",
            "",
            crosstab_markdown(df, "material_family_raw", "carrier_type"),
            "",
            "## Match Method Carrier Overview",
            "",
            crosstab_markdown(df, "match_method", "carrier_type"),
            "",
            "## Sigma Source Carrier Overview",
            "",
            crosstab_markdown(df, "sigma_source", "carrier_type"),
            "",
            "## Carrier Counts By Material Family",
            "",
            dataframe_to_markdown(family_counts) if not family_counts.empty else "n/a",
            "",
            "## Sanity Check",
            "",
        ]
    )
    for name, ok in checks.items():
        lines.append(f"- {name}: {ok}")
    lines.extend(["", "## Warnings And Step2 Notes", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
    else:
        lines.append("- WARNING: none")
    lines.extend(
        [
            "- Step2 では eta 計算に進む前に、mixed_sign sample を主解析から除くか感度分析に回すかを判断してください。",
            "- unknown_near_zero は S がしきい値近傍のため、eta 入力候補からは外しています。",
            "- eta、F0_eta、sigma0 はこの Step1 では計算していません。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    input_path = resolve_input(args.input)

    log("loading input...")
    df = read_input(input_path)
    log(f"input rows: {len(df)}")

    log("validating required columns...")
    validate_required_columns(df)

    log("assigning carrier_type...")
    classified, sample_summary = add_step1_columns(df, args.zero_threshold_uV)

    log("summarizing sample sign behavior...")
    family_counts = build_family_counts(classified)
    eta_candidates = classified[classified["is_usable_for_eta"]].copy()
    conservative = classified[classified["is_conservative_main_analysis"]].copy()

    log("writing outputs...")
    parquet_statuses = write_outputs(classified, sample_summary, family_counts, args.output)

    log("running sanity checks...")
    checks, failures, warnings = run_sanity_checks(
        len(df), classified, eta_candidates, conservative, sample_summary
    )
    write_report(
        args.report,
        input_path,
        len(df),
        classified,
        eta_candidates,
        conservative,
        sample_summary,
        family_counts,
        parquet_statuses,
        args.zero_threshold_uV,
        checks,
        warnings,
        time.time() - started,
    )
    if failures:
        for failure in failures:
            print(f"[step1] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
