import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_DIR = EXP_DIR / "reports"

DEFAULT_INPUT_PARQUET = PROCESSED_DIR / "step4_sigma0_binned_input_rows.parquet"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "step4_sigma0_binned_input_rows.csv"

REQUIRED_COLUMNS = [
    "row_id",
    "paper_id",
    "doi",
    "sample_id",
    "sample_key",
    "sample_group_id",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "T_K",
    "T_bin_index",
    "T_bin_left_K",
    "T_bin_right_K",
    "T_bin_center_K",
    "T_bin_label",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "carrier_type",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "eta",
    "F0_eta",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "is_valid_sigma0",
    "is_conservative_valid_sigma0",
    "is_conservative_main_analysis",
    "sample_has_sign_change",
    "sigma_source",
    "match_method",
]

CORE_REQUIRED_COLUMNS = [
    "row_id",
    "sample_group_id",
    "T_K",
    "T_bin_center_K",
    "carrier_type",
    "sigma_S_per_m",
    "eta",
    "F0_eta",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "is_valid_sigma0",
    "is_conservative_valid_sigma0",
]

DROP_COLUMNS = [
    "row_id",
    "reject_reason",
    "T_K",
    "carrier_type",
    "sigma_S_per_m",
    "F0_eta",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "is_valid_sigma0",
    "is_conservative_valid_sigma0",
    "paper_id",
    "sample_id",
    "sample_key",
    "material_family_raw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step5A validation splits and coverage preflight.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--min-rows-per-bin", type=int, default=3)
    parser.add_argument("--min-samples-per-bin", type=int, default=3)
    parser.add_argument("--min-papers-per-bin", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step5a] {message}", flush=True)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def resolve_input(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"Step4 binned input rows not found: {explicit}")
    if DEFAULT_INPUT_PARQUET.exists():
        return DEFAULT_INPUT_PARQUET
    if DEFAULT_INPUT_CSV.exists():
        return DEFAULT_INPUT_CSV
    raise FileNotFoundError("Step4 binned input rows not found in experiments/exp006/data/processed")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def clean_material_family(value: Any) -> str:
    text = clean_text(value)
    if not text or text.casefold() in {"unknown", "unknown_material_family"}:
        return "unknown_material_family"
    return text


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def stable_uniform(group_id: Any, seed: int) -> float:
    key = f"{seed}::{clean_text(group_id)}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:16]
    return int(digest, 16) / float(16**16)


def validation_sample_group_id(row: pd.Series) -> str:
    sample_group_id = clean_text(row.get("sample_group_id"))
    if sample_group_id:
        return sample_group_id
    paper_id = clean_text(row.get("paper_id")) or "unknown_paper"
    sample_id = clean_text(row.get("sample_id"))
    if sample_id:
        return f"{paper_id}::{sample_id}"
    sample_key = clean_text(row.get("sample_key"))
    if sample_key:
        return f"{paper_id}::{sample_key}"
    return f"row_fallback::{clean_text(row.get('row_id'))}"


def validation_paper_group_id(row: pd.Series) -> str:
    paper_id = clean_text(row.get("paper_id"))
    if paper_id:
        return paper_id
    doi = clean_text(row.get("doi"))
    if doi:
        return doi
    return f"unknown_paper::{row['validation_sample_group_id']}"


def validate_columns(df: pd.DataFrame) -> None:
    missing_core = sorted(set(CORE_REQUIRED_COLUMNS) - set(df.columns))
    if missing_core:
        raise ValueError(f"input missing required analysis columns: {missing_core}")
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""


def reject_reason(row: pd.Series) -> str:
    if not bool(row.get("is_valid_sigma0_bool", False)):
        return "is_valid_sigma0_not_true"
    for col, reason in [
        ("sigma_S_per_m_num", "invalid_sigma"),
        ("F0_eta_num", "invalid_F0_eta"),
        ("sigma0_S_per_m_num", "invalid_sigma0"),
        ("T_K_num", "invalid_T_K"),
    ]:
        value = row.get(col, np.nan)
        if not np.isfinite(value) or value <= 0:
            return reason
    for col, reason in [
        ("log10_sigma0_S_per_m_num", "invalid_log10_sigma0"),
        ("T_bin_center_K_num", "invalid_T_bin_center"),
    ]:
        value = row.get(col, np.nan)
        if not np.isfinite(value):
            return reason
    if str(row.get("carrier_type", "")) not in {"p", "n"}:
        return "invalid_carrier_type"
    return ""


def filter_validation_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["is_valid_sigma0_bool"] = as_bool(work["is_valid_sigma0"])
    work["is_conservative_valid_sigma0_bool"] = as_bool(work["is_conservative_valid_sigma0"])
    numeric_columns = [
        "sigma_S_per_m",
        "F0_eta",
        "sigma0_S_per_m",
        "log10_sigma0_S_per_m",
        "T_K",
        "T_bin_left_K",
        "T_bin_right_K",
        "T_bin_center_K",
        "log10_sigma_S_per_m",
        "eta",
        "S_uV_per_K",
        "S_abs_uV_per_K",
    ]
    for column in numeric_columns:
        work[f"{column}_num"] = pd.to_numeric(work[column], errors="coerce")
    work["reject_reason"] = work.apply(reject_reason, axis=1)
    usable = work[work["reject_reason"].eq("")].copy()
    for column in numeric_columns:
        if column in usable.columns:
            usable[column] = usable[f"{column}_num"]
    dropped = work[~work["reject_reason"].eq("")].copy()
    dropped_out = pd.DataFrame(columns=DROP_COLUMNS)
    if not dropped.empty:
        for column in DROP_COLUMNS:
            dropped_out[column] = dropped[column] if column in dropped.columns else ""
    return usable.drop(columns=["reject_reason"]), dropped_out


def assign_ids_and_splits(df: pd.DataFrame, test_size: float, n_folds: int, seed: int) -> pd.DataFrame:
    if not (0.0 < test_size < 1.0):
        raise ValueError("--test-size must be between 0 and 1")
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2")
    out = df.copy()
    out["validation_sample_group_id"] = out.apply(validation_sample_group_id, axis=1)
    out["validation_paper_group_id"] = out.apply(validation_paper_group_id, axis=1)
    out["material_group_key"] = out["material_family_raw"].map(clean_material_family)
    out["sample_holdout_split"] = [
        "test" if stable_uniform(group_id, seed) < test_size else "train"
        for group_id in out["validation_sample_group_id"]
    ]
    out["paper_holdout_split"] = [
        "test" if stable_uniform(group_id, seed) < test_size else "train"
        for group_id in out["validation_paper_group_id"]
    ]
    out["sample_cv_fold"] = [
        min(n_folds - 1, int(math.floor(stable_uniform(group_id, seed + 99991) * n_folds)))
        for group_id in out["validation_sample_group_id"]
    ]
    out["paper_cv_fold"] = [
        min(n_folds - 1, int(math.floor(stable_uniform(group_id, seed + 99991) * n_folds)))
        for group_id in out["validation_paper_group_id"]
    ]
    out["is_sample_holdout_train"] = out["sample_holdout_split"].eq("train")
    out["is_sample_holdout_test"] = out["sample_holdout_split"].eq("test")
    out["is_paper_holdout_train"] = out["paper_holdout_split"].eq("train")
    out["is_paper_holdout_test"] = out["paper_holdout_split"].eq("test")
    out["eval_eligible_all_valid"] = True
    out["eval_eligible_conservative_valid"] = out["is_conservative_valid_sigma0_bool"]
    return out


def unique_examples(series: pd.Series, limit: int = 5) -> str:
    values: list[str] = []
    for value in series:
        text = clean_text(value)
        if text and text not in values:
            values.append(text)
        if len(values) >= limit:
            break
    return " | ".join(values)


def build_group_assignments(df: pd.DataFrame, group_col: str, split_col: str, fold_col: str, kind: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, group in df.groupby(group_col, sort=False):
        row = {
            group_col: group_id,
            f"{kind}_holdout_split": group[split_col].iloc[0],
            f"{kind}_cv_fold": int(group[fold_col].iloc[0]),
            "row_count": len(group),
            "material_family_count": group["material_group_key"].nunique(),
            "carrier_type_values": unique_examples(group["carrier_type"]),
            "T_min_K": group["T_K"].min(),
            "T_max_K": group["T_K"].max(),
            "material_family_raw_examples": unique_examples(group["material_family_raw"]),
        }
        if kind == "sample":
            row["paper_count"] = group["validation_paper_group_id"].nunique()
            row["formula_raw_examples"] = unique_examples(group["formula_raw"])
        else:
            row["sample_count"] = group["validation_sample_group_id"].nunique()
            row["paper_id_examples"] = unique_examples(group["paper_id"])
            row["doi_examples"] = unique_examples(group["doi"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_subset(df: pd.DataFrame, split_scheme: str, split_label: str) -> dict[str, Any]:
    return {
        "split_scheme": split_scheme,
        "split_label": str(split_label),
        "row_count": len(df),
        "sample_count": df["validation_sample_group_id"].nunique(),
        "paper_count": df["validation_paper_group_id"].nunique(),
        "p_row_count": int(df["carrier_type"].eq("p").sum()),
        "n_row_count": int(df["carrier_type"].eq("n").sum()),
        "conservative_row_count": int(df["eval_eligible_conservative_valid"].sum()),
        "material_family_count": df["material_group_key"].nunique(),
        "T_min_K": df["T_K"].min() if not df.empty else np.nan,
        "T_max_K": df["T_K"].max() if not df.empty else np.nan,
        "sigma_median_S_per_m": df["sigma_S_per_m"].median() if not df.empty else np.nan,
        "log10_sigma_median_S_per_m": df["log10_sigma_S_per_m"].median() if not df.empty else np.nan,
        "sigma0_median_S_per_m": df["sigma0_S_per_m"].median() if not df.empty else np.nan,
        "log10_sigma0_median_S_per_m": df["log10_sigma0_S_per_m"].median() if not df.empty else np.nan,
    }


def build_split_summary(df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, group in df.groupby("sample_holdout_split", sort=False):
        rows.append(summarize_subset(group, "sample_holdout", label))
    for label, group in df.groupby("paper_holdout_split", sort=False):
        rows.append(summarize_subset(group, "paper_holdout", label))
    for fold in range(n_folds):
        rows.append(summarize_subset(df[df["sample_cv_fold"].eq(fold)], "sample_cv_fold", fold))
    for fold in range(n_folds):
        rows.append(summarize_subset(df[df["paper_cv_fold"].eq(fold)], "paper_cv_fold", fold))
    return pd.DataFrame(rows)


def subset_mask(df: pd.DataFrame, subset_name: str) -> pd.Series:
    if subset_name == "all_valid":
        return df["eval_eligible_all_valid"]
    if subset_name == "conservative_valid":
        return df["eval_eligible_conservative_valid"]
    raise ValueError(f"unknown subset: {subset_name}")


def key_columns(group_scheme: str) -> list[str]:
    if group_scheme == "global":
        return ["group_scheme_preflight", "carrier_type", "T_bin_center_K"]
    if group_scheme == "material_family":
        return ["group_scheme_preflight", "material_group_key", "carrier_type", "T_bin_center_K"]
    raise ValueError(f"unknown group_scheme: {group_scheme}")


def add_preflight_group_scheme(df: pd.DataFrame, group_scheme: str) -> pd.DataFrame:
    out = df.copy()
    out["group_scheme_preflight"] = group_scheme
    if group_scheme == "global":
        out["material_group_key_preflight"] = "ALL"
    else:
        out["material_group_key_preflight"] = out["material_group_key"]
    return out


def build_reference_counts(train: pd.DataFrame, group_scheme: str) -> pd.DataFrame:
    train = add_preflight_group_scheme(train, group_scheme)
    if group_scheme == "global":
        train["material_group_key"] = "ALL"
    keys = key_columns(group_scheme)
    counts = (
        train.groupby(keys, dropna=False)
        .agg(
            train_row_count=("row_id", "count"),
            train_sample_count=("validation_sample_group_id", "nunique"),
            train_paper_count=("validation_paper_group_id", "nunique"),
        )
        .reset_index()
    )
    return counts


def build_coverage_preflight(
    df: pd.DataFrame,
    min_rows: int,
    min_samples: int,
    min_papers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    default_uncovered = pd.DataFrame()
    combinations: list[tuple[str, str, str, str, str]] = []
    for split_scheme in ["sample_holdout", "paper_holdout"]:
        for reference_subset in ["all_valid", "conservative_valid"]:
            for eval_subset in ["all_valid", "conservative_valid"]:
                for group_scheme in ["global", "material_family"]:
                    for curve_method in ["row_median", "sample_median"]:
                        combinations.append((split_scheme, reference_subset, eval_subset, group_scheme, curve_method))
    for split_scheme, reference_subset, eval_subset, group_scheme, curve_method in combinations:
        split_col = "sample_holdout_split" if split_scheme == "sample_holdout" else "paper_holdout_split"
        train = df[df[split_col].eq("train") & subset_mask(df, reference_subset)].copy()
        test = df[df[split_col].eq("test") & subset_mask(df, eval_subset)].copy()
        ref_counts = build_reference_counts(train, group_scheme)
        ref_counts["is_reliable_key"] = (
            (ref_counts["train_row_count"] >= min_rows)
            & (ref_counts["train_sample_count"] >= min_samples)
            & (ref_counts["train_paper_count"] >= min_papers)
        )
        test_keyed = add_preflight_group_scheme(test, group_scheme)
        if group_scheme == "global":
            test_keyed["material_group_key"] = "ALL"
        keys = key_columns(group_scheme)
        merged = test_keyed.merge(ref_counts, on=keys, how="left")
        has_reference = merged["is_reliable_key"].eq(True)
        row = {
            "split_scheme": split_scheme,
            "reference_source_subset": reference_subset,
            "eval_target_subset": eval_subset,
            "group_scheme": group_scheme,
            "curve_method": curve_method,
            "min_rows_per_bin": min_rows,
            "min_samples_per_bin": min_samples,
            "min_papers_per_bin": min_papers,
            "train_rows": len(train),
            "train_samples": train["validation_sample_group_id"].nunique(),
            "train_papers": train["validation_paper_group_id"].nunique(),
            "test_rows": len(test),
            "test_samples": test["validation_sample_group_id"].nunique(),
            "test_papers": test["validation_paper_group_id"].nunique(),
            "train_reference_keys_total": len(ref_counts),
            "train_reference_keys_reliable": int(ref_counts["is_reliable_key"].sum()),
            "test_rows_with_reference": int(has_reference.sum()),
            "test_rows_without_reference": int((~has_reference).sum()),
            "coverage_fraction": float(has_reference.mean()) if len(test) else np.nan,
            "p_test_rows": int(test["carrier_type"].eq("p").sum()),
            "n_test_rows": int(test["carrier_type"].eq("n").sum()),
            "p_test_rows_with_reference": int((has_reference & merged["carrier_type"].eq("p")).sum()) if len(merged) else 0,
            "n_test_rows_with_reference": int((has_reference & merged["carrier_type"].eq("n")).sum()) if len(merged) else 0,
            "material_family_count_in_test": test["material_group_key"].nunique(),
            "T_bin_count_in_test": test["T_bin_center_K"].nunique(),
        }
        rows.append(row)
        if (
            split_scheme == "sample_holdout"
            and reference_subset == "conservative_valid"
            and eval_subset == "all_valid"
            and group_scheme == "material_family"
            and curve_method == "sample_median"
        ):
            missing = merged[~has_reference].copy()
            if not missing.empty:
                missing["missing_reference_reason"] = np.where(
                    missing["train_row_count"].isna(),
                    "no_train_reference_key",
                    "insufficient_train_reference_key",
                )
                default_uncovered = missing.head(1000)
    return pd.DataFrame(rows), default_uncovered


def output_name(base: str, suffix: str, ext: str) -> str:
    return f"{base}{suffix}.{ext}"


def save_parquet(df: pd.DataFrame, path: Path) -> tuple[bool, str]:
    try:
        df.to_parquet(path, index=False)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def write_outputs(
    output_dir: Path,
    suffix: str,
    rows: pd.DataFrame,
    sample_assignments: pd.DataFrame,
    paper_assignments: pd.DataFrame,
    split_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    uncovered: pd.DataFrame,
    dropped: pd.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    rows.to_csv(output_dir / output_name("step5a_validation_rows_with_splits", suffix, "csv"), index=False, encoding="utf-8-sig")
    ok, error = save_parquet(rows, output_dir / output_name("step5a_validation_rows_with_splits", suffix, "parquet"))
    statuses[output_name("step5a_validation_rows_with_splits", suffix, "parquet")] = "saved" if ok else f"not saved: {error}"
    sample_assignments.to_csv(output_dir / output_name("step5a_sample_group_split_assignments", suffix, "csv"), index=False, encoding="utf-8-sig")
    paper_assignments.to_csv(output_dir / output_name("step5a_paper_group_split_assignments", suffix, "csv"), index=False, encoding="utf-8-sig")
    split_summary.to_csv(output_dir / output_name("step5a_split_summary", suffix, "csv"), index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / output_name("step5a_holdout_coverage_preflight", suffix, "csv"), index=False, encoding="utf-8-sig")
    uncovered_cols = [
        "row_id",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "material_group_key",
        "carrier_type",
        "T_bin_center_K",
        "T_K",
        "S_uV_per_K",
        "eta",
        "sigma_S_per_m",
        "sigma0_S_per_m",
        "paper_id",
        "sample_id",
        "sample_key",
        "formula_raw",
        "material_name_raw",
        "material_family_raw",
        "missing_reference_reason",
    ]
    if uncovered.empty:
        uncovered = pd.DataFrame(columns=uncovered_cols)
    else:
        for col in uncovered_cols:
            if col not in uncovered.columns:
                uncovered[col] = ""
        uncovered = uncovered[uncovered_cols]
    uncovered.to_csv(output_dir / output_name("step5a_uncovered_test_rows_default_examples", suffix, "csv"), index=False, encoding="utf-8-sig")
    dropped.to_csv(output_dir / output_name("step5a_dropped_rows", suffix, "csv"), index=False, encoding="utf-8-sig")
    return statuses


def build_dropped_output(dropped: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "row_id",
        "reject_reason",
        "T_K",
        "carrier_type",
        "sigma_S_per_m",
        "F0_eta",
        "sigma0_S_per_m",
        "log10_sigma0_S_per_m",
        "is_valid_sigma0",
        "is_conservative_valid_sigma0",
        "paper_id",
        "sample_id",
        "sample_key",
        "material_family_raw",
    ]
    if dropped.empty:
        return pd.DataFrame(columns=cols)
    for col in cols:
        if col not in dropped.columns:
            dropped[col] = ""
    return dropped[cols]


def run_sanity_checks(input_rows: int, rows: pd.DataFrame, dropped: pd.DataFrame, coverage: pd.DataFrame, n_folds: int, full_run: bool) -> tuple[dict[str, bool], list[str], list[str]]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    checks["input_rows_equal_used_plus_dropped"] = input_rows == len(rows) + len(dropped)
    checks["row_id_unique"] = rows["row_id"].is_unique
    checks["validation_sample_group_id_not_missing"] = rows["validation_sample_group_id"].map(clean_text).ne("").all()
    checks["validation_paper_group_id_not_missing"] = rows["validation_paper_group_id"].map(clean_text).ne("").all()
    checks["sample_holdout_split_allowed"] = set(rows["sample_holdout_split"].dropna()).issubset({"train", "test"})
    checks["paper_holdout_split_allowed"] = set(rows["paper_holdout_split"].dropna()).issubset({"train", "test"})
    checks["sample_group_no_holdout_leak"] = rows.groupby("validation_sample_group_id")["sample_holdout_split"].nunique().max() == 1
    checks["paper_group_no_holdout_leak"] = rows.groupby("validation_paper_group_id")["paper_holdout_split"].nunique().max() == 1
    checks["sample_cv_fold_range"] = rows["sample_cv_fold"].between(0, n_folds - 1).all()
    checks["paper_cv_fold_range"] = rows["paper_cv_fold"].between(0, n_folds - 1).all()
    checks["sample_group_single_cv_fold"] = rows.groupby("validation_sample_group_id")["sample_cv_fold"].nunique().max() == 1
    checks["paper_group_single_cv_fold"] = rows.groupby("validation_paper_group_id")["paper_cv_fold"].nunique().max() == 1
    checks["T_inside_bins"] = ((rows["T_bin_left_K"] <= rows["T_K"]) & (rows["T_K"] < rows["T_bin_right_K"])).all()
    checks["carrier_type_p_or_n_only"] = set(rows["carrier_type"].dropna()).issubset({"p", "n"})
    checks["positive_finite_values"] = bool(
        np.isfinite(rows["sigma_S_per_m"]).all()
        and (rows["sigma_S_per_m"] > 0).all()
        and np.isfinite(rows["F0_eta"]).all()
        and (rows["F0_eta"] > 0).all()
        and np.isfinite(rows["sigma0_S_per_m"]).all()
        and (rows["sigma0_S_per_m"] > 0).all()
    )
    checks["coverage_fraction_range"] = coverage["coverage_fraction"].dropna().between(0.0, 1.0).all()
    checks["coverage_counts_consistent"] = (
        coverage["test_rows_with_reference"] + coverage["test_rows_without_reference"] == coverage["test_rows"]
    ).all()
    if full_run:
        checks["sample_holdout_train_test_nonzero"] = set(rows["sample_holdout_split"]) == {"train", "test"}
        checks["paper_holdout_train_test_nonzero"] = set(rows["paper_holdout_split"]) == {"train", "test"}
        checks["coverage_preflight_nonempty"] = not coverage.empty
    else:
        if set(rows["sample_holdout_split"]) != {"train", "test"}:
            warnings.append("small test does not contain both sample holdout splits")
        if set(rows["paper_holdout_split"]) != {"train", "test"}:
            warnings.append("small test does not contain both paper holdout splits")
        if coverage.empty:
            warnings.append("small test produced empty coverage preflight")
    material_family_cov = coverage[coverage["group_scheme"].eq("material_family")]
    if not material_family_cov.empty and material_family_cov["coverage_fraction"].fillna(0).median() < 0.5:
        warnings.append("material_family median coverage_fraction is below 0.5")
    failures = [name for name, ok in checks.items() if not ok]
    return checks, failures, warnings


def numeric_summary(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    return f"min={values.min():.6g}, median={values.median():.6g}, max={values.max():.6g}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "n/a"
    text = df.copy()
    text.columns = [str(col) for col in text.columns]
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join("---" for _ in text.columns) + " |"
    body = ["| " + " | ".join(row[col] for col in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, sep, *body])


def write_report(report_path: Path, input_path: Path, input_rows: int, rows: pd.DataFrame, dropped: pd.DataFrame, split_summary: pd.DataFrame, coverage: pd.DataFrame, uncovered: pd.DataFrame, checks: dict[str, bool], warnings: list[str], parquet_statuses: dict[str, str], args: argparse.Namespace, elapsed_sec: float) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    def cov(split: str, group: str) -> Any:
        sub = coverage[
            coverage["split_scheme"].eq(split)
            & coverage["reference_source_subset"].eq("conservative_valid")
            & coverage["eval_target_subset"].eq("all_valid")
            & coverage["group_scheme"].eq(group)
            & coverage["curve_method"].eq("sample_median")
        ]
        return "n/a" if sub.empty else float(sub["coverage_fraction"].iloc[0])

    lines = [
        "# Step5A Validation Split Report",
        "",
        "## Summary",
        "",
        f"- input_file: {input_path}",
        f"- input_rows: {input_rows}",
        f"- validation rows used: {len(rows)}",
        f"- dropped rows: {len(dropped)}",
        f"- test_size: {args.test_size}",
        f"- n_folds: {args.n_folds}",
        f"- seed: {args.seed}",
        f"- sample_holdout row counts: {rows['sample_holdout_split'].value_counts().to_dict()}",
        f"- sample_holdout sample counts: {rows.groupby('sample_holdout_split')['validation_sample_group_id'].nunique().to_dict()}",
        f"- paper_holdout row counts: {rows['paper_holdout_split'].value_counts().to_dict()}",
        f"- paper_holdout paper counts: {rows.groupby('paper_holdout_split')['validation_paper_group_id'].nunique().to_dict()}",
        f"- sample_cv_fold row counts: {rows['sample_cv_fold'].value_counts().sort_index().to_dict()}",
        f"- paper_cv_fold row counts: {rows['paper_cv_fold'].value_counts().sort_index().to_dict()}",
        f"- default coverage sample_holdout/material_family/conservative_ref/all_test/sample_median: {cov('sample_holdout', 'material_family')}",
        f"- default coverage sample_holdout/global/conservative_ref/all_test/sample_median: {cov('sample_holdout', 'global')}",
        f"- default coverage paper_holdout/material_family/conservative_ref/all_test/sample_median: {cov('paper_holdout', 'material_family')}",
        f"- default coverage paper_holdout/global/conservative_ref/all_test/sample_median: {cov('paper_holdout', 'global')}",
        f"- uncovered default example rows: {len(uncovered)}",
        f"- elapsed_seconds: {elapsed_sec:.2f}",
        "",
        "## Parquet Status",
        "",
    ]
    for name, status in parquet_statuses.items():
        lines.append(f"- {name}: {status}")
    if not dropped.empty:
        lines.extend(["", "## Dropped Reasons", "", str(dropped["reject_reason"].value_counts().to_dict())])
    lines.extend(
        [
            "",
            "## Split Summary",
            "",
            dataframe_to_markdown(split_summary),
            "",
            "## Coverage Preflight",
            "",
            dataframe_to_markdown(coverage),
            "",
            "## Sanity Check",
            "",
        ]
    )
    for name, ok in checks.items():
        lines.append(f"- {name}: {ok}")
    lines.extend(["", "## Warnings And Step5B Notes", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
    else:
        lines.append("- WARNING: none")
    lines.append("- Step5B should build sigma0_ref(T) using train rows only.")
    lines.append("- Step5B should apply sigma_pred = sigma0_ref(T) * F0_eta to test rows only.")
    lines.append("- Evaluate errors with log10(sigma_pred / sigma_exp).")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    input_path = resolve_input(args.input)
    report_path = args.report or (REPORT_DIR / output_name("step5a_validation_split_report", args.output_suffix, "md"))
    full_run = args.max_rows is None

    log("loading step4 binned sigma0 rows...")
    df = read_table(input_path)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        df = df.head(args.max_rows).copy()
    input_rows = len(df)
    log(f"input rows: {input_rows}")
    log("validating required columns...")
    validate_columns(df)
    log("filtering validation candidates...")
    usable, dropped_raw = filter_validation_candidates(df)
    log("normalizing sample, paper, and material group ids...")
    log("assigning sample_holdout split...")
    log("assigning paper_holdout split...")
    log("assigning sample and paper CV folds...")
    rows = assign_ids_and_splits(usable, args.test_size, args.n_folds, args.seed)
    log("building split summaries...")
    sample_assignments = build_group_assignments(
        rows, "validation_sample_group_id", "sample_holdout_split", "sample_cv_fold", "sample"
    )
    paper_assignments = build_group_assignments(
        rows, "validation_paper_group_id", "paper_holdout_split", "paper_cv_fold", "paper"
    )
    split_summary = build_split_summary(rows, args.n_folds)
    log("running coverage preflight...")
    coverage, uncovered = build_coverage_preflight(
        rows, args.min_rows_per_bin, args.min_samples_per_bin, args.min_papers_per_bin
    )
    dropped = build_dropped_output(dropped_raw)
    log("running sanity checks...")
    checks, failures, warnings = run_sanity_checks(input_rows, rows, dropped, coverage, args.n_folds, full_run)
    if failures:
        for failure in failures:
            print(f"[step5a] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    log("writing outputs...")
    parquet_statuses = write_outputs(
        args.output,
        args.output_suffix,
        rows,
        sample_assignments,
        paper_assignments,
        split_summary,
        coverage,
        uncovered,
        dropped,
    )
    write_report(
        report_path,
        input_path,
        input_rows,
        rows,
        dropped,
        split_summary,
        coverage,
        uncovered,
        checks,
        warnings,
        parquet_statuses,
        args,
        time.time() - started,
    )
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
