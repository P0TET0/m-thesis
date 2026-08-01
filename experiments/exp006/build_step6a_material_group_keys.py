import argparse
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_DIR = EXP_DIR / "reports"

DEFAULT_INPUT_PARQUET = PROCESSED_DIR / "step5a_validation_rows_with_splits.parquet"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "step5a_validation_rows_with_splits.csv"
DEFAULT_STEP3_PARQUET = PROCESSED_DIR / "step3_sigma0_valid.parquet"
DEFAULT_STEP3_CSV = PROCESSED_DIR / "step3_sigma0_valid.csv"
DEFAULT_STEP0_PARQUET = Path("data/processed/step0_te_analysis_base.parquet")
DEFAULT_STEP0_CSV = Path("data/processed/step0_te_analysis_base.csv")

CORE_COLUMNS = [
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "material_group_key",
    "T_bin_center_K",
    "carrier_type",
    "is_valid_sigma0",
    "is_conservative_valid_sigma0",
    "sample_holdout_split",
    "paper_holdout_split",
]

REQUIRED_COLUMNS = [
    "row_id",
    "paper_id",
    "doi",
    "sample_id",
    "sample_key",
    "sample_group_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "material_group_key",
    "T_K",
    "T_bin_center_K",
    "T_bin_label",
    "carrier_type",
    "sigma_S_per_m",
    "eta",
    "F0_eta",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "is_valid_sigma0",
    "is_conservative_valid_sigma0",
    "sample_holdout_split",
    "paper_holdout_split",
    "sample_cv_fold",
    "paper_cv_fold",
]

TEXT_SOURCE_COLUMNS = ["formula_raw", "material_name_raw", "material_family_raw", "sample_key", "sample_label", "source_notes"]

CANDIDATE_COLUMNS = [
    "material_group_key_existing_clean",
    "material_group_key_formula_system",
    "material_group_key_broad_family",
    "material_group_key_hybrid_v1",
    "material_group_key_hybrid_v2_broad_first",
    "material_group_key_formula_system_collapsed",
    "material_group_key_hybrid_v1_collapsed",
]

VARIANTS = {
    "formula_system": "material_group_key_formula_system",
    "broad_family": "material_group_key_broad_family",
    "hybrid_v1": "material_group_key_hybrid_v1",
    "hybrid_v2_broad_first": "material_group_key_hybrid_v2_broad_first",
    "formula_system_collapsed": "material_group_key_formula_system_collapsed",
    "hybrid_v1_collapsed": "material_group_key_hybrid_v1_collapsed",
}

ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
}
AMBIGUOUS_SINGLE_ELEMENTS = {"K", "W", "V", "S", "C"}
UNKNOWN_TOKENS = {"", "nan", "none", "null", "unknown", "unknown_material_family", "other", "others", "na", "n/a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step6A material group key candidates.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--step3", type=Path, default=None)
    parser.add_argument("--step0", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_DIR / "step6a_material_group_key_rebuild_report.md")
    parser.add_argument("--min-rows-per-material-group", type=int, default=30)
    parser.add_argument("--min-samples-per-material-group", type=int, default=3)
    parser.add_argument("--min-rows-per-bin", type=int, default=3)
    parser.add_argument("--min-samples-per-bin", type=int, default=3)
    parser.add_argument("--min-papers-per-bin", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step6a] {message}", flush=True)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def resolve_existing(explicit: Path | None, parquet_path: Path, csv_path: Path) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in UNKNOWN_TOKENS:
        return ""
    return text


def clean_family(value: Any) -> str:
    text = clean_text(value)
    return text if text else "unknown_material_group"


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def validate_columns(df: pd.DataFrame) -> None:
    missing_core = sorted(set(CORE_COLUMNS) - set(df.columns))
    if missing_core:
        raise ValueError(f"input missing required analysis columns: {missing_core}")
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""


def merge_optional_metadata(df: pd.DataFrame, path: Path | None, label: str) -> tuple[pd.DataFrame, bool]:
    if path is None or not path.exists():
        return df, False
    meta = read_table(path)
    if "row_id" not in meta.columns:
        return df, False
    useful_cols = [c for c in ["row_id", "formula_raw", "material_name_raw", "material_family_raw", "sample_key", "sample_label", "source_notes"] if c in meta.columns]
    meta = meta[useful_cols].drop_duplicates("row_id")
    out = df.merge(meta, on="row_id", how="left", suffixes=("", f"_{label}"))
    for base in ["formula_raw", "material_name_raw", "material_family_raw", "sample_key", "sample_label", "source_notes"]:
        aux = f"{base}_{label}"
        if aux in out.columns:
            if base not in out.columns:
                out[base] = ""
            out[base] = out[base].where(out[base].map(clean_text).ne(""), out[aux])
            out = out.drop(columns=[aux])
    return out, True


def build_combined_text(row: pd.Series) -> tuple[str, str]:
    values: list[str] = []
    cols: list[str] = []
    for col in TEXT_SOURCE_COLUMNS:
        if col in row.index:
            value = clean_text(row.get(col))
            if value and value not in values:
                values.append(value)
                cols.append(col)
    return " | ".join(values), ";".join(cols)


def extract_elements(text: str) -> list[str]:
    found = re.findall(r"[A-Z][a-z]?", text or "")
    return sorted(set(token for token in found if token in ELEMENTS))


def parse_formula_sources(row: pd.Series) -> tuple[str, int, str, str, str, str]:
    sources = [
        ("formula_raw", clean_text(row.get("formula_raw"))),
        ("material_name_raw", clean_text(row.get("material_name_raw"))),
        ("sample_key", clean_text(row.get("sample_key"))),
        ("combined_text", clean_text(row.get("material_text_combined"))),
    ]
    for source, text in sources:
        if not text:
            continue
        elements = extract_elements(text)
        if not elements:
            continue
        status = "ok"
        if len(elements) == 1 and elements[0] in AMBIGUOUS_SINGLE_ELEMENTS:
            status = "low_confidence"
        if source == "combined_text" and len(elements) == 1:
            status = "low_confidence"
        system_key = "-".join(elements)
        return ";".join(elements), len(elements), status, source, system_key, f"system::{system_key}"
    return "", 0, "failed", "none", "unknown_formula_system", "unknown_material_group"


def broad_family(elements_text: str, combined_text: str, formula_system_key: str) -> tuple[str, str]:
    text = (combined_text or "").casefold()
    normalized = re.sub(r"[\s_-]+", " ", text)
    if "half heusler" in normalized or re.search(r"\bHH\b", combined_text or ""):
        return "broad::half_heusler", "keyword_half_heusler"
    for keyword in ["skutterudite", "clathrate", "zintl", "tetradymite"]:
        if keyword in normalized:
            return f"broad::{keyword}", f"keyword_{keyword}"
    elements = set(elements_text.split(";")) if elements_text else set()
    def has(*items: str) -> bool:
        return set(items).issubset(elements)
    if has("Bi", "Sb", "Te"):
        return "broad::BiSbTe_tetradymite_like", "elements_Bi_Sb_Te"
    if has("Bi", "Te") and "Sb" not in elements:
        return "broad::BiTe_like", "elements_Bi_Te"
    if has("Sb", "Te") and "Bi" not in elements:
        return "broad::SbTe_like", "elements_Sb_Te"
    if has("Pb", "Te"):
        return "broad::PbTe_like", "elements_Pb_Te"
    if has("Sn", "Te"):
        return "broad::SnTe_like", "elements_Sn_Te"
    if has("Ge", "Te"):
        return "broad::GeTe_like", "elements_Ge_Te"
    if has("Si", "Ge"):
        return "broad::SiGe_like", "elements_Si_Ge"
    if "Mg" in elements and ("Si" in elements or "Sn" in elements):
        return "broad::Mg2SiSn_like", "elements_Mg_Si_or_Sn"
    if has("Co", "Sb"):
        return "broad::CoSb_skutterudite_like", "elements_Co_Sb"
    if "O" in elements:
        return "broad::oxide", "contains_O"
    if "Te" in elements:
        return "broad::telluride", "contains_Te"
    if "Se" in elements:
        return "broad::selenide", "contains_Se"
    if "S" in elements:
        return "broad::sulfide", "contains_S"
    if formula_system_key != "unknown_formula_system":
        return "broad::other_formula_system", "formula_system_available"
    return "unknown_material_group", "no_formula_or_keyword"


def build_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["material_group_key_original"] = out["material_group_key"].map(lambda v: clean_text(v) or "unknown_material_group")
    out["material_family_raw_clean"] = out["material_family_raw"].map(clean_family)
    combined = out.apply(build_combined_text, axis=1, result_type="expand")
    out["material_text_combined"] = combined[0]
    out["material_text_source_columns_used"] = combined[1]
    parsed = out.apply(parse_formula_sources, axis=1, result_type="expand")
    out["parsed_elements"] = parsed[0]
    out["parsed_element_count"] = parsed[1].astype(int)
    out["formula_parse_status"] = parsed[2]
    out["formula_parse_source"] = parsed[3]
    out["formula_system_key"] = parsed[4]
    out["formula_system_key_prefixed"] = parsed[5]
    broad = out.apply(lambda row: broad_family(row["parsed_elements"], row["material_text_combined"], row["formula_system_key"]), axis=1, result_type="expand")
    out["material_group_key_broad_family"] = broad[0]
    out["material_group_key_broad_family_reason"] = broad[1]
    out["material_group_key_existing_clean"] = np.where(
        out["material_family_raw_clean"].ne("unknown_material_group"),
        "existing::" + out["material_family_raw_clean"].astype(str),
        "unknown_material_group",
    )
    out["material_group_key_formula_system"] = np.where(
        out["formula_parse_status"].isin(["ok", "low_confidence"]) & out["formula_system_key"].ne("unknown_formula_system"),
        "system::" + out["formula_system_key"].astype(str),
        "unknown_material_group",
    )
    out["material_group_key_hybrid_v1"] = out["material_group_key_existing_clean"]
    mask = out["material_group_key_hybrid_v1"].eq("unknown_material_group") & out["material_group_key_formula_system"].ne("unknown_material_group")
    out.loc[mask, "material_group_key_hybrid_v1"] = out.loc[mask, "material_group_key_formula_system"]
    mask = out["material_group_key_hybrid_v1"].eq("unknown_material_group") & out["material_group_key_broad_family"].ne("unknown_material_group")
    out.loc[mask, "material_group_key_hybrid_v1"] = out.loc[mask, "material_group_key_broad_family"]
    out["material_group_key_hybrid_v2_broad_first"] = out["material_group_key_existing_clean"]
    mask = out["material_group_key_hybrid_v2_broad_first"].eq("unknown_material_group") & out["material_group_key_broad_family"].ne("unknown_material_group")
    out.loc[mask, "material_group_key_hybrid_v2_broad_first"] = out.loc[mask, "material_group_key_broad_family"]
    mask = out["material_group_key_hybrid_v2_broad_first"].eq("unknown_material_group") & out["material_group_key_formula_system"].ne("unknown_material_group")
    out.loc[mask, "material_group_key_hybrid_v2_broad_first"] = out.loc[mask, "material_group_key_formula_system"]
    return out


def collapse_rare_groups(df: pd.DataFrame, source_col: str, min_rows: int, min_samples: int) -> pd.Series:
    counts = df.groupby(source_col, dropna=False).agg(
        row_count=("row_id", "count"),
        sample_count=("validation_sample_group_id", "nunique"),
    )
    rare = counts[(counts["row_count"] < min_rows) | (counts["sample_count"] < min_samples)].index
    collapsed = df[source_col].copy()
    mask = df[source_col].isin(rare) & df["material_group_key_broad_family"].ne("unknown_material_group")
    collapsed.loc[mask] = df.loc[mask, "material_group_key_broad_family"]
    return collapsed


def add_collapsed_candidates(df: pd.DataFrame, min_rows: int, min_samples: int) -> pd.DataFrame:
    out = df.copy()
    out["material_group_key_formula_system_collapsed"] = collapse_rare_groups(out, "material_group_key_formula_system", min_rows, min_samples)
    out["material_group_key_hybrid_v1_collapsed"] = collapse_rare_groups(out, "material_group_key_hybrid_v1", min_rows, min_samples)
    return out


def output_path(output_dir: Path, base: str, suffix: str, ext: str) -> Path:
    return output_dir / f"{base}{suffix}.{ext}"


def save_parquet(df: pd.DataFrame, path: Path) -> tuple[bool, str]:
    try:
        df.to_parquet(path, index=False)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def write_variant_files(df: pd.DataFrame, output_dir: Path, suffix: str) -> dict[str, dict[str, str]]:
    statuses: dict[str, dict[str, str]] = {}
    for variant, col in VARIANTS.items():
        frame = df.copy()
        frame["material_group_key"] = frame[col]
        base = f"step6a_validation_rows_with_splits_key_{variant}"
        csv_path = output_path(output_dir, base, suffix, "csv")
        parquet_path = output_path(output_dir, base, suffix, "parquet")
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        ok, err = save_parquet(frame, parquet_path)
        statuses[variant] = {"csv": str(csv_path), "parquet": "saved" if ok else f"not saved: {err}", "column": col}
    return statuses


def subset_mask(df: pd.DataFrame, subset_name: str) -> pd.Series:
    if subset_name == "all_valid":
        return as_bool(df["is_valid_sigma0"])
    if subset_name == "conservative_valid":
        return as_bool(df["is_conservative_valid_sigma0"])
    raise ValueError(subset_name)


def preflight_for_variant(df: pd.DataFrame, variant: str, key_col: str, min_rows: int, min_samples: int, min_papers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = df.copy()
    work["material_group_key_preflight"] = work[key_col]
    for split_scheme, split_col in [("sample_holdout", "sample_holdout_split"), ("paper_holdout", "paper_holdout_split")]:
        for ref_subset in ["conservative_valid", "all_valid"]:
            for eval_subset in ["all_valid", "conservative_valid"]:
                train = work[work[split_col].eq("train") & subset_mask(work, ref_subset)].copy()
                test = work[work[split_col].eq("test") & subset_mask(work, eval_subset)].copy()
                keys = ["material_group_key_preflight", "carrier_type", "T_bin_center_K"]
                ref_counts = train.groupby(keys, dropna=False).agg(
                    train_row_count=("row_id", "count"),
                    train_sample_count=("validation_sample_group_id", "nunique"),
                    train_paper_count=("validation_paper_group_id", "nunique"),
                ).reset_index()
                ref_counts["is_reliable_key"] = (
                    (ref_counts["train_row_count"] >= min_rows)
                    & (ref_counts["train_sample_count"] >= min_samples)
                    & (ref_counts["train_paper_count"] >= min_papers)
                )
                merged = test.merge(ref_counts, on=keys, how="left")
                has_ref = merged["is_reliable_key"].eq(True)
                uncovered = merged[~has_ref]
                common_uncovered = (
                    uncovered["material_group_key_preflight"].value_counts().head(10).to_dict() if not uncovered.empty else {}
                )
                rows.append(
                    {
                        "material_key_variant": variant,
                        "split_scheme": split_scheme,
                        "reference_source_subset": ref_subset,
                        "eval_target_subset": eval_subset,
                        "train_rows": len(train),
                        "train_samples": train["validation_sample_group_id"].nunique(),
                        "train_papers": train["validation_paper_group_id"].nunique(),
                        "test_rows": len(test),
                        "test_samples": test["validation_sample_group_id"].nunique(),
                        "test_papers": test["validation_paper_group_id"].nunique(),
                        "material_group_count_train": train["material_group_key_preflight"].nunique(),
                        "material_group_count_test": test["material_group_key_preflight"].nunique(),
                        "train_reference_keys_total": len(ref_counts),
                        "train_reference_keys_reliable": int(ref_counts["is_reliable_key"].sum()) if not ref_counts.empty else 0,
                        "test_rows_with_reference": int(has_ref.sum()),
                        "test_rows_without_reference": int((~has_ref).sum()),
                        "coverage_fraction": float(has_ref.mean()) if len(test) else np.nan,
                        "p_test_rows": int(test["carrier_type"].eq("p").sum()),
                        "n_test_rows": int(test["carrier_type"].eq("n").sum()),
                        "p_test_rows_with_reference": int((has_ref & merged["carrier_type"].eq("p")).sum()) if len(merged) else 0,
                        "n_test_rows_with_reference": int((has_ref & merged["carrier_type"].eq("n")).sum()) if len(merged) else 0,
                        "T_bin_count_test": test["T_bin_center_K"].nunique(),
                        "material_group_count_uncovered": uncovered["material_group_key_preflight"].nunique() if not uncovered.empty else 0,
                        "most_common_uncovered_groups": str(common_uncovered),
                    }
                )
    return rows


def build_preflight(df: pd.DataFrame, min_rows: int, min_samples: int, min_papers: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, col in VARIANTS.items():
        rows.extend(preflight_for_variant(df, variant, col, min_rows, min_samples, min_papers))
    return pd.DataFrame(rows)


def summarize_key_variant(df: pd.DataFrame, variant: str, col: str) -> dict[str, Any]:
    group = df.groupby(col, dropna=False).agg(
        row_count=("row_id", "count"),
        sample_count=("validation_sample_group_id", "nunique"),
        paper_count=("validation_paper_group_id", "nunique"),
    )
    unknown_mask = df[col].eq("unknown_material_group")
    top = group.sort_values("row_count", ascending=False).head(20)["row_count"].to_dict()
    return {
        "material_key_variant": variant,
        "unique_group_count": int(df[col].nunique(dropna=False)),
        "unknown_row_count": int(unknown_mask.sum()),
        "unknown_row_fraction": float(unknown_mask.mean()),
        "unknown_sample_count": int(df.loc[unknown_mask, "validation_sample_group_id"].nunique()),
        "row_count": len(df),
        "sample_count": df["validation_sample_group_id"].nunique(),
        "paper_count": df["validation_paper_group_id"].nunique(),
        "median_rows_per_group": float(group["row_count"].median()) if not group.empty else np.nan,
        "max_rows_per_group": int(group["row_count"].max()) if not group.empty else 0,
        "median_samples_per_group": float(group["sample_count"].median()) if not group.empty else np.nan,
        "max_samples_per_group": int(group["sample_count"].max()) if not group.empty else 0,
        "top20_groups_by_rows": str(top),
    }


def build_key_summary(df: pd.DataFrame) -> pd.DataFrame:
    variants = {"existing_clean": "material_group_key_existing_clean", **VARIANTS}
    return pd.DataFrame([summarize_key_variant(df, variant, col) for variant, col in variants.items()])


def examples(series: pd.Series, limit: int = 5) -> str:
    vals: list[str] = []
    for value in series:
        text = clean_text(value)
        if text and text not in vals:
            vals.append(text)
        if len(vals) >= limit:
            break
    return " | ".join(vals)


def build_key_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variants = {"existing_clean": "material_group_key_existing_clean", **VARIANTS}
    for variant, col in variants.items():
        for value, group in df.groupby(col, dropna=False, sort=False):
            rows.append(
                {
                    "material_key_variant": variant,
                    "material_group_key_value": value,
                    "row_count": len(group),
                    "sample_count": group["validation_sample_group_id"].nunique(),
                    "paper_count": group["validation_paper_group_id"].nunique(),
                    "carrier_type_values": examples(group["carrier_type"]),
                    "T_min_K": pd.to_numeric(group.get("T_K", pd.Series(dtype=float)), errors="coerce").min(),
                    "T_max_K": pd.to_numeric(group.get("T_K", pd.Series(dtype=float)), errors="coerce").max(),
                    "formula_raw_examples": examples(group["formula_raw"]),
                    "material_name_raw_examples": examples(group["material_name_raw"]),
                    "material_family_raw_examples": examples(group["material_family_raw"]),
                }
            )
    return pd.DataFrame(rows)


def build_failure_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    failure_cols = [
        "row_id", "formula_raw", "material_name_raw", "material_family_raw", "sample_key",
        "material_text_combined", "formula_parse_status", "formula_parse_source", "parsed_elements", "formula_system_key",
    ]
    failures = df[df["formula_parse_status"].isin(["failed", "low_confidence"])].copy().head(5000)
    ambiguous = df[
        df["material_group_key_formula_system"].eq("unknown_material_group")
        | df["material_group_key_broad_family"].eq("unknown_material_group")
        | (df["parsed_element_count"] <= 1)
        | df["material_text_combined"].map(clean_text).eq("")
        | (df["material_group_key_formula_system"] != df["material_group_key_broad_family"])
    ].copy().head(5000)
    for frame in [failures, ambiguous]:
        for col in failure_cols:
            if col not in frame.columns:
                frame[col] = ""
    return failures[failure_cols], ambiguous[failure_cols]


def recommend_variants(summary: pd.DataFrame, preflight: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        s = summary[summary["material_key_variant"].eq(variant)].iloc[0].to_dict()
        sample_cov = preflight[
            preflight["material_key_variant"].eq(variant)
            & preflight["split_scheme"].eq("sample_holdout")
            & preflight["reference_source_subset"].eq("conservative_valid")
            & preflight["eval_target_subset"].eq("all_valid")
        ]["coverage_fraction"]
        paper_cov = preflight[
            preflight["material_key_variant"].eq(variant)
            & preflight["split_scheme"].eq("paper_holdout")
            & preflight["reference_source_subset"].eq("conservative_valid")
            & preflight["eval_target_subset"].eq("all_valid")
        ]["coverage_fraction"]
        sc = float(sample_cov.iloc[0]) if not sample_cov.empty else np.nan
        pc = float(paper_cov.iloc[0]) if not paper_cov.empty else np.nan
        unique = int(s["unique_group_count"])
        unknown = float(s["unknown_row_fraction"])
        score = np.nanmean([sc, pc]) - unknown
        if unique <= 1:
            score -= 2
        if unique > 500:
            score -= 0.25
        rows.append(
            {
                "material_key_variant": variant,
                "reason": "ranked by representative coverage, unknown fraction, and group count",
                "unique_group_count": unique,
                "unknown_row_fraction": unknown,
                "representative_coverage_fraction_sample_holdout": sc,
                "representative_coverage_fraction_paper_holdout": pc,
                "comment": "collapsed variants may improve coverage but reduce chemical specificity" if "collapsed" in variant else "candidate for Step5B rerun",
                "_score": score,
            }
        )
    out = pd.DataFrame(rows).sort_values("_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.drop(columns=["_score"])


def run_sanity(input_rows: int, df: pd.DataFrame, summary: pd.DataFrame, preflight: pd.DataFrame, variant_status: dict[str, dict[str, str]], report_path: Path, output_dir: Path, suffix: str, full_run: bool) -> tuple[dict[str, bool], list[str], list[str]]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    checks["candidate_rows_match_input"] = len(df) == input_rows
    checks["row_id_unique"] = df["row_id"].is_unique
    checks["material_group_key_original_exists"] = "material_group_key_original" in df.columns
    checks["candidate_columns_exist"] = set(CANDIDATE_COLUMNS).issubset(df.columns)
    checks["candidate_columns_not_missing"] = all(df[col].notna().all() for col in CANDIDATE_COLUMNS)
    checks["formula_parse_status_allowed"] = set(df["formula_parse_status"]).issubset({"ok", "low_confidence", "failed"})
    checks["parsed_element_count_nonnegative"] = (df["parsed_element_count"] >= 0).all()
    checks["formula_system_key_not_empty"] = df["formula_system_key"].map(clean_text).ne("").all()
    checks["six_variant_files_created"] = len(variant_status) == 6
    variant_rows_ok = True
    variant_replaced_ok = True
    variant_original_ok = True
    split_ok = True
    for variant, col in VARIANTS.items():
        csv_path = output_path(output_dir, f"step6a_validation_rows_with_splits_key_{variant}", suffix, "csv")
        if not csv_path.exists():
            variant_rows_ok = False
            continue
        sample = pd.read_csv(csv_path, usecols=["row_id", "material_group_key", "material_group_key_original", "sample_holdout_split", "paper_holdout_split"], low_memory=False)
        variant_rows_ok &= len(sample) == input_rows
        variant_replaced_ok &= sample["material_group_key"].equals(df[col].reset_index(drop=True))
        variant_original_ok &= "material_group_key_original" in sample.columns
        split_ok &= sample["sample_holdout_split"].notna().all() and sample["paper_holdout_split"].notna().all()
    checks["variant_file_rows_match_input"] = variant_rows_ok
    checks["variant_material_group_key_replaced"] = variant_replaced_ok
    checks["variant_original_preserved"] = variant_original_ok
    checks["variant_splits_preserved"] = split_ok
    checks["sample_holdout_no_leakage"] = df.groupby("validation_sample_group_id")["sample_holdout_split"].nunique().max() == 1
    checks["paper_holdout_no_leakage"] = df.groupby("validation_paper_group_id")["paper_holdout_split"].nunique().max() == 1
    checks["preflight_coverage_range"] = preflight["coverage_fraction"].dropna().between(0, 1).all()
    checks["preflight_nonempty"] = not preflight.empty
    any_multi = bool((summary[summary["material_key_variant"].isin(VARIANTS.keys())]["unique_group_count"] > 1).any())
    if full_run:
        checks["candidate_unique_group_count_gt_1"] = any_multi
    elif not any_multi:
        warnings.append("small test has no candidate with unique_group_count > 1")
    unknown_not_all = bool((summary[summary["material_key_variant"].isin(["formula_system", "hybrid_v1", "hybrid_v2_broad_first"])]["unknown_row_fraction"] < 1.0).any())
    if full_run:
        checks["formula_or_hybrid_unknown_not_all"] = unknown_not_all
    elif not unknown_not_all:
        warnings.append("small test formula/hybrid unknown fraction is 1.0")
    checks["report_exists"] = report_path.exists() and report_path.stat().st_size > 0
    failures = [name for name, ok in checks.items() if not ok]
    return checks, failures, warnings


def df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "n/a"
    text = df.head(max_rows).copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join("---" for _ in text.columns) + " |"
    body = ["| " + " | ".join(row[col] for col in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, sep, *body])


def write_report(report_path: Path, input_path: Path, input_rows: int, used_step3: bool, used_step0: bool, df: pd.DataFrame, summary: pd.DataFrame, counts: pd.DataFrame, preflight: pd.DataFrame, recommended: pd.DataFrame, checks: dict[str, bool], warnings: list[str], elapsed: float) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing_unknown = df["material_group_key_original"].eq("unknown_material_family") | df["material_group_key_original"].eq("unknown_material_group")
    default_cov = preflight[
        preflight["reference_source_subset"].eq("conservative_valid")
        & preflight["eval_target_subset"].eq("all_valid")
    ][["material_key_variant", "split_scheme", "coverage_fraction", "material_group_count_train", "material_group_count_test"]]
    lines = [
        "# Step6A Material Group Key Rebuild Report",
        "",
        "## Summary",
        "",
        f"- input_file: {input_path}",
        f"- input_rows: {input_rows}",
        f"- used_step3_metadata: {used_step3}",
        f"- used_step0_metadata: {used_step0}",
        f"- existing material_group_key unique count: {df['material_group_key_original'].nunique()}",
        f"- existing material_group_key unknown fraction: {float(existing_unknown.mean())}",
        f"- formula_raw missing fraction: {float(df['formula_raw'].map(clean_text).eq('').mean())}",
        f"- material_name_raw missing fraction: {float(df['material_name_raw'].map(clean_text).eq('').mean())}",
        f"- material_family_raw missing fraction: {float(df['material_family_raw'].map(clean_text).eq('').mean())}",
        f"- formula_parse_status counts: {df['formula_parse_status'].value_counts().to_dict()}",
        f"- elapsed_seconds: {elapsed:.2f}",
        "",
        "## Candidate Key Summary",
        "",
        df_to_markdown(summary),
        "",
        "## Top Groups",
        "",
        df_to_markdown(counts.sort_values(["material_key_variant", "row_count"], ascending=[True, False]).head(50)),
        "",
        "## Preflight Coverage Default-like Settings",
        "",
        df_to_markdown(default_cov),
        "",
        "## Recommended Variants",
        "",
        df_to_markdown(recommended),
        "",
        "## Sanity Check",
        "",
    ]
    for name, ok in checks.items():
        lines.append(f"- {name}: {ok}")
    lines.extend(["", "## Notes", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
    else:
        lines.append("- WARNING: none")
    lines.append("- broad_family is a heuristic grouping for validation only, not a final material taxonomy.")
    lines.append("- formula_system can over-split doped or noisy formula strings and can include regex false positives.")
    lines.append("- collapsed variants improve bin coverage by mapping rare formula systems to broad families, but lose specificity.")
    lines.append("- Next: choose one recommended variant, rerun Step5B with that variant input, then rerun Step5C and Step5D-1.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    input_path = resolve_existing(args.input, DEFAULT_INPUT_PARQUET, DEFAULT_INPUT_CSV)
    if input_path is None:
        raise FileNotFoundError("Step5A validation rows not found")
    step3_path = resolve_existing(args.step3, DEFAULT_STEP3_PARQUET, DEFAULT_STEP3_CSV)
    step0_path = resolve_existing(args.step0, DEFAULT_STEP0_PARQUET, DEFAULT_STEP0_CSV)
    full_run = args.max_rows is None

    log("loading Step5A validation rows...")
    df = read_table(input_path)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        df = df.head(args.max_rows).copy()
    input_rows = len(df)
    log(f"input rows: {input_rows}")
    log("loading optional Step3/Step0 metadata...")
    df, used_step3 = merge_optional_metadata(df, step3_path, "step3")
    df, used_step0 = merge_optional_metadata(df, step0_path, "step0")
    log("validating required columns...")
    validate_columns(df)
    log("preserving original material_group_key...")
    log("building combined material text...")
    log("parsing elements and formula systems...")
    log("assigning broad chemical family keys...")
    log("building material group key candidates...")
    candidates = build_candidates(df)
    log("collapsing rare groups...")
    candidates = add_collapsed_candidates(candidates, args.min_rows_per_material_group, args.min_samples_per_material_group)
    args.output.mkdir(parents=True, exist_ok=True)
    log("writing Step5B-ready variant files...")
    variant_status = write_variant_files(candidates, args.output, args.output_suffix)
    log("running coverage preflight...")
    preflight = build_preflight(candidates, args.min_rows_per_bin, args.min_samples_per_bin, args.min_papers_per_bin)
    log("building summaries...")
    summary = build_key_summary(candidates)
    counts = build_key_counts(candidates)
    failures, ambiguous = build_failure_tables(candidates)
    recommended = recommend_variants(summary, preflight)

    candidates.to_csv(output_path(args.output, "step6a_material_group_candidate_rows", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    save_parquet(candidates, output_path(args.output, "step6a_material_group_candidate_rows", args.output_suffix, "parquet"))
    summary.to_csv(output_path(args.output, "step6a_material_group_key_summary", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    counts.to_csv(output_path(args.output, "step6a_material_group_key_counts", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    failures.to_csv(output_path(args.output, "step6a_formula_parse_failures", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    ambiguous.to_csv(output_path(args.output, "step6a_ambiguous_material_group_examples", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    recommended.to_csv(output_path(args.output, "step6a_recommended_material_key_variants", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    preflight.to_csv(output_path(args.output, "step6a_material_group_key_preflight_coverage", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")

    log("writing report...")
    write_report(args.report, input_path, input_rows, used_step3, used_step0, candidates, summary, counts, preflight, recommended, {}, [], time.time() - started)
    log("running sanity checks...")
    checks, check_failures, warnings = run_sanity(input_rows, candidates, summary, preflight, variant_status, args.report, args.output, args.output_suffix, full_run)
    if check_failures:
        for failure in check_failures:
            print(f"[step6a] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    write_report(args.report, input_path, input_rows, used_step3, used_step0, candidates, summary, counts, preflight, recommended, checks, warnings, time.time() - started)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
