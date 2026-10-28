"""Build the exp007 Phase 3 inventory of the current broad-oxide cohort.

Phase 3 is an audit, not the final oxide taxonomy.  It preserves the Phase 1
held-out prediction cohort, joins the exact parsing metadata that produced the
current ``broad::oxide`` label, and creates row-, sample-, and label-level
tables.  A deliberately provisional composition split is included to expose
the sizes, residual patterns, and review burden of plausible next-step groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
PHASE1_ROWS = EXP007 / "data" / "processed" / "phase1_sigma_residual_rows.csv"
PHASE1_SAMPLES = EXP007 / "data" / "processed" / "phase1_sigma_sample_summary.csv"
STEP6A_CANDIDATES = (
    ROOT
    / "experiments"
    / "exp006"
    / "data"
    / "processed"
    / "step6a_material_group_candidate_rows.csv"
)
STEP6A_SCRIPT = ROOT / "experiments" / "exp006" / "build_step6a_material_group_keys.py"
PHASE0_DECISION = EXP007 / "data" / "processed" / "phase0_baseline_decision.csv"

DEFAULT_OUTPUT = EXP007 / "data" / "processed"
DEFAULT_REPORT = EXP007 / "reports" / "phase3_oxide_inventory.md"
DEFAULT_VALIDATION = EXP007 / "reports" / "phase3_validation_log.txt"
DEFAULT_PAYLOAD = EXP007 / "reports" / ".phase3_workbook_payload.json"

ATOL = 1e-12
FACTOR_2_LOG10 = math.log10(2.0)
FACTOR_5_LOG10 = math.log10(5.0)
FACTOR_10_LOG10 = 1.0

PARSE_COLUMNS = [
    "row_id",
    "parsed_elements",
    "parsed_element_count",
    "formula_parse_status",
    "formula_parse_source",
    "formula_system_key",
    "material_text_combined",
    "material_text_source_columns_used",
    "material_group_key_broad_family",
    "material_group_key_broad_family_reason",
]

TRANSITION_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}
CHALCOGENS = {"S", "Se", "Te"}
OTHER_ANION_MARKERS = {"N", "P", "As", "F", "Cl", "Br", "I"}
ORGANIC_KEYWORDS = (
    "poly(", "polymer", "polyaniline", "pedot", "polystyrene",
    "thiophene", "organic", "pss",
)
PRIORITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validation-log", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--workbook-payload", type=Path, default=DEFAULT_PAYLOAD)
    return parser.parse_args()


def require_inputs() -> None:
    missing = [
        path
        for path in (
            PHASE1_ROWS,
            PHASE1_SAMPLES,
            STEP6A_CANDIDATES,
            STEP6A_SCRIPT,
            PHASE0_DECISION,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Phase 3 inputs: {missing}")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def first_nonempty(series: pd.Series) -> Any:
    for value in series.dropna():
        text = str(value).strip()
        if text:
            return value
    return None


def join_unique(series: Iterable[Any], limit: int = 12) -> str:
    values: list[str] = []
    for value in series:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    if len(values) <= limit:
        return " | ".join(values)
    return " | ".join(values[:limit]) + f" | ... (+{len(values) - limit})"


def normalize_material_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.translate(
        str.maketrans(
            {
                "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
                "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
                "−": "-", "–": "-", "—": "-", "·": ".", "∙": ".",
            }
        )
    )
    return re.sub(r"\s+", " ", text).strip()


def element_set(value: Any) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {token for token in str(value).split(";") if token}


def looks_organic(label: str, elements: set[str]) -> bool:
    lowered = label.casefold()
    if any(keyword in lowered for keyword in ORGANIC_KEYWORDS):
        return True
    return bool(
        "C" in elements
        and "H" in elements
        and re.search(r"(?:^|\()C(?:\d+(?:\.\d+)?)?H(?:\d+(?:\.\d+)?)?", label)
    )


def has_multicomponent_notation(label: str) -> bool:
    lowered = label.casefold()
    explicit = any(
        token in lowered
        for token in (" composite", "mixture", "wt%", "vol%", "doped ", ":")
    )
    adjacent_formulae = bool(re.search(r"\)\s*\d*(?:\.\d+)?\s*\(", label))
    return explicit or adjacent_formulae or "+" in label


def has_free_text(label: str) -> bool:
    if len(label) > 60:
        return True
    words = re.findall(r"[A-Za-z]{4,}", label)
    return any(word.casefold() not in {"oxide"} for word in words)


def candidate_family(elements: set[str], organic: bool) -> tuple[str, str]:
    """Return a provisional, composition-only family and its explicit rule."""
    if organic:
        return (
            "review::organic_or_polymer_O-containing",
            "organic/polymer keyword or C-H-O molecular notation",
        )
    if elements & CHALCOGENS:
        return (
            "mixed_anion::oxychalcogenide",
            "contains O and at least one of S, Se, or Te",
        )
    if elements & {"As", "P"}:
        return (
            "mixed_anion::oxypnictide_candidate",
            "contains O and As or P; manual chemistry review required",
        )
    if {"Co", "Mn"}.issubset(elements):
        return "oxide::Co-Mn_mixed", "contains both Co and Mn"
    if "Co" in elements:
        return "oxide::cobalt_based", "contains Co and no Mn"
    if "Mn" in elements:
        return "oxide::manganese_based", "contains Mn and no Co"
    if "Zn" in elements:
        return "oxide::zinc_based", "contains Zn after higher-priority rules"
    if "Ti" in elements:
        return "oxide::titanium_based", "contains Ti after higher-priority rules"
    if "Cu" in elements:
        return "oxide::copper_based", "contains Cu after higher-priority rules"
    if elements & {"In", "Sn", "Cd"}:
        return (
            "oxide::In-Sn-Cd_based",
            "contains In, Sn, or Cd after higher-priority rules",
        )
    if elements & {"Mo", "W", "V", "Nb"}:
        return (
            "oxide::Mo-W-V-Nb_based",
            "contains Mo, W, V, or Nb after higher-priority rules",
        )
    if elements & {"Fe", "Ni", "Cr"}:
        return (
            "oxide::Fe-Ni-Cr_based",
            "contains Fe, Ni, or Cr after higher-priority rules",
        )
    return (
        "oxide::other_inorganic_O-containing",
        "contains O but no earlier provisional composition rule matched",
    )


def classify_review(
    organic: bool,
    chalcogen: bool,
    other_anion: bool,
    multicomponent: bool,
    free_text: bool,
    formula_name_match: bool,
) -> tuple[str, str]:
    reasons: list[str] = []
    priority = "none"
    if organic:
        reasons.append("likely organic/polymer rather than inorganic oxide")
        priority = "high"
    if chalcogen:
        reasons.append("mixed-anion O+S/Se/Te system")
        priority = max(priority, "medium", key=PRIORITY_ORDER.get)
    if other_anion:
        reasons.append("contains N/P/As/halogen in addition to O")
        priority = max(priority, "medium", key=PRIORITY_ORDER.get)
    if free_text:
        reasons.append("free-text or long material label")
        priority = max(priority, "medium", key=PRIORITY_ORDER.get)
    if not formula_name_match:
        reasons.append("formula and material-name labels differ")
        priority = max(priority, "medium", key=PRIORITY_ORDER.get)
    if multicomponent:
        reasons.append("multi-component notation")
        priority = max(priority, "low", key=PRIORITY_ORDER.get)
    return priority, "; ".join(reasons)


def load_parsing_metadata(row_ids: set[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        STEP6A_CANDIDATES,
        usecols=PARSE_COLUMNS,
        chunksize=100_000,
        low_memory=False,
    ):
        matched = chunk.loc[chunk["row_id"].astype(str).isin(row_ids)]
        if not matched.empty:
            pieces.append(matched)
    if not pieces:
        return pd.DataFrame(columns=PARSE_COLUMNS)
    return pd.concat(pieces, ignore_index=True).drop_duplicates("row_id")


def numeric_summary(residuals: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(residuals, errors="coerce").dropna().to_numpy(float)
    if not values.size:
        return {
            "mean_residual_log10": np.nan,
            "median_residual_log10": np.nan,
            "mae_log10": np.nan,
            "rmse_log10": np.nan,
            "max_abs_residual_log10": np.nan,
            "overprediction_fraction": np.nan,
            "underprediction_fraction": np.nan,
            "within_factor_2_fraction": np.nan,
            "within_factor_5_fraction": np.nan,
            "within_factor_10_fraction": np.nan,
            "beyond_factor_10_count": 0,
        }
    absolute = np.abs(values)
    return {
        "mean_residual_log10": float(np.mean(values)),
        "median_residual_log10": float(np.median(values)),
        "mae_log10": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(values**2))),
        "max_abs_residual_log10": float(np.max(absolute)),
        "overprediction_fraction": float(np.mean(values > ATOL)),
        "underprediction_fraction": float(np.mean(values < -ATOL)),
        "within_factor_2_fraction": float(np.mean(absolute <= FACTOR_2_LOG10 + ATOL)),
        "within_factor_5_fraction": float(np.mean(absolute <= FACTOR_5_LOG10 + ATOL)),
        "within_factor_10_fraction": float(np.mean(absolute <= FACTOR_10_LOG10 + ATOL)),
        "beyond_factor_10_count": int(np.sum(absolute > FACTOR_10_LOG10 + ATOL)),
    }


def build_row_inventory(phase1: pd.DataFrame, parsing: pd.DataFrame) -> pd.DataFrame:
    oxide = phase1.loc[as_bool(phase1["is_oxide_broad"])].copy()
    oxide["row_id"] = oxide["row_id"].astype(str)
    rows = oxide.merge(parsing, on="row_id", how="left", validate="one_to_one")

    records: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        formula = normalize_material_label(row.formula_raw)
        material_name = normalize_material_label(row.material_name_raw)
        elements = element_set(row.parsed_elements)
        label = formula or material_name
        organic = looks_organic(label, elements)
        chalcogen = bool(elements & CHALCOGENS)
        other_anion = bool(elements & OTHER_ANION_MARKERS)
        multicomponent = has_multicomponent_notation(label)
        free_text = has_free_text(label)
        formula_name_match = formula.casefold() == material_name.casefold()
        family, family_rule = candidate_family(elements, organic)
        priority, review_reasons = classify_review(
            organic,
            chalcogen,
            other_anion,
            multicomponent,
            free_text,
            formula_name_match,
        )
        records.append(
            {
                "row_id": row.row_id,
                "paper_id": row.paper_id,
                "doi": row.doi,
                "sample_id": row.sample_id,
                "sample_key": row.sample_key,
                "validation_sample_group_id": row.validation_sample_group_id,
                "validation_paper_group_id": row.validation_paper_group_id,
                "formula_raw": row.formula_raw,
                "material_name_raw": row.material_name_raw,
                "material_label_normalized": label,
                "formula_name_exact_match": formula_name_match,
                "parsed_elements": row.parsed_elements,
                "parsed_element_count": row.parsed_element_count,
                "formula_parse_status": row.formula_parse_status,
                "formula_parse_source": row.formula_parse_source,
                "formula_system_key": row.formula_system_key,
                "current_broad_family": row.material_group_key_broad_family,
                "current_broad_family_reason": row.material_group_key_broad_family_reason,
                "candidate_oxide_family": family,
                "candidate_family_rule": family_rule,
                "transition_metal_elements": ";".join(sorted(elements & TRANSITION_ELEMENTS)),
                "n_transition_metal_elements": len(elements & TRANSITION_ELEMENTS),
                "has_chalcogen_anion_with_O": chalcogen,
                "has_other_anion_marker_with_O": other_anion,
                "has_multicomponent_notation": multicomponent,
                "has_free_text_label": free_text,
                "likely_not_inorganic_oxide": organic,
                "manual_review_required": priority != "none",
                "review_priority": priority,
                "review_reasons": review_reasons,
                "carrier_type": row.carrier_type,
                "T_K": row.T_K,
                "T_bin_label": row.T_bin_label,
                "S_abs_uV_per_K": row.S_abs_uV_per_K,
                "sigma_observed_S_per_m": row.sigma_observed_S_per_m,
                "sigma_predicted_S_per_m": row.sigma_predicted_S_per_m,
                "residual_log10_pred_over_observed": row.residual_log10_pred_over_observed,
                "abs_residual_log10": row.abs_residual_log10,
                "absolute_error_factor": row.absolute_error_factor,
                "prediction_direction": row.prediction_direction,
                "residual_band": row.residual_band,
                "within_factor_10": row.within_factor_10,
                "beyond_factor_10": row.beyond_factor_10,
            }
        )
    output = pd.DataFrame(records)
    order = output["review_priority"].map(PRIORITY_ORDER)
    output = output.assign(_priority_order=order).sort_values(
        ["_priority_order", "abs_residual_log10", "validation_sample_group_id", "T_K"],
        ascending=[False, False, True, True],
        na_position="last",
    )
    return output.drop(columns="_priority_order").reset_index(drop=True)


def build_sample_inventory(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sample_group, group in rows.groupby("validation_sample_group_id", sort=False):
        families = sorted(group["candidate_oxide_family"].dropna().unique())
        priorities = group["review_priority"].map(PRIORITY_ORDER)
        review_priority = max(PRIORITY_ORDER, key=lambda key: int((priorities == PRIORITY_ORDER[key]).any()) * PRIORITY_ORDER[key])
        if priorities.max() >= 0:
            review_priority = max(
                group["review_priority"].dropna().unique(), key=PRIORITY_ORDER.get
            )
        record: dict[str, Any] = {
            "validation_sample_group_id": sample_group,
            "validation_paper_group_id": first_nonempty(group["validation_paper_group_id"]),
            "paper_id": first_nonempty(group["paper_id"]),
            "doi": first_nonempty(group["doi"]),
            "sample_id": first_nonempty(group["sample_id"]),
            "sample_key": first_nonempty(group["sample_key"]),
            "formula_raw": join_unique(group["formula_raw"]),
            "material_name_raw": join_unique(group["material_name_raw"]),
            "material_labels_normalized": join_unique(group["material_label_normalized"]),
            "n_distinct_material_labels": int(group["material_label_normalized"].nunique()),
            "parsed_elements": join_unique(group["parsed_elements"]),
            "formula_system_keys": join_unique(group["formula_system_key"]),
            "candidate_oxide_family": families[0] if len(families) == 1 else "mixed_candidates",
            "candidate_family_count": len(families),
            "candidate_family_values": " | ".join(families),
            "carrier_types": join_unique(group["carrier_type"]),
            "n_rows": int(len(group)),
            "T_min_K": float(pd.to_numeric(group["T_K"], errors="coerce").min()),
            "T_max_K": float(pd.to_numeric(group["T_K"], errors="coerce").max()),
            "likely_not_inorganic_oxide": bool(group["likely_not_inorganic_oxide"].any()),
            "has_chalcogen_anion_with_O": bool(group["has_chalcogen_anion_with_O"].any()),
            "has_other_anion_marker_with_O": bool(group["has_other_anion_marker_with_O"].any()),
            "has_multicomponent_notation": bool(group["has_multicomponent_notation"].any()),
            "has_free_text_label": bool(group["has_free_text_label"].any()),
            "formula_name_exact_match": bool(group["formula_name_exact_match"].all()),
            "manual_review_required": bool(group["manual_review_required"].any()),
            "review_priority": review_priority,
            "review_reasons": join_unique(group["review_reasons"]),
        }
        record.update(numeric_summary(group["residual_log10_pred_over_observed"]))
        records.append(record)
    output = pd.DataFrame(records)
    output["sample_abs_median_residual_rank"] = (
        output["median_residual_log10"].abs().rank(method="first", ascending=False).astype("Int64")
    )
    output["_priority_order"] = output["review_priority"].map(PRIORITY_ORDER)
    return output.sort_values(
        ["_priority_order", "sample_abs_median_residual_rank", "validation_sample_group_id"],
        ascending=[False, True, True],
    ).drop(columns="_priority_order").reset_index(drop=True)


def build_label_inventory(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for label, group in rows.groupby("material_label_normalized", dropna=False, sort=False):
        priorities = group["review_priority"].dropna().unique()
        record: dict[str, Any] = {
            "material_label_normalized": label,
            "formula_raw_examples": join_unique(group["formula_raw"], limit=5),
            "material_name_examples": join_unique(group["material_name_raw"], limit=5),
            "parsed_elements": join_unique(group["parsed_elements"]),
            "formula_system_keys": join_unique(group["formula_system_key"]),
            "candidate_oxide_family": join_unique(group["candidate_oxide_family"]),
            "candidate_family_rule": join_unique(group["candidate_family_rule"]),
            "n_rows": int(len(group)),
            "n_samples": int(group["validation_sample_group_id"].nunique()),
            "n_papers": int(group["validation_paper_group_id"].nunique()),
            "carrier_types": join_unique(group["carrier_type"]),
            "T_min_K": float(pd.to_numeric(group["T_K"], errors="coerce").min()),
            "T_max_K": float(pd.to_numeric(group["T_K"], errors="coerce").max()),
            "likely_not_inorganic_oxide": bool(group["likely_not_inorganic_oxide"].any()),
            "manual_review_required": bool(group["manual_review_required"].any()),
            "review_priority": max(priorities, key=PRIORITY_ORDER.get) if len(priorities) else "none",
            "review_reasons": join_unique(group["review_reasons"]),
        }
        record.update(numeric_summary(group["residual_log10_pred_over_observed"]))
        records.append(record)
    output = pd.DataFrame(records)
    output["single_sample_label"] = output["n_samples"].eq(1)
    output["_priority_order"] = output["review_priority"].map(PRIORITY_ORDER)
    return output.sort_values(
        ["_priority_order", "n_samples", "n_rows", "material_label_normalized"],
        ascending=[False, False, False, True],
    ).drop(columns="_priority_order").reset_index(drop=True)


def build_family_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    total_rows = len(rows)
    total_samples = rows["validation_sample_group_id"].nunique()
    for family, group in rows.groupby("candidate_oxide_family", sort=False):
        record: dict[str, Any] = {
            "candidate_oxide_family": family,
            "candidate_family_rule": first_nonempty(group["candidate_family_rule"]),
            "n_rows": int(len(group)),
            "row_share": float(len(group) / total_rows),
            "n_samples": int(group["validation_sample_group_id"].nunique()),
            "sample_share": float(group["validation_sample_group_id"].nunique() / total_samples),
            "n_papers": int(group["validation_paper_group_id"].nunique()),
            "n_material_labels": int(group["material_label_normalized"].nunique()),
            "n_manual_review_samples": int(
                group.loc[group["manual_review_required"], "validation_sample_group_id"].nunique()
            ),
            "n_likely_nonoxide_samples": int(
                group.loc[group["likely_not_inorganic_oxide"], "validation_sample_group_id"].nunique()
            ),
            "carrier_types": join_unique(group["carrier_type"]),
            "T_min_K": float(pd.to_numeric(group["T_K"], errors="coerce").min()),
            "T_max_K": float(pd.to_numeric(group["T_K"], errors="coerce").max()),
        }
        record.update(numeric_summary(group["residual_log10_pred_over_observed"]))
        record["overall_eval_size_ge_30rows_3samples_3papers"] = bool(
            record["n_rows"] >= 30 and record["n_samples"] >= 3 and record["n_papers"] >= 3
        )
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["n_samples", "n_rows", "candidate_oxide_family"], ascending=[False, False, True]
    ).reset_index(drop=True)


def build_element_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    elements = sorted({element for value in rows["parsed_elements"] for element in element_set(value)})
    for element in elements:
        mask = rows["parsed_elements"].map(lambda value: element in element_set(value))
        group = rows.loc[mask]
        record: dict[str, Any] = {
            "element": element,
            "n_rows": int(len(group)),
            "n_samples": int(group["validation_sample_group_id"].nunique()),
            "n_papers": int(group["validation_paper_group_id"].nunique()),
            "n_material_labels": int(group["material_label_normalized"].nunique()),
            "sample_share": float(
                group["validation_sample_group_id"].nunique()
                / rows["validation_sample_group_id"].nunique()
            ),
        }
        record.update(numeric_summary(group["residual_log10_pred_over_observed"]))
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["n_samples", "n_rows", "element"], ascending=[False, False, True]
    ).reset_index(drop=True)


def build_temperature_support(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_cols = ["candidate_oxide_family", "carrier_type", "T_bin_label"]
    for keys, group in rows.groupby(group_cols, dropna=False, sort=False):
        family, carrier, t_bin = keys
        record: dict[str, Any] = {
            "candidate_oxide_family": family,
            "carrier_type": carrier,
            "T_bin_label": t_bin,
            "T_bin_center_K": float(pd.to_numeric(group["T_K"], errors="coerce").median()),
            "n_rows": int(len(group)),
            "n_samples": int(group["validation_sample_group_id"].nunique()),
            "n_papers": int(group["validation_paper_group_id"].nunique()),
        }
        record.update(numeric_summary(group["residual_log10_pred_over_observed"]))
        record["meets_reference_like_minimum_3rows_3samples_1paper"] = bool(
            record["n_rows"] >= 3 and record["n_samples"] >= 3 and record["n_papers"] >= 1
        )
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["candidate_oxide_family", "carrier_type", "T_bin_center_K"],
        na_position="last",
    ).reset_index(drop=True)


def build_summary(
    rows: pd.DataFrame,
    samples: pd.DataFrame,
    labels: pd.DataFrame,
    families: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        ("oxide_rows", len(rows), "All held-out Phase 1 rows currently classified as broad::oxide"),
        ("oxide_samples", samples["validation_sample_group_id"].nunique(), "Unique validation sample groups"),
        ("oxide_papers", rows["validation_paper_group_id"].nunique(), "Unique validation paper groups"),
        ("distinct_material_labels", len(labels), "Normalized formula/material labels"),
        ("candidate_composition_families", len(families), "Provisional audit groups; not a final taxonomy"),
        ("manual_review_samples", int(samples["manual_review_required"].sum()), "Samples with at least one review flag"),
        ("high_priority_review_samples", int(samples["review_priority"].eq("high").sum()), "Likely organic/polymer O-containing entries"),
        ("mixed_anion_samples", int(samples["has_chalcogen_anion_with_O"].sum()), "Contains O plus S, Se, or Te"),
        ("likely_not_inorganic_oxide_samples", int(samples["likely_not_inorganic_oxide"].sum()), "Heuristic false-positive candidates for broad oxide"),
        ("single_sample_material_labels", int(labels["single_sample_label"].sum()), "Labels represented by one held-out sample"),
        ("formula_name_mismatch_samples", int((~samples["formula_name_exact_match"]).sum()), "Formula and material-name text differ"),
        ("current_parse_reason_contains_O_rows", int(rows["current_broad_family_reason"].eq("contains_O").sum()), "Rows admitted by the current O-element rule"),
        ("parsed_from_formula_raw_rows", int(rows["formula_parse_source"].eq("formula_raw").sum()), "Rows whose current element parse used formula_raw"),
        ("row_equal_MAE_log10", float(rows["abs_residual_log10"].mean()), "Descriptive error on the oxide held-out cohort"),
        ("row_equal_factor10_accuracy", float(rows["within_factor_10"].astype(bool).mean()), "Share with absolute log10 residual <= 1"),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "definition"])


def make_manifest() -> pd.DataFrame:
    records = []
    for role, path in (
        ("primary Phase 1 residual rows", PHASE1_ROWS),
        ("Phase 1 sample summary", PHASE1_SAMPLES),
        ("exp006 parsing metadata", STEP6A_CANDIDATES),
        ("exp006 parsing/classification script", STEP6A_SCRIPT),
        ("Phase 0 baseline decision", PHASE0_DECISION),
    ):
        stat = path.stat()
        records.append(
            {
                "role": role,
                "path": relative(path),
                "size_bytes": stat.st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(records)


def append_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def write_payload(path: Path, tables: dict[str, pd.DataFrame], insights: list[str]) -> None:
    payload = {
        "title": "Phase 3 oxide data inventory",
        "scope_note": (
            "Frozen broad_family_100K held-out cohort. Candidate families are "
            "composition-only audit groups and are not the final oxide taxonomy."
        ),
        "insights": insights,
        "tables": {name: json_records(frame) for name, frame in tables.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_report(
    path: Path,
    rows: pd.DataFrame,
    samples: pd.DataFrame,
    labels: pd.DataFrame,
    families: pd.DataFrame,
    review: pd.DataFrame,
    temperature: pd.DataFrame,
) -> None:
    top = families.iloc[0]
    likely = int(samples["likely_not_inorganic_oxide"].sum())
    mixed = int(samples["has_chalcogen_anion_with_O"].sum())
    thin_bins = int((~temperature["meets_reference_like_minimum_3rows_3samples_1paper"]).sum())
    lines = [
        "# Phase 3: oxide data inventory",
        "",
        "## Scope",
        "",
        "Phase 3 audits the `broad::oxide` rows in the frozen `broad_family_100K` held-out cohort. It does not change the model, residuals, or exp006 taxonomy.",
        "",
        "## Inventory size",
        "",
        f"- Rows: {len(rows):,}",
        f"- Samples: {samples['validation_sample_group_id'].nunique():,}",
        f"- Papers: {rows['validation_paper_group_id'].nunique():,}",
        f"- Distinct normalized labels: {len(labels):,}",
        f"- Provisional composition families: {len(families):,}",
        "",
        "## Main findings",
        "",
        f"- Every current oxide row was admitted by the same rule: the parser found element `O` in `formula_raw`.",
        f"- The largest provisional group is `{top['candidate_oxide_family']}` with {int(top['n_samples']):,} samples and {int(top['n_rows']):,} rows.",
        f"- {int(samples['manual_review_required'].sum()):,} samples carry at least one review flag.",
        f"- {likely:,} samples are heuristic organic/polymer O-containing false-positive candidates rather than clear inorganic oxides.",
        f"- {mixed:,} samples are mixed-anion O+S/Se/Te systems and should not be silently merged with ordinary oxides.",
        f"- {int(labels['single_sample_label'].sum()):,} of {len(labels):,} material labels occur in only one held-out sample, so label-level subclasses would be too sparse without pooling.",
        f"- {thin_bins:,} provisional family/carrier/100 K evaluation cells have fewer than 3 rows, 3 samples, or 1 paper. This is an evaluation-cohort warning, not a training-feasibility test.",
        "",
        "## Interpretation",
        "",
        "The current `broad::oxide` group is compositionally heterogeneous. A finer taxonomy should first separate likely non-oxide organic entries and mixed-anion materials, then test a small number of pooled oxide families. Crystal structure labels such as perovskite, layered cobaltite, or delafossite cannot be assigned reliably from the present formula strings alone and require explicit rules or manual metadata review.",
        "",
        "## Outputs",
        "",
        "The CSV files are canonical machine-readable outputs. The Excel workbook is a filtered review view. `phase3_oxide_review_queue.csv` contains the samples needing manual chemistry review before the final taxonomy is fixed.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    manifest_before = make_manifest()
    phase1 = pd.read_csv(PHASE1_ROWS, low_memory=False)
    oxide_ids = set(
        phase1.loc[as_bool(phase1["is_oxide_broad"]), "row_id"].astype(str)
    )
    parsing = load_parsing_metadata(oxide_ids)
    rows = build_row_inventory(phase1, parsing)
    samples = build_sample_inventory(rows)
    labels = build_label_inventory(rows)
    families = build_family_summary(rows)
    elements = build_element_summary(rows)
    temperature = build_temperature_support(rows)
    review = samples.loc[samples["manual_review_required"]].copy().reset_index(drop=True)
    review_front = [
        "review_priority",
        "review_reasons",
        "likely_not_inorganic_oxide",
        "has_chalcogen_anion_with_O",
        "has_other_anion_marker_with_O",
        "has_multicomponent_notation",
        "has_free_text_label",
        "candidate_oxide_family",
        "formula_raw",
        "material_name_raw",
        "parsed_elements",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "doi",
        "paper_id",
        "sample_id",
        "sample_key",
        "carrier_types",
        "n_rows",
        "T_min_K",
        "T_max_K",
        "median_residual_log10",
        "mae_log10",
        "max_abs_residual_log10",
        "within_factor_10_fraction",
        "beyond_factor_10_count",
        "sample_abs_median_residual_rank",
    ]
    review = review[review_front + [column for column in review.columns if column not in review_front]]
    summary = build_summary(rows, samples, labels, families)

    checks: list[dict[str, Any]] = []
    append_check(checks, "all_current_oxide_rows_captured", len(rows) == len(oxide_ids), f"rows={len(rows):,}; expected={len(oxide_ids):,}")
    append_check(checks, "oxide_row_ids_unique", rows["row_id"].is_unique, f"unique={rows['row_id'].nunique():,}")
    append_check(checks, "parsing_join_complete", rows["parsed_elements"].notna().all(), f"missing={int(rows['parsed_elements'].isna().sum())}")
    append_check(checks, "all_parsed_elements_contain_O", rows["parsed_elements"].map(lambda value: "O" in element_set(value)).all(), "expected for current broad::oxide rule")
    append_check(checks, "all_current_reasons_are_contains_O", rows["current_broad_family_reason"].eq("contains_O").all(), join_unique(rows["current_broad_family_reason"]))
    append_check(checks, "all_current_parse_sources_are_formula_raw", rows["formula_parse_source"].eq("formula_raw").all(), join_unique(rows["formula_parse_source"]))
    append_check(checks, "sample_count_reconciles", samples["validation_sample_group_id"].nunique() == rows["validation_sample_group_id"].nunique(), f"samples={len(samples):,}")
    append_check(checks, "label_count_reconciles", len(labels) == rows["material_label_normalized"].nunique(), f"labels={len(labels):,}")
    append_check(checks, "family_rows_reconcile", int(families["n_rows"].sum()) == len(rows), f"sum={int(families['n_rows'].sum()):,}")
    append_check(checks, "family_samples_nonempty", families["n_samples"].ge(1).all(), f"families={len(families)}")
    append_check(checks, "review_queue_reconciles", len(review) == int(samples["manual_review_required"].sum()), f"review_samples={len(review):,}")
    append_check(checks, "residuals_finite", np.isfinite(pd.to_numeric(rows["residual_log10_pred_over_observed"], errors="coerce")).all(), f"rows={len(rows):,}")
    manifest_after = make_manifest()
    append_check(checks, "inputs_unchanged_during_build", manifest_before["sha256"].tolist() == manifest_after["sha256"].tolist(), "SHA-256 before and after build")
    validation = pd.DataFrame(checks)

    outputs = {
        "phase3_oxide_inventory_summary.csv": summary,
        "phase3_oxide_row_inventory.csv": rows,
        "phase3_oxide_sample_inventory.csv": samples,
        "phase3_oxide_label_inventory.csv": labels,
        "phase3_oxide_review_queue.csv": review,
        "phase3_oxide_candidate_family_summary.csv": families,
        "phase3_oxide_element_summary.csv": elements,
        "phase3_oxide_temperature_support.csv": temperature,
        "phase3_input_manifest.csv": manifest_after,
        "phase3_validation_checks.csv": validation,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")

    insights = [
        f"Current broad oxide: {len(rows):,} rows, {len(samples):,} samples, {rows['validation_paper_group_id'].nunique():,} papers.",
        f"All rows entered through the same contains-O formula rule; the current key is not a structural oxide taxonomy.",
        f"Manual review is requested for {len(review):,} samples, including {int(samples['likely_not_inorganic_oxide'].sum()):,} likely organic/polymer false-positive candidates.",
        f"Mixed-anion O+S/Se/Te materials account for {int(samples['has_chalcogen_anion_with_O'].sum()):,} samples and should be separated before oxide subtyping.",
        f"A small pooled taxonomy is preferable to label-level classes because {int(labels['single_sample_label'].sum()):,} of {len(labels):,} labels have only one held-out sample.",
    ]
    payload_tables = {
        "summary": summary,
        "families": families,
        "review": review,
        "samples": samples,
        "labels": labels,
        "elements": elements,
        "temperature": temperature,
        "rows": rows,
        "validation": validation,
        "manifest": manifest_after,
    }
    write_payload(args.workbook_payload, payload_tables, insights)
    write_report(args.report, rows, samples, labels, families, review, temperature)

    args.validation_log.write_text(
        "\n".join(
            [
                "Phase 3 oxide inventory validation",
                f"status={'PASS' if validation['status'].eq('PASS').all() else 'FAIL'}",
                f"oxide_rows={len(rows)}",
                f"oxide_samples={len(samples)}",
                f"oxide_papers={rows['validation_paper_group_id'].nunique()}",
                f"material_labels={len(labels)}",
                f"candidate_families={len(families)}",
                f"manual_review_samples={len(review)}",
                f"failed_checks={int(validation['status'].eq('FAIL').sum())}",
            ]
            + [f"{row.check}: {row.status} | {row.detail}" for row in validation.itertuples(index=False)]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Phase 3 inventory complete: rows={len(rows):,}, samples={len(samples):,}, "
        f"labels={len(labels):,}, review={len(review):,}, "
        f"status={'PASS' if validation['status'].eq('PASS').all() else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
