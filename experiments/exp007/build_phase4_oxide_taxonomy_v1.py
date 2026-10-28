"""Define and apply the exp007 Phase 4 oxide taxonomy v1.

The output is a frozen, priority-ordered, composition-only taxonomy.  It uses
the Phase 3 inventory as its immutable source, assigns every current broad-
oxide row, separates organic/polymer entries and mixed-anion systems, and does
not claim crystal-structure labels that are absent from the source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from oxide_taxonomy_v1 import (
    DEFINITION_DATE,
    RULES,
    TAXONOMY_VERSION,
    classify_oxide_v1,
    parse_element_set,
    rules_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
PHASE3_ROWS = EXP007 / "data" / "processed" / "phase3_oxide_row_inventory.csv"
PHASE3_SAMPLES = EXP007 / "data" / "processed" / "phase3_oxide_sample_inventory.csv"
PHASE3_LABELS = EXP007 / "data" / "processed" / "phase3_oxide_label_inventory.csv"
PHASE3_CHECK_LOG = EXP007 / "reports" / "phase3_check_log.txt"
TAXONOMY_MODULE = EXP007 / "oxide_taxonomy_v1.py"

DEFAULT_OUTPUT = EXP007 / "data" / "processed"
DEFAULT_REPORT = EXP007 / "reports" / "phase4_oxide_v1_definition.md"
DEFAULT_VALIDATION = EXP007 / "reports" / "phase4_validation_log.txt"
DEFAULT_PAYLOAD = EXP007 / "reports" / ".phase4_workbook_payload.json"

ATOL = 1e-12
FACTOR_2_LOG10 = math.log10(2.0)
FACTOR_5_LOG10 = math.log10(5.0)
FACTOR_10_LOG10 = 1.0
PRIORITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


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
            PHASE3_ROWS,
            PHASE3_SAMPLES,
            PHASE3_LABELS,
            PHASE3_CHECK_LOG,
            TAXONOMY_MODULE,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Phase 4 inputs: {missing}")
    if "status=PASS" not in PHASE3_CHECK_LOG.read_text(encoding="utf-8"):
        raise ValueError("Phase 3 independent checks are not PASS")


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


def first_nonempty(series: Iterable[Any]) -> Any:
    for value in series:
        if pd.isna(value):
            continue
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


def build_row_assignments(rows: pd.DataFrame) -> pd.DataFrame:
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    classifications: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        key = (str(row.material_label_normalized), str(row.parsed_elements))
        if key not in cache:
            cache[key] = classify_oxide_v1(key[0], key[1])
        classifications.append(cache[key])
    classified = pd.DataFrame(classifications, index=rows.index)
    output = pd.concat([rows.copy(), classified], axis=1)
    output["oxide_v1_changed_from_phase3_candidate"] = (
        output["candidate_oxide_family"].astype(str)
        != output["oxide_v1_group_key"].astype(str)
    )
    order = output["oxide_v1_review_priority"].map(PRIORITY_ORDER)
    return output.assign(_priority_order=order).sort_values(
        ["_priority_order", "abs_residual_log10", "validation_sample_group_id", "T_K"],
        ascending=[False, False, True, True],
        na_position="last",
    ).drop(columns="_priority_order").reset_index(drop=True)


def sample_assignments(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sample_group, group in rows.groupby("validation_sample_group_id", sort=False):
        group_keys = sorted(group["oxide_v1_group_key"].dropna().unique())
        rule_ids = sorted(group["oxide_v1_rule_id"].dropna().unique())
        priorities = group["oxide_v1_review_priority"].dropna().unique()
        confidences = group["oxide_v1_classification_confidence"].dropna().unique()
        inclusion = sorted(group["oxide_v1_inclusion_status"].dropna().unique())
        conflict = len(group_keys) != 1 or len(rule_ids) != 1 or len(inclusion) != 1
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
            "parsed_elements": join_unique(group["parsed_elements"]),
            "phase3_candidate_oxide_family": join_unique(group["candidate_oxide_family"]),
            "taxonomy_version": TAXONOMY_VERSION,
            "oxide_v1_group_key": group_keys[0] if not conflict else "oxide_v1::sample_assignment_conflict",
            "oxide_v1_display_name_en": join_unique(group["oxide_v1_display_name_en"]),
            "oxide_v1_display_name_ja": join_unique(group["oxide_v1_display_name_ja"]),
            "oxide_v1_rule_id": rule_ids[0] if not conflict else "CONFLICT",
            "oxide_v1_scope": join_unique(group["oxide_v1_scope"]),
            "oxide_v1_default_model_action": join_unique(group["oxide_v1_default_model_action"]),
            "oxide_v1_inclusion_status": inclusion[0] if not conflict else "review",
            "oxide_v1_default_model_eligible": bool(group["oxide_v1_default_model_eligible"].all()) and not conflict,
            "oxide_v1_classification_confidence": min(confidences, key=CONFIDENCE_ORDER.get) if len(confidences) else "low",
            "oxide_v1_review_priority": max(priorities, key=PRIORITY_ORDER.get) if len(priorities) else "none",
            "oxide_v1_review_reasons": join_unique(group["oxide_v1_review_reasons"]),
            "oxide_v1_formula_like_label": bool(group["oxide_v1_formula_like_label"].all()),
            "oxide_v1_multicomponent_notation": bool(group["oxide_v1_multicomponent_notation"].any()),
            "oxide_v1_structural_class": "not_assigned_formula_only_v1",
            "sample_assignment_conflict": conflict,
            "carrier_types": join_unique(group["carrier_type"]),
            "n_rows": int(len(group)),
            "T_min_K": float(pd.to_numeric(group["T_K"], errors="coerce").min()),
            "T_max_K": float(pd.to_numeric(group["T_K"], errors="coerce").max()),
        }
        record.update(numeric_summary(group["residual_log10_pred_over_observed"]))
        records.append(record)
    output = pd.DataFrame(records)
    output["sample_abs_median_residual_rank"] = (
        output["median_residual_log10"].abs().rank(method="first", ascending=False).astype("Int64")
    )
    output["_priority_order"] = output["oxide_v1_review_priority"].map(PRIORITY_ORDER)
    return output.sort_values(
        ["_priority_order", "sample_abs_median_residual_rank", "validation_sample_group_id"],
        ascending=[False, True, True],
    ).drop(columns="_priority_order").reset_index(drop=True)


def label_assignments(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for label, group in rows.groupby("material_label_normalized", sort=False, dropna=False):
        priorities = group["oxide_v1_review_priority"].dropna().unique()
        confidences = group["oxide_v1_classification_confidence"].dropna().unique()
        record: dict[str, Any] = {
            "material_label_normalized": label,
            "formula_raw_examples": join_unique(group["formula_raw"], limit=5),
            "material_name_examples": join_unique(group["material_name_raw"], limit=5),
            "parsed_elements": join_unique(group["parsed_elements"]),
            "phase3_candidate_oxide_family": join_unique(group["candidate_oxide_family"]),
            "taxonomy_version": TAXONOMY_VERSION,
            "oxide_v1_group_key": join_unique(group["oxide_v1_group_key"]),
            "oxide_v1_display_name_en": join_unique(group["oxide_v1_display_name_en"]),
            "oxide_v1_display_name_ja": join_unique(group["oxide_v1_display_name_ja"]),
            "oxide_v1_rule_id": join_unique(group["oxide_v1_rule_id"]),
            "oxide_v1_scope": join_unique(group["oxide_v1_scope"]),
            "oxide_v1_inclusion_status": join_unique(group["oxide_v1_inclusion_status"]),
            "oxide_v1_default_model_eligible": bool(group["oxide_v1_default_model_eligible"].all()),
            "oxide_v1_classification_confidence": min(confidences, key=CONFIDENCE_ORDER.get) if len(confidences) else "low",
            "oxide_v1_review_priority": max(priorities, key=PRIORITY_ORDER.get) if len(priorities) else "none",
            "oxide_v1_review_reasons": join_unique(group["oxide_v1_review_reasons"]),
            "oxide_v1_formula_like_label": bool(group["oxide_v1_formula_like_label"].all()),
            "oxide_v1_multicomponent_notation": bool(group["oxide_v1_multicomponent_notation"].any()),
            "n_rows": int(len(group)),
            "n_samples": int(group["validation_sample_group_id"].nunique()),
            "n_papers": int(group["validation_paper_group_id"].nunique()),
            "carrier_types": join_unique(group["carrier_type"]),
            "T_min_K": float(pd.to_numeric(group["T_K"], errors="coerce").min()),
            "T_max_K": float(pd.to_numeric(group["T_K"], errors="coerce").max()),
        }
        record.update(numeric_summary(group["residual_log10_pred_over_observed"]))
        records.append(record)
    output = pd.DataFrame(records)
    output["single_sample_label"] = output["n_samples"].eq(1)
    output["_priority_order"] = output["oxide_v1_review_priority"].map(PRIORITY_ORDER)
    return output.sort_values(
        ["_priority_order", "n_samples", "n_rows", "material_label_normalized"],
        ascending=[False, False, False, True],
    ).drop(columns="_priority_order").reset_index(drop=True)


def class_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    total_rows = len(rows)
    total_samples = rows["validation_sample_group_id"].nunique()
    for group_key, group in rows.groupby("oxide_v1_group_key", sort=False):
        record: dict[str, Any] = {
            "taxonomy_version": TAXONOMY_VERSION,
            "oxide_v1_group_key": group_key,
            "oxide_v1_display_name_en": first_nonempty(group["oxide_v1_display_name_en"]),
            "oxide_v1_display_name_ja": first_nonempty(group["oxide_v1_display_name_ja"]),
            "oxide_v1_rule_id": first_nonempty(group["oxide_v1_rule_id"]),
            "oxide_v1_scope": first_nonempty(group["oxide_v1_scope"]),
            "oxide_v1_default_model_action": first_nonempty(group["oxide_v1_default_model_action"]),
            "n_rows": int(len(group)),
            "row_share": float(len(group) / total_rows),
            "n_samples": int(group["validation_sample_group_id"].nunique()),
            "sample_share": float(group["validation_sample_group_id"].nunique() / total_samples),
            "n_papers": int(group["validation_paper_group_id"].nunique()),
            "n_material_labels": int(group["material_label_normalized"].nunique()),
            "n_default_eligible_rows": int(as_bool(group["oxide_v1_default_model_eligible"]).sum()),
            "n_review_rows": int(group["oxide_v1_inclusion_status"].eq("review").sum()),
            "n_excluded_rows": int(group["oxide_v1_inclusion_status"].eq("exclude").sum()),
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
        ["n_samples", "n_rows", "oxide_v1_group_key"], ascending=[False, False, True]
    ).reset_index(drop=True)


def transition_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, group in rows.groupby(
        ["candidate_oxide_family", "oxide_v1_group_key"], sort=False
    ):
        phase3_family, v1_group = keys
        records.append(
            {
                "phase3_candidate_oxide_family": phase3_family,
                "oxide_v1_group_key": v1_group,
                "n_rows": int(len(group)),
                "n_samples": int(group["validation_sample_group_id"].nunique()),
                "n_papers": int(group["validation_paper_group_id"].nunique()),
                "n_material_labels": int(group["material_label_normalized"].nunique()),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["n_samples", "n_rows", "phase3_candidate_oxide_family"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def rules_with_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    rule_hash = rules_sha256()
    for rule in RULES:
        group = rows.loc[rows["oxide_v1_rule_id"].eq(rule["rule_id"])]
        record = {
            "taxonomy_version": TAXONOMY_VERSION,
            "definition_date": DEFINITION_DATE,
            "taxonomy_rules_sha256": rule_hash,
            **rule,
            "matched_rows": int(len(group)),
            "matched_samples": int(group["validation_sample_group_id"].nunique()),
            "matched_papers": int(group["validation_paper_group_id"].nunique()),
            "matched_material_labels": int(group["material_label_normalized"].nunique()),
        }
        records.append(record)
    return pd.DataFrame(records).sort_values("priority").reset_index(drop=True)


def rule_examples(labels: pd.DataFrame, max_examples: int = 5) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for rule_id, group in labels.groupby("oxide_v1_rule_id", sort=False):
        top = group.sort_values(["n_samples", "n_rows"], ascending=[False, False]).head(max_examples)
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            records.append(
                {
                    "oxide_v1_rule_id": rule_id,
                    "oxide_v1_group_key": row.oxide_v1_group_key,
                    "example_rank": rank,
                    "material_label_normalized": row.material_label_normalized,
                    "parsed_elements": row.parsed_elements,
                    "n_samples": row.n_samples,
                    "n_rows": row.n_rows,
                    "inclusion_status": row.oxide_v1_inclusion_status,
                    "review_reasons": row.oxide_v1_review_reasons,
                }
            )
    return pd.DataFrame(records).sort_values(
        ["oxide_v1_rule_id", "example_rank"]
    ).reset_index(drop=True)


def build_summary(
    rows: pd.DataFrame,
    samples: pd.DataFrame,
    labels: pd.DataFrame,
    classes: pd.DataFrame,
    review: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        ("taxonomy_version", TAXONOMY_VERSION, "Frozen classification version"),
        ("definition_date", DEFINITION_DATE, "Definition date"),
        ("taxonomy_rules_sha256", rules_sha256(), "Hash of the ordered rule definition"),
        ("source_rows", len(rows), "Phase 3 broad-oxide held-out rows classified"),
        ("source_samples", len(samples), "Unique validation sample groups"),
        ("source_papers", rows["validation_paper_group_id"].nunique(), "Unique validation paper groups"),
        ("source_material_labels", len(labels), "Normalized material labels"),
        ("nonempty_v1_groups", len(classes), "v1 groups with at least one row"),
        ("default_eligible_rows", int(as_bool(rows["oxide_v1_default_model_eligible"]).sum()), "Rows eligible by default under v1"),
        ("default_eligible_samples", int(as_bool(samples["oxide_v1_default_model_eligible"]).sum()), "Samples eligible by default under v1"),
        ("review_samples", int(samples["oxide_v1_inclusion_status"].eq("review").sum()), "Samples requiring review before default model use"),
        ("excluded_samples", int(samples["oxide_v1_inclusion_status"].eq("exclude").sum()), "Samples excluded from inorganic oxide v1"),
        ("formula_like_samples", int(as_bool(samples["oxide_v1_formula_like_label"]).sum()), "Samples with strict formula-like labels"),
        ("structural_classes_assigned", 0, "v1 is composition-only"),
        ("row_equal_MAE_log10", float(rows["abs_residual_log10"].mean()), "Descriptive error of the current predictions"),
        ("row_equal_factor10_accuracy", float(as_bool(rows["within_factor_10"]).mean()), "Current prediction share within factor 10"),
        ("review_queue_rows", len(review), "One row per sample needing review or exclusion confirmation"),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "definition"])


def make_manifest() -> pd.DataFrame:
    records = []
    for role, path in (
        ("Phase 3 row inventory", PHASE3_ROWS),
        ("Phase 3 sample inventory", PHASE3_SAMPLES),
        ("Phase 3 label inventory", PHASE3_LABELS),
        ("Phase 3 independent check log", PHASE3_CHECK_LOG),
        ("oxide taxonomy v1 implementation", TAXONOMY_MODULE),
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
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def write_payload(path: Path, tables: dict[str, pd.DataFrame], insights: list[str]) -> None:
    payload = {
        "title": "Phase 4 oxide taxonomy v1",
        "scope_note": (
            "Priority-ordered composition taxonomy applied to the frozen Phase 3 broad-oxide cohort. "
            "No crystal-structure class is inferred."
        ),
        "insights": insights,
        "tables": {name: json_records(frame) for name, frame in tables.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_report(
    path: Path,
    summary: pd.DataFrame,
    classes: pd.DataFrame,
    samples: pd.DataFrame,
) -> None:
    metric = dict(zip(summary["metric"], summary["value"]))
    top = classes.iloc[0]
    lines = [
        "# Phase 4: oxide taxonomy v1 definition",
        "",
        "## Definition",
        "",
        f"- Version: `{TAXONOMY_VERSION}`",
        f"- Definition date: `{DEFINITION_DATE}`",
        f"- Ordered-rule SHA-256: `{rules_sha256()}`",
        "- Source: Phase 3 broad-oxide held-out cohort",
        "- Scope: composition-only; no crystal-structure class is inferred",
        "",
        "## Rule order",
        "",
        "1. Exclude organic/polymer/carbon-composite O-containing labels.",
        "2. Separate mixed-anion O+S/Se/Te, O+As/P, and O+N/halogen entries.",
        "3. Split remaining inorganic O-containing entries by priority-ordered element markers.",
        "4. Use an explicit other-inorganic-O group instead of inventing a structure class.",
        "5. Use a fail-safe outside-scope class if parsed O is absent.",
        "",
        "## Applied cohort",
        "",
        f"- Rows: {int(metric['source_rows']):,}",
        f"- Samples: {int(metric['source_samples']):,}",
        f"- Papers: {int(metric['source_papers']):,}",
        f"- Material labels: {int(metric['source_material_labels']):,}",
        f"- Non-empty v1 groups: {int(metric['nonempty_v1_groups']):,}",
        f"- Default-eligible samples: {int(metric['default_eligible_samples']):,}",
        f"- Review samples: {int(metric['review_samples']):,}",
        f"- Excluded samples: {int(metric['excluded_samples']):,}",
        "",
        "## Descriptive findings",
        "",
        f"- The largest v1 group is `{top['oxide_v1_group_key']}` with {int(top['n_samples']):,} samples.",
        f"- {int(samples['sample_assignment_conflict'].sum()):,} sample-level assignment conflicts were found.",
        "- Error metrics describe the existing broad-family predictions and are not post-retraining performance estimates.",
        "",
        "## Limitations",
        "",
        "`oxide_v1` uses formula text and parsed elements only. It does not distinguish perovskite, spinel, delafossite, rocksalt, layered cobaltite, or other crystal structures. The O+N/halogen group and non-formula-like or multi-component labels remain review items before model revalidation.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    manifest_before = make_manifest()
    phase3_rows = pd.read_csv(PHASE3_ROWS, low_memory=False)
    rows = build_row_assignments(phase3_rows)
    samples = sample_assignments(rows)
    labels = label_assignments(rows)
    classes = class_summary(rows)
    transition = transition_summary(rows)
    rules = rules_with_coverage(rows)
    examples = rule_examples(labels)
    review = samples.loc[
        samples["oxide_v1_inclusion_status"].ne("include")
        | samples["oxide_v1_review_priority"].ne("none")
    ].copy().reset_index(drop=True)
    review_front = [
        "oxide_v1_review_priority",
        "oxide_v1_inclusion_status",
        "oxide_v1_review_reasons",
        "oxide_v1_group_key",
        "oxide_v1_display_name_ja",
        "formula_raw",
        "material_name_raw",
        "parsed_elements",
        "oxide_v1_formula_like_label",
        "oxide_v1_multicomponent_notation",
        "validation_sample_group_id",
        "doi",
        "n_rows",
        "median_residual_log10",
        "mae_log10",
        "within_factor_10_fraction",
    ]
    review = review[review_front + [column for column in review.columns if column not in review_front]]
    summary = build_summary(rows, samples, labels, classes, review)

    checks: list[dict[str, Any]] = []
    append_check(checks, "taxonomy_version_fixed", TAXONOMY_VERSION == "oxide_v1", TAXONOMY_VERSION)
    append_check(checks, "rule_priorities_unique", rules["priority"].is_unique, f"rules={len(rules)}")
    append_check(checks, "rule_ids_unique", rules["rule_id"].is_unique, f"rules={len(rules)}")
    append_check(checks, "rule_hash_consistent", rules["taxonomy_rules_sha256"].eq(rules_sha256()).all(), rules_sha256())
    append_check(checks, "all_phase3_rows_classified", len(rows) == len(phase3_rows), f"rows={len(rows):,}")
    append_check(checks, "row_ids_unique", rows["row_id"].is_unique, f"unique={rows['row_id'].nunique():,}")
    append_check(checks, "all_assignments_have_rule", rows["oxide_v1_rule_id"].notna().all(), f"missing={int(rows['oxide_v1_rule_id'].isna().sum())}")
    append_check(checks, "all_current_rows_contain_O", rows["parsed_elements"].map(lambda value: "O" in parse_element_set(value)).all(), "Phase 3 broad oxide expectation")
    append_check(checks, "fail_safe_rule_unused", int(rules.loc[rules['rule_id'].eq('OXV1_R999'), 'matched_rows'].iloc[0]) == 0, "OXV1_R999")
    append_check(checks, "class_rows_reconcile", int(classes["n_rows"].sum()) == len(rows), f"sum={int(classes['n_rows'].sum()):,}")
    append_check(checks, "transition_rows_reconcile", int(transition["n_rows"].sum()) == len(rows), f"sum={int(transition['n_rows'].sum()):,}")
    append_check(checks, "sample_count_reconciles", len(samples) == rows["validation_sample_group_id"].nunique(), f"samples={len(samples):,}")
    append_check(checks, "label_count_reconciles", len(labels) == rows["material_label_normalized"].nunique(), f"labels={len(labels):,}")
    append_check(checks, "no_sample_assignment_conflicts", not samples["sample_assignment_conflict"].any(), f"conflicts={int(samples['sample_assignment_conflict'].sum())}")
    append_check(checks, "inclusion_status_complete", samples["oxide_v1_inclusion_status"].isin({"include", "review", "exclude"}).all(), join_unique(samples["oxide_v1_inclusion_status"]))
    append_check(checks, "eligible_only_when_included", (as_bool(samples["oxide_v1_default_model_eligible"]) == samples["oxide_v1_inclusion_status"].eq("include")).all(), "sample-level status")
    append_check(checks, "structural_class_not_inferred", samples["oxide_v1_structural_class"].eq("not_assigned_formula_only_v1").all(), "formula-only v1")
    append_check(checks, "organic_exclusion_present", int(rules.loc[rules['rule_id'].eq('OXV1_R010'), 'matched_samples'].iloc[0]) > 0, f"samples={int(rules.loc[rules['rule_id'].eq('OXV1_R010'), 'matched_samples'].iloc[0])}")
    append_check(checks, "mixed_anion_rules_present", set(rows.loc[rows["oxide_v1_scope"].eq("mixed_anion_O_containing"), "oxide_v1_rule_id"]) == {"OXV1_R020", "OXV1_R030", "OXV1_R040"}, join_unique(rows.loc[rows["oxide_v1_scope"].eq("mixed_anion_O_containing"), "oxide_v1_rule_id"]))
    append_check(checks, "review_queue_reconciles", len(review) == int(samples["oxide_v1_inclusion_status"].ne("include").sum()), f"review_or_excluded={len(review):,}")
    manifest_after = make_manifest()
    append_check(checks, "inputs_unchanged_during_build", manifest_before["sha256"].tolist() == manifest_after["sha256"].tolist(), "SHA-256 before and after build")
    validation = pd.DataFrame(checks)

    outputs = {
        "phase4_oxide_v1_inventory_summary.csv": summary,
        "phase4_oxide_v1_rules.csv": rules,
        "phase4_oxide_v1_class_summary.csv": classes,
        "phase4_oxide_v1_row_assignments.csv": rows,
        "phase4_oxide_v1_sample_assignments.csv": samples,
        "phase4_oxide_v1_label_assignments.csv": labels,
        "phase4_oxide_v1_review_queue.csv": review,
        "phase4_oxide_v1_transition_from_phase3.csv": transition,
        "phase4_oxide_v1_rule_examples.csv": examples,
        "phase4_input_manifest.csv": manifest_after,
        "phase4_validation_checks.csv": validation,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")

    config_path = args.output_dir / "phase4_oxide_v1_classification_config.json"
    config_path.write_text(
        json.dumps(
            {
                "taxonomy_version": TAXONOMY_VERSION,
                "definition_date": DEFINITION_DATE,
                "taxonomy_rules_sha256": rules_sha256(),
                "scope": "composition-only classification of current broad::oxide entries",
                "structural_classification": "not assigned",
                "rules": RULES,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metric = dict(zip(summary["metric"], summary["value"]))
    insights = [
        f"oxide_v1 assigns all {len(rows):,} rows and {len(samples):,} samples with no sample-level conflicts.",
        f"{int(metric['default_eligible_samples']):,} samples are included by default; {int(metric['review_samples']):,} require review and {int(metric['excluded_samples']):,} are excluded.",
        "Organic/polymer/carbon-composite labels are removed before mixed-anion and metal-marker rules are applied.",
        "O+S/Se/Te, O+As/P, and O+N/halogen systems are no longer silently pooled with ordinary oxides.",
        "The v1 taxonomy is composition-only; crystal structure remains explicitly unassigned.",
    ]
    payload_tables = {
        "summary": summary,
        "rules": rules,
        "classes": classes,
        "review": review,
        "samples": samples,
        "labels": labels,
        "transition": transition,
        "examples": examples,
        "rows": rows,
        "validation": validation,
        "manifest": manifest_after,
    }
    write_payload(args.workbook_payload, payload_tables, insights)
    write_report(args.report, summary, classes, samples)

    args.validation_log.write_text(
        "\n".join(
            [
                "Phase 4 oxide taxonomy v1 validation",
                f"status={'PASS' if validation['status'].eq('PASS').all() else 'FAIL'}",
                f"taxonomy_version={TAXONOMY_VERSION}",
                f"rules_sha256={rules_sha256()}",
                f"rows={len(rows)}",
                f"samples={len(samples)}",
                f"nonempty_groups={len(classes)}",
                f"default_eligible_samples={int(metric['default_eligible_samples'])}",
                f"review_samples={int(metric['review_samples'])}",
                f"excluded_samples={int(metric['excluded_samples'])}",
                f"failed_checks={int(validation['status'].eq('FAIL').sum())}",
            ]
            + [f"{row.check}: {row.status} | {row.detail}" for row in validation.itertuples(index=False)]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Phase 4 taxonomy complete: rows={len(rows):,}, samples={len(samples):,}, "
        f"groups={len(classes)}, review={int(metric['review_samples']):,}, "
        f"excluded={int(metric['excluded_samples']):,}, "
        f"status={'PASS' if validation['status'].eq('PASS').all() else 'FAIL'}"
    )


if __name__ == "__main__":
    main()

