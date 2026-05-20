import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP5_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step5_core_properties"
DEFAULT_PROPERTY_CORE_CURVES = STEP5_DIR / "property_core_curves_step5.csv"
DEFAULT_CANDIDATE_CORE_CURVES = STEP5_DIR / "candidate_core_curves_step5.csv"
DEFAULT_SAMPLE_AVAILABILITY = STEP5_DIR / "sample_property_availability_step5.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step6_np_classification"

TEXT_COLUMNS = [
    "sample_key",
    "SID",
    "DOI",
    "sample_id",
    "curve_id",
    "curve_key",
    "composition",
]
PROPERTY_SOURCE_COLUMNS = [
    "property_step5",
    "property",
    "property_family",
    "prop_y_canonical",
    "prop_y",
    "prop_y_raw",
]
ABS_LIKE_COLUMNS = [
    "prop_y_raw",
    "prop_y_canonical",
    "property",
    "property_step5",
    "comments",
    "comments_x",
    "comments_y",
    "caption",
    "curve_note",
    "note",
]
PROPERTY_REQUIRED_COLUMNS = {"sample_key", "x_values_json", "y_values_json"}
SAMPLE_REQUIRED_COLUMNS = {"sample_key"}
SEEBECK_ZERO_TOL = 1e-30
EXCEL_MAX_ROWS = 1_048_576
EXCEL_PREVIEW_ROWS = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify samples as n/p/mixed/unknown from Seebeck signs."
    )
    parser.add_argument(
        "--property_core_curves",
        type=Path,
        default=DEFAULT_PROPERTY_CORE_CURVES,
        help="property_core_curves_step5.csv path",
    )
    parser.add_argument(
        "--candidate_core_curves",
        type=Path,
        default=None,
        help="optional candidate_core_curves_step5.csv path",
    )
    parser.add_argument(
        "--sample_availability",
        type=Path,
        default=DEFAULT_SAMPLE_AVAILABILITY,
        help="sample_property_availability_step5.csv path",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output directory",
    )
    return parser.parse_args()


def read_csv_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_bool(value: Any) -> bool:
    return normalize_text(value).casefold() in {"true", "1", "yes", "y"}


def ensure_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in TEXT_COLUMNS:
        if column in output.columns:
            output[column] = output[column].map(normalize_text)
    return output


def validate_property_input(df: pd.DataFrame) -> None:
    missing = sorted(PROPERTY_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise KeyError(f"property_core_curves_step5.csv missing columns: {missing}")
    if not any(column in df.columns for column in PROPERTY_SOURCE_COLUMNS):
        raise KeyError(
            "property_core_curves_step5.csv needs at least one property name column: "
            f"{PROPERTY_SOURCE_COLUMNS}"
        )


def validate_sample_input(df: pd.DataFrame) -> None:
    missing = sorted(SAMPLE_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise KeyError(f"sample_property_availability_step5.csv missing columns: {missing}")


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value).casefold())


def is_seebeck_row(row: pd.Series) -> bool:
    for column in PROPERTY_SOURCE_COLUMNS:
        if column not in row.index:
            continue
        text = compact_text(row[column])
        if not text:
            continue
        if text == "seebeck coefficient":
            return True
        if "seebeck" in text or "thermopower" in text or "thermoelectric power" in text:
            return True
    return False


def numeric_from_token(token: Any) -> float | None:
    text = normalize_text(token)
    if not text:
        return None
    lowered = text.casefold()
    if lowered in {"nan", "none", "null", "inf", "+inf", "-inf", "infinity", "-infinity"}:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def values_from_iterable(values: Any) -> tuple[list[float], bool]:
    if isinstance(values, (int, float, str)):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return [], False

    parsed: list[float] = []
    saw_unparseable = False
    for item in values:
        value = numeric_from_token(item)
        if value is None:
            if normalize_text(item):
                saw_unparseable = True
            continue
        parsed.append(value)
    return parsed, saw_unparseable


def parse_seebeck_values(raw_value: Any) -> tuple[list[float], str]:
    text = normalize_text(raw_value)
    if not text:
        return [], "ok"

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        values, saw_unparseable = values_from_iterable(parsed)
        if values or not saw_unparseable:
            return values, "ok"

    split_attempts = []
    if "," in text:
        split_attempts.append([part.strip() for part in text.split(",")])
    split_attempts.append(text.split())

    for tokens in split_attempts:
        values, saw_unparseable = values_from_iterable(tokens)
        if values:
            return values, "ok"
        if tokens and not saw_unparseable:
            return [], "ok"

    number_pattern = re.compile(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    )
    values = [float(match.group(0)) for match in number_pattern.finditer(text)]
    values = [value for value in values if math.isfinite(value)]
    if values:
        return values, "ok"
    return [], "parse_failed"


def is_abs_like(row: pd.Series) -> bool:
    parts: list[str] = []
    for column in ABS_LIKE_COLUMNS:
        if column in row.index:
            parts.append(normalize_text(row[column]))
    text = " | ".join(parts).casefold()
    if not text:
        return False
    patterns = [
        r"\babsolute\b",
        r"\babs\b",
        r"\|\s*s\s*\|",
        r"\bmagnitude\b",
        r"\bmodulus\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def seebeck_curve_stats(row: pd.Series) -> dict[str, Any]:
    if not is_seebeck_row(row):
        return {
            "seebeck_parse_status": "",
            "seebeck_abs_like_flag": "",
            "seebeck_positive_count_curve": "",
            "seebeck_negative_count_curve": "",
            "seebeck_zero_count_curve": "",
            "seebeck_valid_count_curve": "",
            "seebeck_sign_label_curve": "",
        }

    abs_like = is_abs_like(row)
    values, parse_status = parse_seebeck_values(row.get("y_values_json", ""))
    if parse_status == "parse_failed":
        label = "parse_failed"
    elif abs_like:
        label = "abs_like_excluded"
    elif not values:
        label = "no_valid_value"
    else:
        positive = sum(value > SEEBECK_ZERO_TOL for value in values)
        negative = sum(value < -SEEBECK_ZERO_TOL for value in values)
        zero = len(values) - positive - negative
        if positive and not negative:
            label = "positive"
        elif negative and not positive:
            label = "negative"
        elif positive and negative:
            label = "mixed"
        elif zero:
            label = "zero_only"
        else:
            label = "no_valid_value"
        return {
            "seebeck_parse_status": parse_status,
            "seebeck_abs_like_flag": bool(abs_like),
            "seebeck_positive_count_curve": positive,
            "seebeck_negative_count_curve": negative,
            "seebeck_zero_count_curve": zero,
            "seebeck_valid_count_curve": positive + negative,
            "seebeck_sign_label_curve": label,
        }

    return {
        "seebeck_parse_status": parse_status,
        "seebeck_abs_like_flag": bool(abs_like),
        "seebeck_positive_count_curve": 0,
        "seebeck_negative_count_curve": 0,
        "seebeck_zero_count_curve": 0,
        "seebeck_valid_count_curve": 0,
        "seebeck_sign_label_curve": label,
    }


def add_curve_sign_columns(curves_df: pd.DataFrame) -> pd.DataFrame:
    stats_df = pd.DataFrame.from_records(curves_df.apply(seebeck_curve_stats, axis=1))
    return pd.concat([curves_df.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)


def classify_sample(row: pd.Series) -> tuple[str, str, str, bool]:
    positive = int(row["seebeck_positive_count"])
    negative = int(row["seebeck_negative_count"])
    zero = int(row["seebeck_zero_count"])
    valid = positive + negative
    abs_like = int(row["seebeck_abs_like_curve_count"])
    parse_failed = int(row["seebeck_parse_failed_curve_count"])
    curve_count = int(row["seebeck_curve_count"])

    if valid == 0:
        if curve_count > 0 and abs_like == curve_count:
            return "unknown", "unknown: only absolute-like Seebeck curves available", "unknown", False
        if curve_count > 0 and parse_failed == curve_count:
            return "unknown", "unknown: Seebeck parse failed", "unknown", False
        return "unknown", "unknown: no valid signed Seebeck data", "unknown", False

    positive_ratio = positive / valid
    negative_ratio = negative / valid
    if positive_ratio >= 0.70:
        label = "p"
        basis_prefix = "estimated from Seebeck sign"
    elif negative_ratio >= 0.70:
        label = "n"
        basis_prefix = "estimated from Seebeck sign"
    else:
        label = "mixed"
        basis_prefix = "mixed Seebeck sign"

    if valid >= 5 and (positive_ratio >= 0.90 or negative_ratio >= 0.90):
        confidence = "high"
    elif valid >= 3 and (positive_ratio >= 0.70 or negative_ratio >= 0.70):
        confidence = "medium"
    elif valid > 0 and label == "mixed":
        confidence = "low"
    else:
        confidence = "low"

    basis = (
        f"{basis_prefix}: positive={positive}, negative={negative}, zero={zero}, "
        f"positive_ratio={positive_ratio:.3f}, negative_ratio={negative_ratio:.3f}"
    )
    return label, basis, confidence, label == "mixed"


def first_value(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    for value in group[column]:
        text = normalize_text(value)
        if text:
            return text
    return ""


def build_sample_classification(
    curves_step6: pd.DataFrame,
    availability_df: pd.DataFrame,
) -> pd.DataFrame:
    seebeck = curves_step6[curves_step6.apply(is_seebeck_row, axis=1)].copy()
    usable = seebeck[
        (seebeck["seebeck_parse_status"] == "ok")
        & (seebeck["seebeck_abs_like_flag"] == False)
    ].copy()

    aggregate = (
        usable.groupby("sample_key", sort=False)
        .agg(
            seebeck_positive_count=("seebeck_positive_count_curve", numeric_sum),
            seebeck_negative_count=("seebeck_negative_count_curve", numeric_sum),
            seebeck_zero_count=("seebeck_zero_count_curve", numeric_sum),
        )
        .reset_index()
    )
    curve_summary = (
        seebeck.groupby("sample_key", sort=False)
        .agg(
            seebeck_curve_count=("sample_key", "count"),
            seebeck_point_count_used_for_step6=("n_points_step5", numeric_sum),
            seebeck_abs_like_curve_count=("seebeck_abs_like_flag", bool_sum),
            seebeck_parse_failed_curve_count=(
                "seebeck_parse_status",
                lambda s: int((s == "parse_failed").sum()),
            ),
        )
        .reset_index()
    )

    sample = availability_df.copy()
    if "seebeck_curve_count" in sample.columns:
        sample = sample.rename(columns={"seebeck_curve_count": "seebeck_curve_count_step5"})
    if "seebeck_point_count" in sample.columns:
        sample = sample.rename(columns={"seebeck_point_count": "seebeck_point_count_step5"})

    if "n_or_p" in sample.columns:
        sample["n_or_p_prior"] = sample["n_or_p"]
    else:
        sample["n_or_p_prior"] = ""
    if "n_or_p_basis" in sample.columns:
        sample["n_or_p_basis_prior"] = sample["n_or_p_basis"]
    else:
        sample["n_or_p_basis_prior"] = ""

    sample = sample.merge(aggregate, on="sample_key", how="left")
    sample = sample.merge(curve_summary, on="sample_key", how="left")

    fill_zero = [
        "seebeck_positive_count",
        "seebeck_negative_count",
        "seebeck_zero_count",
        "seebeck_curve_count",
        "seebeck_point_count_used_for_step6",
        "seebeck_abs_like_curve_count",
        "seebeck_parse_failed_curve_count",
    ]
    for column in fill_zero:
        sample[column] = pd.to_numeric(sample[column], errors="coerce").fillna(0).astype(int)
    sample["seebeck_valid_count"] = (
        sample["seebeck_positive_count"] + sample["seebeck_negative_count"]
    )

    classifications = sample.apply(classify_sample, axis=1, result_type="expand")
    classifications.columns = [
        "n_or_p_step6",
        "n_or_p_basis_step6",
        "n_or_p_confidence_step6",
        "seebeck_sign_mixed_flag",
    ]
    sample = pd.concat([sample, classifications], axis=1)
    sample["n_or_p"] = sample["n_or_p_step6"]
    sample["n_or_p_basis"] = sample["n_or_p_basis_step6"]

    if "seebeck_point_count_step5" not in sample.columns:
        sample["seebeck_point_count_step5"] = 0
    sample["seebeck_point_count"] = sample["seebeck_point_count_used_for_step6"]

    return sample.loc[:, sample_classification_columns(sample)]


def numeric_sum(values: pd.Series) -> int:
    return int(pd.to_numeric(values, errors="coerce").fillna(0).sum())


def bool_sum(values: pd.Series) -> int:
    return int(values.map(normalize_bool).sum())


def sample_classification_columns(sample: pd.DataFrame) -> list[str]:
    preferred = [
        "sample_key",
        "SID",
        "DOI",
        "sample_id",
        "paper_title",
        "year",
        "composition",
        "material_system",
        "n_or_p_prior",
        "n_or_p_basis_prior",
        "n_or_p_step6",
        "n_or_p_basis_step6",
        "n_or_p_confidence_step6",
        "seebeck_positive_count",
        "seebeck_negative_count",
        "seebeck_zero_count",
        "seebeck_valid_count",
        "seebeck_curve_count",
        "seebeck_point_count",
        "seebeck_abs_like_curve_count",
        "seebeck_parse_failed_curve_count",
        "seebeck_sign_mixed_flag",
        "has_sigma_or_rho",
        "has_seebeck",
        "has_kappa_or_zt",
        "sigma_or_rho_point_count",
        "seebeck_point_count_step5",
        "seebeck_point_count_used_for_step6",
        "kappa_or_zt_point_count",
        "is_learning_candidate_step5",
        "learning_candidate_reason_step5",
        "n_or_p",
        "n_or_p_basis",
    ]
    return [column for column in preferred if column in sample.columns] + [
        column for column in sample.columns if column not in preferred
    ]


def merge_np_into_availability(
    availability_df: pd.DataFrame, classification_df: pd.DataFrame
) -> pd.DataFrame:
    output = availability_df.copy()
    if "seebeck_curve_count" in output.columns:
        output = output.rename(columns={"seebeck_curve_count": "seebeck_curve_count_step5"})
    if "seebeck_point_count" in output.columns:
        output = output.rename(columns={"seebeck_point_count": "seebeck_point_count_step5"})
    output["n_or_p_prior"] = output.get("n_or_p", "")
    output["n_or_p_basis_prior"] = output.get("n_or_p_basis", "")
    drop_columns = [column for column in ["n_or_p", "n_or_p_basis"] if column in output.columns]
    output = output.drop(columns=drop_columns)
    merge_columns = [
        "sample_key",
        "n_or_p",
        "n_or_p_basis",
        "n_or_p_step6",
        "n_or_p_basis_step6",
        "n_or_p_confidence_step6",
        "seebeck_positive_count",
        "seebeck_negative_count",
        "seebeck_zero_count",
        "seebeck_valid_count",
        "seebeck_curve_count",
        "seebeck_point_count",
        "seebeck_abs_like_curve_count",
        "seebeck_parse_failed_curve_count",
        "seebeck_sign_mixed_flag",
    ]
    output = output.merge(classification_df[merge_columns], on="sample_key", how="left")
    return output


def candidate_sample_mask(df: pd.DataFrame) -> pd.Series:
    if "is_learning_candidate_step5" in df.columns:
        return df["is_learning_candidate_step5"].map(normalize_bool)
    return (
        df.get("has_sigma_or_rho", "").map(normalize_bool)
        & df.get("has_seebeck", "").map(normalize_bool)
        & df.get("has_kappa_or_zt", "").map(normalize_bool)
        & (pd.to_numeric(df.get("sigma_or_rho_point_count", 0), errors="coerce").fillna(0) >= 5)
    )


def merge_np_into_curves(curves_df: pd.DataFrame, classification_df: pd.DataFrame) -> pd.DataFrame:
    output = curves_df.copy()
    output["n_or_p_prior"] = output.get("n_or_p", "")
    output["n_or_p_basis_prior"] = output.get("n_or_p_basis", "")
    output = output.drop(columns=[c for c in ["n_or_p", "n_or_p_basis"] if c in output.columns])
    merge_columns = [
        "sample_key",
        "n_or_p",
        "n_or_p_basis",
        "n_or_p_step6",
        "n_or_p_basis_step6",
        "n_or_p_confidence_step6",
    ]
    return output.merge(classification_df[merge_columns], on="sample_key", how="left")


def load_candidate_core(
    candidate_path: Path | None, property_curves_step6: pd.DataFrame
) -> pd.DataFrame:
    if candidate_path is not None and candidate_path.exists():
        candidate = ensure_text_columns(read_csv_text(candidate_path))
        validate_property_input(candidate)
        return add_curve_sign_columns(candidate)
    if "is_candidate_sample" in property_curves_step6.columns:
        return property_curves_step6[property_curves_step6["is_candidate_sample"].map(normalize_bool)].copy()
    if "is_learning_candidate_step5" in property_curves_step6.columns:
        return property_curves_step6[
            property_curves_step6["is_learning_candidate_step5"].map(normalize_bool)
        ].copy()
    raise KeyError(
        "candidate_core_curves_step5.csv not found and property_core_curves has no candidate flag"
    )


def build_report(
    property_input_rows: int,
    candidate_input_rows: int,
    availability_input_rows: int,
    property_curves_step6: pd.DataFrame,
    candidate_curves_step6: pd.DataFrame,
    classification_df: pd.DataFrame,
    candidate_samples_df: pd.DataFrame,
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    seebeck = property_curves_step6[property_curves_step6.apply(is_seebeck_row, axis=1)]
    parse_failed = seebeck["seebeck_parse_status"].eq("parse_failed")
    abs_like = seebeck["seebeck_abs_like_flag"].map(normalize_bool)

    report_rows: list[tuple[str, str]] = [
        ("input_property_core_curves_step5_rows", str(property_input_rows)),
        ("input_candidate_core_curves_step5_rows", str(candidate_input_rows)),
        ("input_sample_property_availability_step5_rows", str(availability_input_rows)),
        ("seebeck_coefficient_curve_count", str(len(seebeck))),
        ("seebeck_coefficient_point_count", str(numeric_sum(seebeck.get("n_points_step5", pd.Series(dtype=str))))),
        ("seebeck_y_values_json_parse_success_curve_count", str(int((~parse_failed).sum()))),
        ("seebeck_y_values_json_parse_failed_curve_count", str(int(parse_failed.sum()))),
        ("absolute_like_seebeck_curve_count", str(int(abs_like.sum()))),
        ("absolute_like_seebeck_sample_count", str(seebeck.loc[abs_like, "sample_key"].nunique())),
    ]

    for label, frame, column in (
        ("n_or_p_step6", classification_df, "n_or_p_step6"),
        ("candidate_n_or_p", candidate_samples_df, "n_or_p"),
        ("n_or_p_confidence_step6", classification_df, "n_or_p_confidence_step6"),
        ("candidate_n_or_p_confidence_step6", candidate_samples_df, "n_or_p_confidence_step6"),
    ):
        counts = frame[column].value_counts(dropna=False).sort_index()
        for key, value in counts.items():
            report_rows.append((f"{label}_{key}_sample_count", str(int(value))))

    for value in ["p", "n", "mixed", "unknown"]:
        report_rows.append(
            (f"{value}_sample_count", str(int(classification_df["n_or_p_step6"].eq(value).sum())))
        )
        report_rows.append(
            (f"candidate_{value}_sample_count", str(int(candidate_samples_df["n_or_p"].eq(value).sum())))
        )

    report_rows.extend(
        [
            (
                "seebeck_sign_mixed_sample_count",
                str(int(classification_df["seebeck_sign_mixed_flag"].map(normalize_bool).sum())),
            ),
            (
                "seebeck_valid_count_zero_sample_count",
                str(int(pd.to_numeric(classification_df["seebeck_valid_count"], errors="coerce").fillna(0).eq(0).sum())),
            ),
            (
                "seebeck_absolute_like_only_sample_count",
                str(int(classification_df["n_or_p_basis_step6"].eq("unknown: only absolute-like Seebeck curves available").sum())),
            ),
            (
                "seebeck_parse_failed_only_sample_count",
                str(int(classification_df["n_or_p_basis_step6"].eq("unknown: Seebeck parse failed").sum())),
            ),
        ]
    )

    prior = classification_df["n_or_p_prior"].map(normalize_np)
    step6 = classification_df["n_or_p_step6"].map(normalize_np)
    prior_known = prior.isin({"n", "p", "mixed"})
    step6_known = step6.isin({"n", "p", "mixed"})
    report_rows.extend(
        [
            ("prior_step6_match_sample_count", str(int((prior == step6).sum()))),
            (
                "prior_step6_mismatch_sample_count",
                str(int((prior.ne(step6) & prior.ne("") & step6.ne("")).sum())),
            ),
            (
                "prior_unknown_step6_classified_sample_count",
                str(int(((~prior_known) & step6_known).sum())),
            ),
            (
                "prior_classified_step6_unknown_sample_count",
                str(int((prior_known & step6.eq("unknown")).sum())),
            ),
        ]
    )

    learning_candidate_count = int(len(candidate_samples_df))
    report_rows.append(("learning_candidate_sample_count", str(learning_candidate_count)))
    for value in ["p", "n", "mixed", "unknown"]:
        report_rows.append(
            (
                f"learning_candidate_{value}_sample_count",
                str(int(candidate_samples_df["n_or_p"].eq(value).sum())),
            )
        )

    for note in excel_notes:
        report_rows.append(("excel_note", note))

    report_df = pd.DataFrame(report_rows, columns=["metric", "value"])
    report_text = "\n".join(f"{metric}: {value}" for metric, value in report_rows) + "\n"
    return report_text, report_df


def normalize_np(value: Any) -> str:
    text = normalize_text(value).casefold()
    if text in {"n", "n-type"}:
        return "n"
    if text in {"p", "p-type"}:
        return "p"
    if text == "mixed":
        return "mixed"
    if text == "unknown":
        return "unknown"
    return ""


def assert_acceptance(
    classification_df: pd.DataFrame,
    availability_step6: pd.DataFrame,
    candidate_samples: pd.DataFrame,
    property_curves_step6: pd.DataFrame,
    candidate_curves_step6: pd.DataFrame,
) -> None:
    required_class = {
        "sample_key",
        "n_or_p_step6",
        "n_or_p_basis_step6",
        "n_or_p_confidence_step6",
        "seebeck_positive_count",
        "seebeck_negative_count",
        "seebeck_valid_count",
    }
    missing = required_class - set(classification_df.columns)
    if missing:
        raise KeyError(f"sample_np_classification_step6.csv missing columns: {sorted(missing)}")
    if classification_df["sample_key"].duplicated().any():
        raise ValueError("sample_np_classification_step6.csv has duplicate sample_key")

    required_availability = {
        "n_or_p",
        "n_or_p_basis",
        "n_or_p_step6",
        "n_or_p_basis_step6",
        "n_or_p_confidence_step6",
    }
    missing = required_availability - set(availability_step6.columns)
    if missing:
        raise KeyError(f"sample_property_availability_step6.csv missing columns: {sorted(missing)}")
    if availability_step6["sample_key"].duplicated().any():
        raise ValueError("sample_property_availability_step6.csv has duplicate sample_key")

    if not candidate_samples["is_learning_candidate_step5"].map(normalize_bool).all():
        raise ValueError("candidate_samples_np_step6.csv contains non-learning-candidate samples")

    for label, frame in (
        ("property_core_curves_step6.csv", property_curves_step6),
        ("candidate_core_curves_step6.csv", candidate_curves_step6),
    ):
        missing = {"x_values_json", "y_values_json", "n_or_p"} - set(frame.columns)
        if missing:
            raise KeyError(f"{label} missing columns: {sorted(missing)}")


def write_csv_outputs(
    output_dir: Path,
    classification_df: pd.DataFrame,
    availability_step6: pd.DataFrame,
    candidate_samples: pd.DataFrame,
    property_curves_step6: pd.DataFrame,
    candidate_curves_step6: pd.DataFrame,
    report_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    classification_df.to_csv(output_dir / "sample_np_classification_step6.csv", index=False)
    availability_step6.to_csv(output_dir / "sample_property_availability_step6.csv", index=False)
    candidate_samples.to_csv(output_dir / "candidate_samples_np_step6.csv", index=False)
    property_curves_step6.to_csv(output_dir / "property_core_curves_step6.csv", index=False)
    candidate_curves_step6.to_csv(output_dir / "candidate_core_curves_step6.csv", index=False)
    (output_dir / "step6_np_classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )


def excel_frame(df: pd.DataFrame, sheet_name: str, excel_notes: list[str]) -> pd.DataFrame:
    if len(df) <= EXCEL_MAX_ROWS - 1:
        return df
    excel_notes.append(
        f"{sheet_name} exceeded Excel row limit; wrote first {EXCEL_PREVIEW_ROWS} rows to workbook"
    )
    return df.head(EXCEL_PREVIEW_ROWS)


def fit_worksheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
    preview = df.head(200)
    for index, column in enumerate(df.columns, start=1):
        max_length = len(str(column))
        if not preview.empty:
            max_length = max(max_length, int(preview[column].astype(str).map(len).max()))
        worksheet.column_dimensions[worksheet.cell(row=1, column=index).column_letter].width = min(
            max(max_length + 2, 12), 60
        )


def write_excel_output(
    output_dir: Path,
    classification_df: pd.DataFrame,
    availability_step6: pd.DataFrame,
    candidate_samples: pd.DataFrame,
    property_curves_step6: pd.DataFrame,
    candidate_curves_step6: pd.DataFrame,
    report_df: pd.DataFrame,
    excel_notes: list[str],
) -> None:
    workbook_path = output_dir / "starrydata2_step6_np_classification.xlsx"
    sheets = {
        "sample_np_classification": classification_df,
        "sample_property_availability": availability_step6,
        "candidate_samples_np": candidate_samples,
        "property_core_curves": excel_frame(property_curves_step6, "property_core_curves", excel_notes),
        "candidate_core_curves": excel_frame(candidate_curves_step6, "candidate_core_curves", excel_notes),
        "classification_report": report_df,
    }
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            fit_worksheet(writer, sheet_name, frame)


def main() -> None:
    args = parse_args()
    candidate_path = args.candidate_core_curves
    if candidate_path is None and DEFAULT_CANDIDATE_CORE_CURVES.exists():
        candidate_path = DEFAULT_CANDIDATE_CORE_CURVES

    property_core = ensure_text_columns(read_csv_text(args.property_core_curves))
    sample_availability = ensure_text_columns(read_csv_text(args.sample_availability))
    validate_property_input(property_core)
    validate_sample_input(sample_availability)

    property_with_curve_stats = add_curve_sign_columns(property_core)
    classification_df = build_sample_classification(property_with_curve_stats, sample_availability)
    availability_step6 = merge_np_into_availability(sample_availability, classification_df)
    candidate_samples = availability_step6[candidate_sample_mask(availability_step6)].copy()

    property_curves_step6 = merge_np_into_curves(property_with_curve_stats, classification_df)
    candidate_base = load_candidate_core(candidate_path, property_with_curve_stats)
    candidate_curves_step6 = merge_np_into_curves(candidate_base, classification_df)

    assert_acceptance(
        classification_df,
        availability_step6,
        candidate_samples,
        property_curves_step6,
        candidate_curves_step6,
    )

    excel_notes: list[str] = []
    report_text, report_df = build_report(
        property_input_rows=len(property_core),
        candidate_input_rows=len(candidate_base),
        availability_input_rows=len(sample_availability),
        property_curves_step6=property_curves_step6,
        candidate_curves_step6=candidate_curves_step6,
        classification_df=classification_df,
        candidate_samples_df=candidate_samples,
        excel_notes=excel_notes,
    )

    write_csv_outputs(
        args.output_dir,
        classification_df,
        availability_step6,
        candidate_samples,
        property_curves_step6,
        candidate_curves_step6,
        report_text,
    )
    write_excel_output(
        args.output_dir,
        classification_df,
        availability_step6,
        candidate_samples,
        property_curves_step6,
        candidate_curves_step6,
        report_df,
        excel_notes,
    )
    if excel_notes:
        report_text, report_df = build_report(
            property_input_rows=len(property_core),
            candidate_input_rows=len(candidate_base),
            availability_input_rows=len(sample_availability),
            property_curves_step6=property_curves_step6,
            candidate_curves_step6=candidate_curves_step6,
            classification_df=classification_df,
            candidate_samples_df=candidate_samples,
            excel_notes=excel_notes,
        )
        (args.output_dir / "step6_np_classification_report.txt").write_text(
            report_text, encoding="utf-8"
        )

    counts = classification_df["n_or_p_step6"].value_counts()
    candidate_counts = candidate_samples["n_or_p"].value_counts()
    seebeck_curves = property_curves_step6[property_curves_step6.apply(is_seebeck_row, axis=1)]
    print("Done.")
    print("Created:")
    print("- sample_np_classification_step6.csv")
    print("- sample_property_availability_step6.csv")
    print("- candidate_samples_np_step6.csv")
    print("- property_core_curves_step6.csv")
    print("- candidate_core_curves_step6.csv")
    print("- step6_np_classification_report.txt")
    print("- starrydata2_step6_np_classification.xlsx")
    print("")
    print("Summary:")
    print(f"samples classified: {len(classification_df)}")
    print(f"p samples: {int(counts.get('p', 0))}")
    print(f"n samples: {int(counts.get('n', 0))}")
    print(f"mixed samples: {int(counts.get('mixed', 0))}")
    print(f"unknown samples: {int(counts.get('unknown', 0))}")
    print(f"candidate samples: {len(candidate_samples)}")
    print(f"candidate p samples: {int(candidate_counts.get('p', 0))}")
    print(f"candidate n samples: {int(candidate_counts.get('n', 0))}")
    print(f"candidate mixed samples: {int(candidate_counts.get('mixed', 0))}")
    print(f"candidate unknown samples: {int(candidate_counts.get('unknown', 0))}")
    print(f"Seebeck curves: {len(seebeck_curves)}")
    print(
        "Seebeck parse failed curves: "
        f"{int(seebeck_curves['seebeck_parse_status'].eq('parse_failed').sum())}"
    )
    print(
        "absolute-like Seebeck curves: "
        f"{int(seebeck_curves['seebeck_abs_like_flag'].map(normalize_bool).sum())}"
    )


if __name__ == "__main__":
    main()
