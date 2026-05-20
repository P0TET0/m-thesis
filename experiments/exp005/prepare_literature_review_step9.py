import argparse
import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP8_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step8_learning_candidates"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step9_literature_annotations"

INPUT_FILES = {
    "sample_availability": "sample_property_availability_step8.csv",
    "learning": "learning_candidates_step8.csv",
    "initial": "initial_tau_fit_candidates_step8.csv",
    "review": "review_needed_candidates_step8.csv",
    "candidate_curves": "candidate_core_curves_step8.csv",
    "sigma_rho_curves": "sigma_rho_curves_for_fitting_step8.csv",
}

PROPERTY_SOURCE_COLUMNS = [
    "property_step8",
    "property_step5",
    "property",
    "property_family",
    "prop_y_canonical",
    "prop_y",
    "prop_y_raw",
]

TEXT_COLUMNS = [
    "paper_title",
    "composition",
    "material_system",
    "caption",
    "comments",
    "prop_y_raw",
    "prop_y_canonical",
    "property",
    "property_step8",
    "property_step5",
]

NP_COLUMNS = [
    "n_or_p",
    "n_or_p_basis",
    "n_or_p_step6",
    "n_or_p_basis_step6",
    "n_or_p_confidence_step6",
]

MANUAL_COLUMNS = [
    "paper_checked_step9",
    "paper_check_date_step9",
    "paper_check_scope_step9",
    "additive_manual_step9",
    "structure_manual_step9",
    "np_type_paper_manual_step9",
    "np_basis_paper_manual_step9",
    "np_checked_in_paper_step9",
    "np_check_note_step9",
    "rare_metal_note_manual_step9",
    "toxicity_note_manual_step9",
    "manual_review_note_step9",
    "needs_sintering_check_later_step9",
]

STEP9_COLUMNS = [
    "doi_url",
    "text_for_literature_hint_step9",
    "sintering_status_step9",
    "sintering_note_step9",
    "sintering_keyword_detected_step9",
    "additive_auto_step9",
    "additive_evidence_auto_step9",
    "additive_keyword_detected_step9",
    "structure_auto_step9",
    "structure_evidence_auto_step9",
    "structure_keyword_detected_step9",
    "nanocarbon_keyword_detected_step9",
    "nanocarbon_type_auto_step9",
    "nanocarbon_evidence_auto_step9",
    "np_basis_auto_step9",
    "np_basis_source_step9",
    "np_type_paper_manual_step9",
    "np_basis_paper_manual_step9",
    "np_checked_in_paper_step9",
    "np_check_note_step9",
    "paper_checked_step9",
    "paper_check_date_step9",
    "paper_check_scope_step9",
    "additive_manual_step9",
    "structure_manual_step9",
    "rare_metal_note_manual_step9",
    "toxicity_note_manual_step9",
    "manual_review_note_step9",
    "needs_sintering_check_later_step9",
    "rare_metal_flag_auto_step9",
    "rare_metal_elements_auto_step9",
    "toxicity_flag_auto_step9",
    "toxicity_elements_auto_step9",
    "zt_max_observed_step9",
    "zt_point_count_step9",
    "thermal_conductivity_min_observed_step9",
    "thermal_conductivity_point_count_step9",
    "sigma_or_rho_point_count_step9",
    "seebeck_point_count_step9",
    "review_priority_score_step9",
    "review_priority_tier_step9",
    "review_priority_reason_step9",
]

SINTERING_NOTE_STEP9 = (
    "sintering intentionally not checked at step9; check only after error analysis "
    "for high-error, high-ZT, or final-paper samples"
)

ADDITIVE_KEYWORDS = [
    "doped",
    "doping",
    "substituted",
    "substitution",
    "added",
    "addition",
    "additive",
    "composite",
    "nanocomposite",
    "alloyed",
    "incorporated",
    "embedded",
    "loaded",
    "decorated",
    "coated",
    "filled",
    "codoped",
    "co-doped",
    "co doping",
    "impurity",
    "添加",
    "ドープ",
    "置換",
    "複合",
    "混合",
]

STRUCTURE_KEYWORDS = [
    "nanostructured",
    "nanostructure",
    "nanoparticle",
    "nanowire",
    "nanorod",
    "nanosheet",
    "nanotube",
    "nanocomposite",
    "porous",
    "pore",
    "grain boundary",
    "grain",
    "polycrystalline",
    "single crystal",
    "thin film",
    "film",
    "bulk",
    "layered",
    "lamellar",
    "textured",
    "oriented",
    "anisotropic",
    "superlattice",
    "mesoporous",
    "microstructure",
    "phase boundary",
    "defect",
    "vacancy",
    "dislocation",
    "2d",
    "one-dimensional",
    "one dimensional",
    "1d",
    "ナノ",
    "粒界",
    "多孔",
    "薄膜",
    "バルク",
    "層状",
    "配向",
    "異方性",
    "欠陥",
]

SINTERING_KEYWORDS = [
    "sps",
    "spark plasma sintering",
    "hot press",
    "hot pressing",
    "sinter",
    "sintering",
    "solid state reaction",
    "焼結",
]

NANOCARBON_PATTERNS = [
    ("CNT", ["carbon nanotube", "swcnt", "mwcnt", "カーボンナノチューブ"]),
    ("CNT", [r"\bcnt\b"]),
    ("graphene", ["reduced graphene oxide", "graphene oxide", "graphene", "rgo", "グラフェン"]),
    ("graphene", [r"\bgo\b"]),
    ("graphite", ["graphite", "黒鉛"]),
    ("carbon black", ["carbon black"]),
    ("carbon fiber", ["carbon fiber"]),
    ("fullerene", ["fullerene", "c60"]),
    ("other carbon", ["nanocarbon", "carbon-based", "carbon based", "炭素", "ナノカーボン"]),
]

RARE_METAL_ELEMENTS = {
    "Te",
    "Se",
    "Bi",
    "Sb",
    "Ge",
    "In",
    "Ga",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Pt",
    "Pd",
    "Rh",
    "Ru",
    "Ir",
    "Os",
    "Ag",
    "Au",
    "Y",
    "La",
    "Ce",
    "Nd",
    "Sm",
    "Gd",
    "Dy",
    "Yb",
}
TOXICITY_ELEMENTS = {"Pb", "Cd", "Hg", "Tl", "As", "Se", "Te", "Sb"}
NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9<>+]+", re.IGNORECASE)
EXCEL_MAX_ROWS = 1_048_576
EXCEL_PREVIEW_ROWS = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Step9 literature annotation tables from Step8 candidates."
    )
    parser.add_argument("--step8_dir", type=Path, default=DEFAULT_STEP8_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manual_annotations", type=Path, default=None)
    parser.add_argument("--top_n_review", type=int, default=300)
    return parser.parse_args()


def read_csv_text(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Input file not found: {path}")
        return pd.DataFrame()
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


def finite_float(value: Any) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value).casefold())


def validate_inputs(learning: pd.DataFrame, candidate_curves: pd.DataFrame) -> None:
    if "sample_key" not in learning.columns:
        raise KeyError("learning_candidates_step8.csv missing required column: sample_key")
    missing = [
        column
        for column in ["sample_key", "x_values_json", "y_values_json"]
        if column not in candidate_curves.columns
    ]
    if missing:
        raise KeyError(f"candidate_core_curves_step8.csv missing required columns: {missing}")
    if not any(column in candidate_curves.columns for column in PROPERTY_SOURCE_COLUMNS):
        raise KeyError(
            "candidate_core_curves_step8.csv needs at least one property source column: "
            f"{PROPERTY_SOURCE_COLUMNS}"
        )


def ensure_sample_key(df: pd.DataFrame, filename: str) -> None:
    if "sample_key" not in df.columns:
        raise KeyError(f"{filename} missing required column: sample_key")


def extract_doi(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    text = text.split(",", 1)[0].split("，", 1)[0].strip()
    decoded = unquote(text)
    match = DOI_RE.search(decoded)
    if not match:
        return ""
    return match.group(0).rstrip(".,")


def doi_to_url(value: Any) -> str:
    doi = extract_doi(value)
    if not doi:
        return ""
    return f"https://doi.org/{quote(doi, safe='/')}"


def standardize_sintering(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["sintering_method"] = "unknown"
    output["sintering_checked"] = "no"
    output["record_checked"] = "no"
    output["sintering_status_step9"] = "not_checked"
    output["sintering_note_step9"] = SINTERING_NOTE_STEP9
    return output


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def combine_unique_texts(values: pd.Series, max_chars: int = 6000) -> str:
    seen: set[str] = set()
    chunks: list[str] = []
    for value in values:
        text = normalize_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        chunks.append(text)
        if sum(len(chunk) for chunk in chunks) > max_chars:
            break
    combined = " | ".join(chunks)
    return combined[:max_chars]


def build_curve_texts(candidate_curves: pd.DataFrame) -> pd.DataFrame:
    existing = [column for column in TEXT_COLUMNS if column in candidate_curves.columns]
    if not existing:
        return pd.DataFrame(columns=["sample_key", "curve_text_for_literature_hint_step9"])
    text_rows = candidate_curves.loc[:, ["sample_key"] + existing].copy()
    text_rows["curve_text_for_literature_hint_step9"] = text_rows[existing].agg(
        lambda row: " | ".join(normalize_text(value) for value in row if normalize_text(value)),
        axis=1,
    )
    return (
        text_rows.groupby("sample_key", sort=True)["curve_text_for_literature_hint_step9"]
        .apply(combine_unique_texts)
        .reset_index()
    )


def keyword_context(text: str, keyword: str, context_chars: int = 45) -> str:
    lower = text.casefold()
    target = keyword.casefold()
    index = lower.find(target)
    if index < 0:
        return ""
    start = max(index - context_chars, 0)
    end = min(index + len(keyword) + context_chars, len(text))
    return re.sub(r"\s+", " ", text[start:end]).strip()


def detect_keywords(text: str, keywords: list[str]) -> tuple[bool, str, str]:
    raw_text = normalize_text(text)
    lower = raw_text.casefold()
    found: list[str] = []
    evidence: list[str] = []
    for keyword in keywords:
        key = keyword.casefold()
        if key in lower:
            found.append(keyword)
            context = keyword_context(raw_text, keyword)
            if context:
                evidence.append(f"{keyword}: {context}")
        if len(found) >= 6:
            break
    return bool(found), ", ".join(found), " || ".join(evidence[:4])


def detect_regex_or_text(text: str, pattern: str) -> bool:
    lower = normalize_text(text).casefold()
    if pattern.startswith("\\"):
        return re.search(pattern, lower, flags=re.IGNORECASE) is not None
    return pattern.casefold() in lower


def detect_nanocarbon(text: str) -> tuple[bool, str, str]:
    raw_text = normalize_text(text)
    detected_types: list[str] = []
    evidence: list[str] = []
    for carbon_type, patterns in NANOCARBON_PATTERNS:
        for pattern in patterns:
            if detect_regex_or_text(raw_text, pattern):
                if carbon_type not in detected_types:
                    detected_types.append(carbon_type)
                literal = pattern.replace("\\b", "")
                context = keyword_context(raw_text, literal) if not pattern.startswith("\\") else ""
                if context:
                    evidence.append(f"{carbon_type}: {context}")
                break
    if not detected_types:
        return False, "unknown", ""
    return True, ", ".join(detected_types), " || ".join(evidence[:4])


def detect_elements(composition: Any, target_elements: set[str]) -> list[str]:
    text = normalize_text(composition)
    if not text:
        return []
    elements = set(re.findall(r"[A-Z][a-z]?", text))
    return sorted(elements & target_elements)


def values_from_iterable(parsed: Any) -> tuple[list[float], bool]:
    if isinstance(parsed, (str, bytes)):
        return [], True
    if not isinstance(parsed, (list, tuple, np.ndarray)):
        return [], True
    values: list[float] = []
    saw_unparseable = False
    for item in parsed:
        number = finite_float(item)
        if number is None:
            if normalize_text(item):
                saw_unparseable = True
            continue
        values.append(number)
    return values, saw_unparseable


def parse_numeric_values(raw_value: Any) -> tuple[list[float], bool]:
    text = normalize_text(raw_value)
    if not text:
        return [], True
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        values, saw_unparseable = values_from_iterable(parsed)
        if values or not saw_unparseable:
            return values, False
    for separator in (",", None):
        if separator == "," and "," not in text:
            continue
        tokens = text.split(separator) if separator else text.split()
        if len(tokens) <= 1:
            continue
        values = [number for token in tokens if (number := finite_float(token)) is not None]
        if values:
            return values, False
    values = [float(match.group(0)) for match in NUMERIC_RE.finditer(text)]
    values = [value for value in values if math.isfinite(value)]
    return values, not bool(values)


def classify_property_text(value: Any) -> str:
    text = compact_text(value)
    if not text:
        return ""
    if "power factor" in text:
        return ""
    if text == "zt" or "dimensionless figure of merit" in text or "figure of merit" in text:
        return "ZT"
    if "seebeck" in text or "thermopower" in text or "thermoelectric power" in text:
        return "Seebeck coefficient"
    if (
        "electrical_resistivity" in text
        or "electrical resistivity" in text
        or "electric resistivity" in text
        or re.search(r"(^|[^a-z])rho([^a-z]|$)", text)
        or "\u03c1" in text
    ):
        return "Electrical resistivity"
    if "resistivity" in text and "thermal" not in text:
        return "Electrical resistivity"
    if (
        "thermal_conductivity" in text
        or "thermal conductivity" in text
        or "total thermal conductivity" in text
    ):
        return "Thermal conductivity"
    if (
        "electrical_conductivity" in text
        or "electrical conductivity" in text
        or "electric conductivity" in text
        or re.search(r"(^|[^a-z])sigma([^a-z]|$)", text)
        or "\u03c3" in text
    ):
        return "Electrical conductivity"
    if "conductivity" in text and "thermal" not in text:
        return "Electrical conductivity"
    return ""


def classify_property_row(row: pd.Series) -> str:
    for column in PROPERTY_SOURCE_COLUMNS:
        if column in row.index:
            property_name = classify_property_text(row[column])
            if property_name:
                return property_name
    return ""


def build_property_stats(candidate_curves: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    curves = candidate_curves.copy()
    curves["property_step9_tmp"] = curves.apply(classify_property_row, axis=1)
    records: list[dict[str, Any]] = []
    parse_failures = {"zt_parse_failed_curve_count": 0, "thermal_parse_failed_curve_count": 0}
    for sample_key, group in curves.groupby("sample_key", sort=True):
        zt_values: list[float] = []
        kappa_values: list[float] = []
        for row in group.itertuples(index=False):
            property_name = getattr(row, "property_step9_tmp")
            if property_name not in {"ZT", "Thermal conductivity"}:
                continue
            values, failed = parse_numeric_values(getattr(row, "y_values_json", ""))
            if property_name == "ZT":
                zt_values.extend(values)
                if failed:
                    parse_failures["zt_parse_failed_curve_count"] += 1
            elif property_name == "Thermal conductivity":
                kappa_values.extend(values)
                if failed:
                    parse_failures["thermal_parse_failed_curve_count"] += 1
        records.append(
            {
                "sample_key": sample_key,
                "zt_max_observed_step9": max(zt_values) if zt_values else "",
                "zt_point_count_step9": len(zt_values),
                "thermal_conductivity_min_observed_step9": min(kappa_values)
                if kappa_values
                else "",
                "thermal_conductivity_point_count_step9": len(kappa_values),
            }
        )
    return pd.DataFrame(records), parse_failures


def choose_basis(row: pd.Series) -> str:
    basis = normalize_text(row.get("n_or_p_basis_step6", ""))
    if basis:
        return basis
    return normalize_text(row.get("n_or_p_basis", ""))


def score_to_tier(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 50:
        return "B"
    if score >= 25:
        return "C"
    if score >= 0:
        return "review"
    return "low"


def review_priority(row: pd.Series) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []

    if normalize_bool(row.get("is_initial_tau_fit_candidate_step8", "")):
        score += 30
        reasons.append("+30 initial tau fit")

    tier8 = normalize_text(row.get("candidate_priority_tier_step8", "")).upper()
    if tier8 == "A":
        score += 30
        reasons.append("+30 Step8 tier A")
    elif tier8 == "B":
        score += 20
        reasons.append("+20 Step8 tier B")
    elif tier8 == "C":
        score += 10
        reasons.append("+10 Step8 tier C")

    zt_max = finite_float(row.get("zt_max_observed_step9", ""))
    if zt_max is not None and zt_max >= 1:
        score += 40
        reasons.append("+40 ZT>=1")
    elif zt_max is not None and zt_max >= 0.5:
        score += 20
        reasons.append("+20 ZT>=0.5")

    if normalize_bool(row.get("nanocarbon_keyword_detected_step9", "")):
        score += 25
        reasons.append("+25 nanocarbon")
    if normalize_bool(row.get("structure_keyword_detected_step9", "")):
        score += 10
        reasons.append("+10 structure keyword")
    if normalize_bool(row.get("additive_keyword_detected_step9", "")):
        score += 10
        reasons.append("+10 additive keyword")

    confidence = normalize_text(row.get("n_or_p_confidence_step6", "")).casefold()
    if confidence == "high":
        score += 10
        reasons.append("+10 high n/p confidence")
    elif confidence == "medium":
        score += 5
        reasons.append("+5 medium n/p confidence")

    n_or_p = normalize_text(row.get("n_or_p", "")).casefold()
    if n_or_p == "mixed":
        score -= 10
        reasons.append("-10 mixed n/p")
    elif n_or_p == "unknown":
        score -= 20
        reasons.append("-20 unknown n/p")

    if normalize_bool(row.get("toxicity_flag_auto_step9", "")):
        score -= 5
        reasons.append("-5 toxicity memo")
    if normalize_bool(row.get("rare_metal_flag_auto_step9", "")):
        score -= 5
        reasons.append("-5 rare metal memo")

    return score, "; ".join(reasons) if reasons else "no priority signal"


def build_base_sample_table(
    sample_availability: pd.DataFrame,
    learning: pd.DataFrame,
    initial: pd.DataFrame,
    review: pd.DataFrame,
) -> pd.DataFrame:
    if not sample_availability.empty:
        base = sample_availability.copy()
    else:
        base = pd.concat([learning, initial, review], ignore_index=True, sort=False)
    base = base.drop_duplicates("sample_key", keep="first").copy()
    return base


def add_step9_annotations(
    base: pd.DataFrame,
    candidate_curves: pd.DataFrame,
    property_stats: pd.DataFrame,
) -> pd.DataFrame:
    output = base.copy()
    curve_texts = build_curve_texts(candidate_curves)
    output = output.merge(curve_texts, on="sample_key", how="left")
    output["curve_text_for_literature_hint_step9"] = output[
        "curve_text_for_literature_hint_step9"
    ].fillna("")

    sample_text = output.apply(
        lambda row: " | ".join(
            normalize_text(row.get(column, ""))
            for column in ["paper_title", "composition", "material_system"]
            if normalize_text(row.get(column, ""))
        ),
        axis=1,
    )
    output["text_for_literature_hint_step9"] = (
        sample_text + " | " + output["curve_text_for_literature_hint_step9"]
    ).map(lambda value: normalize_text(value).strip(" |")[:6000])
    output = output.drop(columns=["curve_text_for_literature_hint_step9"])

    if "DOI" in output.columns:
        output["DOI"] = output["DOI"].map(extract_doi)
        output["doi_url"] = output["DOI"].map(doi_to_url)
    else:
        output["doi_url"] = ""
    output = standardize_sintering(output)

    sintering_detected = output["text_for_literature_hint_step9"].map(
        lambda text: detect_keywords(text, SINTERING_KEYWORDS)[0]
    )
    output["sintering_keyword_detected_step9"] = sintering_detected

    additive = output["text_for_literature_hint_step9"].map(
        lambda text: detect_keywords(text, ADDITIVE_KEYWORDS)
    )
    output["additive_keyword_detected_step9"] = additive.map(lambda item: item[0])
    output["additive_auto_step9"] = additive.map(lambda item: item[1] if item[0] else "unknown")
    output["additive_evidence_auto_step9"] = additive.map(lambda item: item[2])

    structure = output["text_for_literature_hint_step9"].map(
        lambda text: detect_keywords(text, STRUCTURE_KEYWORDS)
    )
    output["structure_keyword_detected_step9"] = structure.map(lambda item: item[0])
    output["structure_auto_step9"] = structure.map(lambda item: item[1] if item[0] else "unknown")
    output["structure_evidence_auto_step9"] = structure.map(lambda item: item[2])

    nanocarbon = output["text_for_literature_hint_step9"].map(detect_nanocarbon)
    output["nanocarbon_keyword_detected_step9"] = nanocarbon.map(lambda item: item[0])
    output["nanocarbon_type_auto_step9"] = nanocarbon.map(lambda item: item[1])
    output["nanocarbon_evidence_auto_step9"] = nanocarbon.map(lambda item: item[2])

    composition = output["composition"] if "composition" in output.columns else pd.Series("", index=output.index)
    rare_elements = composition.map(lambda value: detect_elements(value, RARE_METAL_ELEMENTS))
    toxic_elements = composition.map(lambda value: detect_elements(value, TOXICITY_ELEMENTS))
    output["rare_metal_flag_auto_step9"] = rare_elements.map(bool)
    output["rare_metal_elements_auto_step9"] = rare_elements.map(lambda items: ", ".join(items))
    output["toxicity_flag_auto_step9"] = toxic_elements.map(bool)
    output["toxicity_elements_auto_step9"] = toxic_elements.map(lambda items: ", ".join(items))

    output["np_basis_auto_step9"] = output.apply(choose_basis, axis=1)
    output["np_basis_source_step9"] = "Seebeck sign from Starrydata2"
    output["np_type_paper_manual_step9"] = ""
    output["np_basis_paper_manual_step9"] = ""
    output["np_checked_in_paper_step9"] = "no"
    output["np_check_note_step9"] = ""

    output["paper_checked_step9"] = "no"
    output["paper_check_date_step9"] = ""
    output["paper_check_scope_step9"] = "not checked"
    output["additive_manual_step9"] = ""
    output["structure_manual_step9"] = ""
    output["rare_metal_note_manual_step9"] = ""
    output["toxicity_note_manual_step9"] = ""
    output["manual_review_note_step9"] = ""
    output["needs_sintering_check_later_step9"] = "no"

    output = output.merge(property_stats, on="sample_key", how="left")
    for column in [
        "zt_max_observed_step9",
        "zt_point_count_step9",
        "thermal_conductivity_min_observed_step9",
        "thermal_conductivity_point_count_step9",
    ]:
        if column not in output.columns:
            output[column] = ""
        output[column] = output[column].fillna("")

    output["sigma_or_rho_point_count_step9"] = output.get(
        "valid_sigma_or_rho_point_count_step8", output.get("sigma_or_rho_point_count_step8", "")
    )
    output["seebeck_point_count_step9"] = output.get(
        "seebeck_point_count_step8", output.get("seebeck_point_count", "")
    )

    priorities = output.apply(review_priority, axis=1, result_type="expand")
    priorities.columns = ["review_priority_score_step9", "review_priority_reason_step9"]
    output = pd.concat([output, priorities], axis=1)
    output["review_priority_tier_step9"] = output["review_priority_score_step9"].map(score_to_tier)
    return output


def apply_manual_annotations(
    annotations: pd.DataFrame,
    manual_path: Path | None,
) -> tuple[pd.DataFrame, int, bool]:
    if manual_path is None:
        return annotations, 0, False
    manual = read_csv_text(manual_path)
    ensure_sample_key(manual, str(manual_path))
    output = annotations.copy()
    manual = manual.drop_duplicates("sample_key", keep="last").set_index("sample_key")
    output = output.set_index("sample_key")
    reflected_keys: set[str] = set()
    for column in MANUAL_COLUMNS:
        if column not in manual.columns:
            continue
        for sample_key, value in manual[column].items():
            text = normalize_text(value)
            if not text or sample_key not in output.index:
                continue
            output.at[sample_key, column] = text
            reflected_keys.add(sample_key)
    return output.reset_index(), len(reflected_keys), True


def step9_info_for_merge(annotations: pd.DataFrame) -> pd.DataFrame:
    columns = ["sample_key"] + [column for column in STEP9_COLUMNS if column in annotations.columns]
    return annotations.loc[:, columns].drop_duplicates("sample_key", keep="first")


def merge_step9_columns(base: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    info = step9_info_for_merge(annotations)
    drop_columns = [column for column in info.columns if column != "sample_key" and column in base.columns]
    output = base.drop(columns=drop_columns, errors="ignore").merge(info, on="sample_key", how="left")
    output = standardize_sintering(output)
    return output


def order_columns(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    columns = [column for column in preferred if column in df.columns]
    columns += [column for column in df.columns if column not in columns]
    return df.loc[:, columns]


def sample_annotation_columns() -> list[str]:
    return [
        "sample_key",
        "SID",
        "DOI",
        "doi_url",
        "sample_id",
        "paper_title",
        "year",
        "composition",
        "material_system",
        "n_or_p",
        "n_or_p_basis",
        "n_or_p_step6",
        "n_or_p_basis_step6",
        "n_or_p_confidence_step6",
        "np_basis_auto_step9",
        "np_basis_source_step9",
        "np_type_paper_manual_step9",
        "np_basis_paper_manual_step9",
        "np_checked_in_paper_step9",
        "np_check_note_step9",
        "sintering_method",
        "sintering_checked",
        "record_checked",
        "sintering_status_step9",
        "sintering_note_step9",
        "sintering_keyword_detected_step9",
        "additive_auto_step9",
        "additive_evidence_auto_step9",
        "additive_keyword_detected_step9",
        "additive_manual_step9",
        "structure_auto_step9",
        "structure_evidence_auto_step9",
        "structure_keyword_detected_step9",
        "structure_manual_step9",
        "nanocarbon_keyword_detected_step9",
        "nanocarbon_type_auto_step9",
        "nanocarbon_evidence_auto_step9",
        "rare_metal_flag_auto_step9",
        "rare_metal_elements_auto_step9",
        "toxicity_flag_auto_step9",
        "toxicity_elements_auto_step9",
        "rare_metal_note_manual_step9",
        "toxicity_note_manual_step9",
        "zt_max_observed_step9",
        "thermal_conductivity_min_observed_step9",
        "sigma_or_rho_point_count_step9",
        "seebeck_point_count_step9",
        "candidate_priority_tier_step8",
        "fitting_source_preference_step8",
        "valid_sigma_or_rho_point_count_step8",
        "sigma_or_rho_temperature_span_step8",
        "review_priority_score_step9",
        "review_priority_tier_step9",
        "review_priority_reason_step9",
        "paper_checked_step9",
        "paper_check_date_step9",
        "paper_check_scope_step9",
        "manual_review_note_step9",
        "needs_sintering_check_later_step9",
        "text_for_literature_hint_step9",
    ]


def high_priority_columns() -> list[str]:
    return [
        "sample_key",
        "DOI",
        "doi_url",
        "paper_title",
        "year",
        "sample_id",
        "composition",
        "material_system",
        "n_or_p",
        "n_or_p_confidence_step6",
        "np_basis_auto_step9",
        "additive_auto_step9",
        "structure_auto_step9",
        "nanocarbon_type_auto_step9",
        "zt_max_observed_step9",
        "thermal_conductivity_min_observed_step9",
        "review_priority_score_step9",
        "review_priority_tier_step9",
        "review_priority_reason_step9",
        "paper_checked_step9",
        "additive_manual_step9",
        "structure_manual_step9",
        "np_type_paper_manual_step9",
        "np_basis_paper_manual_step9",
        "manual_review_note_step9",
    ]


def manual_template_columns() -> list[str]:
    return [
        "sample_key",
        "DOI",
        "doi_url",
        "paper_title",
        "sample_id",
        "composition",
        "n_or_p",
        "np_basis_auto_step9",
        "paper_checked_step9",
        "paper_check_date_step9",
        "paper_check_scope_step9",
        "additive_manual_step9",
        "structure_manual_step9",
        "np_type_paper_manual_step9",
        "np_basis_paper_manual_step9",
        "rare_metal_note_manual_step9",
        "toxicity_note_manual_step9",
        "manual_review_note_step9",
        "needs_sintering_check_later_step9",
    ]


def numeric_value(value: Any, default: float = -math.inf) -> float:
    number = finite_float(value)
    return default if number is None else number


def build_high_priority_samples(
    sample_annotations: pd.DataFrame,
    top_n_review: int,
) -> pd.DataFrame:
    candidates = sample_annotations[
        sample_annotations["review_priority_tier_step9"].isin(["A", "B"])
    ].copy()
    candidates["_score"] = candidates["review_priority_score_step9"].map(lambda value: numeric_value(value, 0))
    candidates["_zt"] = candidates["zt_max_observed_step9"].map(lambda value: numeric_value(value, -math.inf))
    candidates["_points"] = candidates["valid_sigma_or_rho_point_count_step8"].map(
        lambda value: numeric_value(value, 0)
    )
    candidates = candidates.sort_values(
        ["_score", "_zt", "_points"], ascending=[False, False, False]
    ).head(top_n_review)
    candidates = candidates.drop(columns=["_score", "_zt", "_points"])
    return order_columns(candidates, high_priority_columns())


def build_manual_template(high_priority: pd.DataFrame) -> pd.DataFrame:
    template = high_priority.copy()
    for column in MANUAL_COLUMNS:
        if column not in template.columns:
            template[column] = ""
    template["paper_checked_step9"] = "no"
    template["paper_check_scope_step9"] = "not checked"
    template["np_checked_in_paper_step9"] = "no"
    template["needs_sintering_check_later_step9"] = "no"
    for column in [
        "paper_check_date_step9",
        "additive_manual_step9",
        "structure_manual_step9",
        "np_type_paper_manual_step9",
        "np_basis_paper_manual_step9",
        "rare_metal_note_manual_step9",
        "toxicity_note_manual_step9",
        "manual_review_note_step9",
    ]:
        template[column] = ""
    return order_columns(template, manual_template_columns())


def csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "doi_url" not in df.columns:
        return df
    columns = [column for column in df.columns if column != "doi_url"] + ["doi_url"]
    return df.loc[:, columns]


def review_group_key(row: pd.Series) -> str:
    doi = normalize_text(row.get("DOI", ""))
    if doi:
        return f"DOI:{doi}"
    sid = normalize_text(row.get("SID", ""))
    if sid:
        return f"SID:{sid}"
    title = normalize_text(row.get("paper_title", ""))
    return f"title:{title}" if title else f"sample:{row.get('sample_key', '')}"


def build_paper_review_queue(sample_annotations: pd.DataFrame) -> pd.DataFrame:
    frame = sample_annotations.copy()
    frame["review_group_key_step9"] = frame.apply(review_group_key, axis=1)
    records: list[dict[str, Any]] = []
    for group_key, group in frame.groupby("review_group_key_step9", sort=False):
        sorted_group = group.sort_values("review_priority_score_step9", ascending=False)
        max_score = int(pd.to_numeric(group["review_priority_score_step9"], errors="coerce").max())
        max_zt = pd.to_numeric(group["zt_max_observed_step9"], errors="coerce").max()
        n_or_p = group["n_or_p"].map(lambda value: normalize_text(value).casefold()) if "n_or_p" in group else pd.Series(dtype=str)
        record = {
            "review_group_key_step9": group_key,
            "DOI": first_nonempty(group["DOI"]) if "DOI" in group else "",
            "doi_url": first_nonempty(group["doi_url"]) if "doi_url" in group else "",
            "SID_list": ", ".join(sorted({normalize_text(value) for value in group.get("SID", []) if normalize_text(value)})),
            "paper_title": first_nonempty(group["paper_title"]) if "paper_title" in group else "",
            "year": first_nonempty(group["year"]) if "year" in group else "",
            "sample_count": len(group),
            "initial_tau_fit_sample_count": int(group["is_initial_tau_fit_candidate_step8"].map(normalize_bool).sum())
            if "is_initial_tau_fit_candidate_step8" in group
            else 0,
            "learning_candidate_sample_count": len(group),
            "max_review_priority_score_step9": max_score,
            "max_zt_observed_step9": "" if pd.isna(max_zt) else float(max_zt),
            "nanocarbon_sample_count": int(group["nanocarbon_keyword_detected_step9"].map(normalize_bool).sum()),
            "additive_keyword_sample_count": int(group["additive_keyword_detected_step9"].map(normalize_bool).sum()),
            "structure_keyword_sample_count": int(group["structure_keyword_detected_step9"].map(normalize_bool).sum()),
            "p_sample_count": int(n_or_p.eq("p").sum()),
            "n_sample_count": int(n_or_p.eq("n").sum()),
            "mixed_sample_count": int(n_or_p.eq("mixed").sum()),
            "unknown_sample_count": int(n_or_p.eq("unknown").sum()),
            "top_sample_keys": ", ".join(sorted_group["sample_key"].head(10).astype(str).tolist()),
            "review_priority_tier_step9": score_to_tier(max_score),
            "paper_review_reason_step9": f"max score={max_score}; samples={len(group)}",
            "paper_checked_step9": "no",
        }
        records.append(record)
    output = pd.DataFrame(records)
    if output.empty:
        return output
    return output.sort_values("max_review_priority_score_step9", ascending=False).reset_index(drop=True)


def write_review_instructions(output_dir: Path) -> None:
    text = """# Step9 literature review instructions

1. Open `manual_annotation_template_step9.csv`.
2. Open the paper from `doi_url` when available, or search by `paper_title`.
3. Check only additive/dopant/composite information, structure/morphology information, and explicit n/p-type evidence.
4. Do not check or fill sintering method in Step9.
5. Write dopants, additive elements, or composite components in `additive_manual_step9`.
6. Write nanostructure, thin film, bulk, grain size, CNT, graphene, layered structure, or related morphology in `structure_manual_step9`.
7. If the paper explicitly states the carrier type, write `n`, `p`, `mixed`, or `unknown` in `np_type_paper_manual_step9`.
8. Briefly write the paper evidence in `np_basis_paper_manual_step9`.
9. Set `paper_checked_step9` to `yes` after checking.
10. Save the completed file as `manual_annotation_template_step9_filled.csv`, then rerun this script with `--manual_annotations`.

Do not check sintering method at this stage. Check it later only for high-ZT samples, high-error samples, final-paper samples, or samples with the same composition but very different properties.
"""
    (output_dir / "step9_literature_review_instructions.md").write_text(text, encoding="utf-8")


def changed_rows(before: pd.DataFrame, after: pd.DataFrame, columns: list[str]) -> int:
    if len(before) != len(after):
        return max(len(before), len(after))
    changed = pd.Series(False, index=before.index)
    for column in columns:
        if column not in before.columns or column not in after.columns:
            continue
        changed = changed | before[column].map(normalize_text).ne(after[column].map(normalize_text))
    return int(changed.sum())


def sintering_invalid_rows(df: pd.DataFrame) -> int:
    count = 0
    for column, expected in [
        ("sintering_method", "unknown"),
        ("sintering_checked", "no"),
        ("record_checked", "no"),
    ]:
        if column not in df.columns:
            count += len(df)
        else:
            count += int(df[column].map(lambda value: normalize_text(value).casefold()).ne(expected).sum())
    return count


def value_counts_rows(prefix: str, series: pd.Series) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for value, count in series.fillna("").astype(str).value_counts(dropna=False).sort_index().items():
        rows.append((f"{prefix}_{value}_count", str(int(count))))
    return rows


def bool_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].map(normalize_bool).sum())


def build_report(
    inputs: dict[str, pd.DataFrame],
    sample_annotations: pd.DataFrame,
    learning_step9: pd.DataFrame,
    initial_step9: pd.DataFrame,
    review_step9: pd.DataFrame,
    candidate_curves_step9: pd.DataFrame,
    sigma_rho_curves_step9: pd.DataFrame,
    paper_queue: pd.DataFrame,
    high_priority: pd.DataFrame,
    manual_template: pd.DataFrame,
    parse_failures: dict[str, int],
    manual_used: bool,
    manual_reflected_count: int,
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    np_changed = sum(
        [
            changed_rows(inputs["learning"], learning_step9, NP_COLUMNS),
            changed_rows(inputs["initial"], initial_step9, NP_COLUMNS),
            changed_rows(inputs["review"], review_step9, NP_COLUMNS),
            changed_rows(inputs["candidate_curves"], candidate_curves_step9, NP_COLUMNS),
            changed_rows(inputs["sigma_rho_curves"], sigma_rho_curves_step9, NP_COLUMNS),
        ]
    )
    sintering_changed = sum(
        sintering_invalid_rows(df)
        for df in [
            sample_annotations,
            learning_step9,
            initial_step9,
            review_step9,
            candidate_curves_step9,
            sigma_rho_curves_step9,
        ]
    )

    zt_numeric = pd.to_numeric(sample_annotations["zt_max_observed_step9"], errors="coerce")
    rows: list[tuple[str, str]] = [
        ("input_learning_candidates_step8_rows", str(len(inputs["learning"]))),
        ("input_initial_tau_fit_candidates_step8_rows", str(len(inputs["initial"]))),
        ("input_review_needed_candidates_step8_rows", str(len(inputs["review"]))),
        ("input_candidate_core_curves_step8_rows", str(len(inputs["candidate_curves"]))),
        ("input_sigma_rho_curves_for_fitting_step8_rows", str(len(inputs["sigma_rho_curves"]))),
        ("output_sample_literature_annotations_step9_rows", str(len(sample_annotations))),
        ("output_learning_candidates_step9_rows", str(len(learning_step9))),
        ("output_initial_tau_fit_candidates_step9_rows", str(len(initial_step9))),
        ("output_review_needed_candidates_step9_rows", str(len(review_step9))),
        ("output_candidate_core_curves_step9_rows", str(len(candidate_curves_step9))),
        ("output_sigma_rho_curves_for_fitting_step9_rows", str(len(sigma_rho_curves_step9))),
        ("output_paper_review_queue_step9_rows", str(len(paper_queue))),
        ("output_high_priority_samples_for_manual_review_step9_rows", str(len(high_priority))),
        ("output_manual_annotation_template_step9_rows", str(len(manual_template))),
    ]
    rows.extend(value_counts_rows("review_priority_tier_step9", sample_annotations["review_priority_tier_step9"]))
    if not paper_queue.empty:
        rows.extend(value_counts_rows("paper_review_queue_tier", paper_queue["review_priority_tier_step9"]))
    rows.extend(
        [
            (
                "additive_keyword_detected_step9_true_sample_count",
                str(bool_count(sample_annotations, "additive_keyword_detected_step9")),
            ),
            (
                "structure_keyword_detected_step9_true_sample_count",
                str(bool_count(sample_annotations, "structure_keyword_detected_step9")),
            ),
            (
                "nanocarbon_keyword_detected_step9_true_sample_count",
                str(bool_count(sample_annotations, "nanocarbon_keyword_detected_step9")),
            ),
            (
                "rare_metal_flag_auto_step9_true_sample_count",
                str(bool_count(sample_annotations, "rare_metal_flag_auto_step9")),
            ),
            (
                "toxicity_flag_auto_step9_true_sample_count",
                str(bool_count(sample_annotations, "toxicity_flag_auto_step9")),
            ),
            ("zt_max_ge_1_sample_count", str(int((zt_numeric >= 1).sum()))),
            ("zt_max_ge_0_5_sample_count", str(int((zt_numeric >= 0.5).sum()))),
            (
                "paper_checked_step9_yes_sample_count",
                str(int(sample_annotations["paper_checked_step9"].map(lambda value: normalize_text(value).casefold()).eq("yes").sum())),
            ),
            (
                "paper_checked_step9_no_sample_count",
                str(int(sample_annotations["paper_checked_step9"].map(lambda value: normalize_text(value).casefold()).eq("no").sum())),
            ),
            ("np_changed_rows", str(np_changed)),
            ("sintering_changed_rows", str(sintering_changed)),
            ("manual_annotations_specified", str(manual_used)),
            ("manual_annotations_reflected_sample_count", str(manual_reflected_count)),
            ("zt_parse_failed_curve_count", str(parse_failures.get("zt_parse_failed_curve_count", 0))),
            (
                "thermal_parse_failed_curve_count",
                str(parse_failures.get("thermal_parse_failed_curve_count", 0)),
            ),
        ]
    )
    if "n_or_p" in sample_annotations.columns:
        rows.extend(value_counts_rows("n_or_p", sample_annotations["n_or_p"]))
    if "n_or_p_confidence_step6" in sample_annotations.columns:
        rows.extend(
            value_counts_rows(
                "n_or_p_confidence_step6", sample_annotations["n_or_p_confidence_step6"]
            )
        )
    for note in excel_notes:
        rows.append(("excel_note", note))
    report_df = pd.DataFrame(rows, columns=["metric", "value"])
    report_text = "\n".join(f"{metric}: {value}" for metric, value in rows) + "\n"
    return report_text, report_df


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
    sample_annotations: pd.DataFrame,
    learning_step9: pd.DataFrame,
    initial_step9: pd.DataFrame,
    review_step9: pd.DataFrame,
    paper_queue: pd.DataFrame,
    high_priority: pd.DataFrame,
    manual_template: pd.DataFrame,
    report_df: pd.DataFrame,
    excel_notes: list[str],
) -> None:
    sheets = {
        "sample_literature_annotations": excel_frame(
            sample_annotations, "sample_literature_annotations", excel_notes
        ),
        "learning_candidates": excel_frame(learning_step9, "learning_candidates", excel_notes),
        "initial_tau_fit_candidates": excel_frame(
            initial_step9, "initial_tau_fit_candidates", excel_notes
        ),
        "review_needed_candidates": excel_frame(review_step9, "review_needed_candidates", excel_notes),
        "paper_review_queue": excel_frame(paper_queue, "paper_review_queue", excel_notes),
        "high_priority_samples": excel_frame(high_priority, "high_priority_samples", excel_notes),
        "manual_annotation_template": excel_frame(
            manual_template, "manual_annotation_template", excel_notes
        ),
        "annotation_report": report_df,
    }
    path = output_dir / "starrydata2_step9_literature_annotations.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            fit_worksheet(writer, sheet_name, frame)


def assert_acceptance(
    inputs: dict[str, pd.DataFrame],
    sample_annotations: pd.DataFrame,
    learning_step9: pd.DataFrame,
    initial_step9: pd.DataFrame,
    review_step9: pd.DataFrame,
    candidate_curves_step9: pd.DataFrame,
    sigma_rho_curves_step9: pd.DataFrame,
    paper_queue: pd.DataFrame,
    manual_template: pd.DataFrame,
) -> None:
    if sample_annotations["sample_key"].duplicated().any():
        raise ValueError("sample_literature_annotations_step9 is not one row per sample_key")
    required = [
        "additive_auto_step9",
        "additive_manual_step9",
        "structure_auto_step9",
        "structure_manual_step9",
        "np_basis_auto_step9",
        "np_type_paper_manual_step9",
        "np_basis_paper_manual_step9",
        "paper_checked_step9",
    ]
    for column in required:
        if column not in sample_annotations.columns:
            raise KeyError(f"sample_literature_annotations_step9 missing {column}")

    checks = [
        (inputs["learning"], learning_step9, "learning_candidates_step9"),
        (inputs["initial"], initial_step9, "initial_tau_fit_candidates_step9"),
        (inputs["review"], review_step9, "review_needed_candidates_step9"),
        (inputs["candidate_curves"], candidate_curves_step9, "candidate_core_curves_step9"),
        (inputs["sigma_rho_curves"], sigma_rho_curves_step9, "sigma_rho_curves_for_fitting_step9"),
    ]
    for before, after, name in checks:
        if len(before) != len(after):
            raise ValueError(f"{name} row count changed")
    for frame_name, frame in [
        ("sample_literature_annotations_step9", sample_annotations),
        ("learning_candidates_step9", learning_step9),
        ("initial_tau_fit_candidates_step9", initial_step9),
        ("review_needed_candidates_step9", review_step9),
        ("candidate_core_curves_step9", candidate_curves_step9),
        ("sigma_rho_curves_for_fitting_step9", sigma_rho_curves_step9),
    ]:
        if sintering_invalid_rows(frame):
            raise ValueError(f"{frame_name} has non-standard sintering values")
    for column in ["x_values_json", "y_values_json"]:
        if column not in candidate_curves_step9.columns:
            raise KeyError(f"candidate_core_curves_step9 missing {column}")
        if column not in sigma_rho_curves_step9.columns:
            raise KeyError(f"sigma_rho_curves_for_fitting_step9 missing {column}")
    if not sigma_rho_curves_step9.empty:
        property_col = "property_step8" if "property_step8" in sigma_rho_curves_step9.columns else "property"
        properties = set(sigma_rho_curves_step9[property_col].dropna().astype(str).unique())
        bad = properties - {"Electrical conductivity", "Electrical resistivity"}
        if bad:
            raise ValueError(f"sigma_rho_curves_for_fitting_step9 has non sigma/rho properties: {bad}")
    if not paper_queue.empty:
        scores = pd.to_numeric(paper_queue["max_review_priority_score_step9"], errors="coerce")
        if not scores.is_monotonic_decreasing:
            raise ValueError("paper_review_queue_step9 is not sorted by score descending")
    if "paper_checked_step9" in manual_template.columns:
        if not manual_template["paper_checked_step9"].map(lambda value: normalize_text(value).casefold()).eq("no").all():
            raise ValueError("manual_annotation_template_step9 paper_checked_step9 is not initialized to no")


def write_csv_outputs(
    output_dir: Path,
    sample_annotations: pd.DataFrame,
    learning_step9: pd.DataFrame,
    initial_step9: pd.DataFrame,
    review_step9: pd.DataFrame,
    candidate_curves_step9: pd.DataFrame,
    sigma_rho_curves_step9: pd.DataFrame,
    paper_queue: pd.DataFrame,
    high_priority: pd.DataFrame,
    manual_template: pd.DataFrame,
    report_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_frame(sample_annotations).to_csv(
        output_dir / "sample_literature_annotations_step9.csv", index=False
    )
    csv_frame(learning_step9).to_csv(output_dir / "learning_candidates_step9.csv", index=False)
    csv_frame(initial_step9).to_csv(
        output_dir / "initial_tau_fit_candidates_step9.csv", index=False
    )
    csv_frame(review_step9).to_csv(output_dir / "review_needed_candidates_step9.csv", index=False)
    csv_frame(candidate_curves_step9).to_csv(
        output_dir / "candidate_core_curves_step9.csv", index=False
    )
    csv_frame(sigma_rho_curves_step9).to_csv(
        output_dir / "sigma_rho_curves_for_fitting_step9.csv", index=False
    )
    csv_frame(paper_queue).to_csv(output_dir / "paper_review_queue_step9.csv", index=False)
    csv_frame(high_priority).to_csv(
        output_dir / "high_priority_samples_for_manual_review_step9.csv", index=False
    )
    csv_frame(manual_template).to_csv(
        output_dir / "manual_annotation_template_step9.csv", index=False
    )
    (output_dir / "step9_literature_annotation_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    write_review_instructions(output_dir)


def main() -> None:
    args = parse_args()
    inputs = {
        label: read_csv_text(args.step8_dir / filename, required=(label != "sample_availability"))
        for label, filename in INPUT_FILES.items()
    }
    validate_inputs(inputs["learning"], inputs["candidate_curves"])
    for label, filename in INPUT_FILES.items():
        if not inputs[label].empty:
            ensure_sample_key(inputs[label], filename)

    base = build_base_sample_table(
        inputs["sample_availability"],
        inputs["learning"],
        inputs["initial"],
        inputs["review"],
    )
    property_stats, parse_failures = build_property_stats(inputs["candidate_curves"])
    all_annotations = add_step9_annotations(base, inputs["candidate_curves"], property_stats)
    all_annotations, manual_reflected_count, manual_used = apply_manual_annotations(
        all_annotations, args.manual_annotations
    )

    learning_keys = set(inputs["learning"]["sample_key"])
    sample_annotations = all_annotations[all_annotations["sample_key"].isin(learning_keys)].copy()
    sample_annotations = order_columns(sample_annotations, sample_annotation_columns())

    learning_step9 = merge_step9_columns(inputs["learning"], all_annotations)
    initial_step9 = merge_step9_columns(inputs["initial"], all_annotations)
    review_step9 = merge_step9_columns(inputs["review"], all_annotations)
    candidate_curves_step9 = merge_step9_columns(inputs["candidate_curves"], all_annotations)
    sigma_rho_curves_step9 = merge_step9_columns(inputs["sigma_rho_curves"], all_annotations)

    paper_queue = build_paper_review_queue(sample_annotations)
    high_priority = build_high_priority_samples(sample_annotations, args.top_n_review)
    manual_template = build_manual_template(high_priority)

    excel_notes: list[str] = []
    report_text, report_df = build_report(
        inputs,
        sample_annotations,
        learning_step9,
        initial_step9,
        review_step9,
        candidate_curves_step9,
        sigma_rho_curves_step9,
        paper_queue,
        high_priority,
        manual_template,
        parse_failures,
        manual_used,
        manual_reflected_count,
        excel_notes,
    )

    assert_acceptance(
        inputs,
        sample_annotations,
        learning_step9,
        initial_step9,
        review_step9,
        candidate_curves_step9,
        sigma_rho_curves_step9,
        paper_queue,
        manual_template,
    )

    write_csv_outputs(
        args.output_dir,
        sample_annotations,
        learning_step9,
        initial_step9,
        review_step9,
        candidate_curves_step9,
        sigma_rho_curves_step9,
        paper_queue,
        high_priority,
        manual_template,
        report_text,
    )
    write_excel_output(
        args.output_dir,
        sample_annotations,
        learning_step9,
        initial_step9,
        review_step9,
        paper_queue,
        high_priority,
        manual_template,
        report_df,
        excel_notes,
    )
    if excel_notes:
        report_text, report_df = build_report(
            inputs,
            sample_annotations,
            learning_step9,
            initial_step9,
            review_step9,
            candidate_curves_step9,
            sigma_rho_curves_step9,
            paper_queue,
            high_priority,
            manual_template,
            parse_failures,
            manual_used,
            manual_reflected_count,
            excel_notes,
        )
        (args.output_dir / "step9_literature_annotation_report.txt").write_text(
            report_text, encoding="utf-8"
        )

    np_changed = sum(
        [
            changed_rows(inputs["learning"], learning_step9, NP_COLUMNS),
            changed_rows(inputs["initial"], initial_step9, NP_COLUMNS),
            changed_rows(inputs["review"], review_step9, NP_COLUMNS),
            changed_rows(inputs["candidate_curves"], candidate_curves_step9, NP_COLUMNS),
            changed_rows(inputs["sigma_rho_curves"], sigma_rho_curves_step9, NP_COLUMNS),
        ]
    )
    sintering_changed = sum(
        sintering_invalid_rows(df)
        for df in [
            sample_annotations,
            learning_step9,
            initial_step9,
            review_step9,
            candidate_curves_step9,
            sigma_rho_curves_step9,
        ]
    )
    zt_numeric = pd.to_numeric(sample_annotations["zt_max_observed_step9"], errors="coerce")

    print("Done.")
    print("Created:")
    print("- sample_literature_annotations_step9.csv")
    print("- learning_candidates_step9.csv")
    print("- initial_tau_fit_candidates_step9.csv")
    print("- review_needed_candidates_step9.csv")
    print("- candidate_core_curves_step9.csv")
    print("- sigma_rho_curves_for_fitting_step9.csv")
    print("- paper_review_queue_step9.csv")
    print("- high_priority_samples_for_manual_review_step9.csv")
    print("- manual_annotation_template_step9.csv")
    print("- step9_literature_review_instructions.md")
    print("- step9_literature_annotation_report.txt")
    print("- starrydata2_step9_literature_annotations.xlsx")
    print("")
    print("Summary:")
    print(f"learning candidates: {len(learning_step9)}")
    print(f"initial tau fit candidates: {len(initial_step9)}")
    print(f"review needed candidates: {len(review_step9)}")
    print(f"paper review groups: {len(paper_queue)}")
    print(f"high priority manual review samples: {len(high_priority)}")
    print(
        "additive keyword detected samples: "
        f"{bool_count(sample_annotations, 'additive_keyword_detected_step9')}"
    )
    print(
        "structure keyword detected samples: "
        f"{bool_count(sample_annotations, 'structure_keyword_detected_step9')}"
    )
    print(
        "nanocarbon keyword detected samples: "
        f"{bool_count(sample_annotations, 'nanocarbon_keyword_detected_step9')}"
    )
    print(f"ZT >= 1 samples: {int((zt_numeric >= 1).sum())}")
    print(f"rare metal flag samples: {bool_count(sample_annotations, 'rare_metal_flag_auto_step9')}")
    print(f"toxicity flag samples: {bool_count(sample_annotations, 'toxicity_flag_auto_step9')}")
    print(
        "paper checked yes samples: "
        f"{int(sample_annotations['paper_checked_step9'].map(lambda value: normalize_text(value).casefold()).eq('yes').sum())}"
    )
    print(f"n/p changed rows: {np_changed}")
    print(f"sintering changed rows: {sintering_changed}")


if __name__ == "__main__":
    main()
