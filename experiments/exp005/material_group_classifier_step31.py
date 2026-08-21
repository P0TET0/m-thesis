import json
import math
import re
from collections import Counter
from typing import Any, Iterable

import numpy as np
import pandas as pd


ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
    "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}
ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")
LOWERCASE_WORD_RE = re.compile(r"[a-z]{3,}")
SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

HOST_PRIORITY = [
    "host::BiSbTe",
    "host::PbTe",
    "host::SnTe",
    "host::GeTe",
    "host::BiTe",
    "host::SbTe",
    "host::SiGe",
    "host::Mg2SiSn",
    "host::CoSb_skutterudite",
    "host::oxide",
    "host::sulfide",
    "host::selenide",
    "host::telluride",
    "host::other_formula_system",
]
BROAD_HOST_KEYS = {
    "host::oxide",
    "host::sulfide",
    "host::selenide",
    "host::telluride",
    "host::other_formula_system",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def extract_elements_from_text(value: Any) -> set[str]:
    return {
        symbol
        for symbol in ELEMENT_RE.findall(normalize_text(value))
        if symbol in ELEMENT_SYMBOLS
    }


def looks_like_prose(value: Any) -> bool:
    text = normalize_text(value)
    if not text:
        return False
    if text.casefold() in {"unknown", "undefined", "not available", "n/a"}:
        return True
    return bool(LOWERCASE_WORD_RE.search(text)) and (" " in text or "-doped" in text.casefold())


def _parse_number(text: str, index: int) -> tuple[float, int]:
    match = NUMBER_RE.match(text, index)
    if not match:
        return 1.0, index
    return float(match.group()), match.end()


def _parse_sequence(text: str, index: int, stop_at_parenthesis: bool) -> tuple[Counter[str], int, bool, bool]:
    totals: Counter[str] = Counter()
    complete = True
    saw_token = False
    while index < len(text):
        char = text[index]
        if char == ")":
            if stop_at_parenthesis:
                return totals, index + 1, complete, saw_token
            complete = False
            index += 1
            continue
        if char == "(":
            inner, next_index, inner_complete, inner_saw = _parse_sequence(text, index + 1, True)
            if next_index <= index + 1:
                complete = False
                index += 1
                continue
            multiplier, index = _parse_number(text, next_index)
            for element, amount in inner.items():
                totals[element] += amount * multiplier
            complete = complete and inner_complete
            saw_token = saw_token or inner_saw
            continue
        if char.isupper():
            symbol_match = re.match(r"[A-Z][a-z]?", text[index:])
            assert symbol_match is not None
            symbol = symbol_match.group()
            index += len(symbol)
            if symbol not in ELEMENT_SYMBOLS:
                complete = False
                continue
            amount, index = _parse_number(text, index)
            totals[symbol] += amount
            saw_token = True
            continue
        if char in "[]{}·+*/:;,_- ":
            if char in ",;-":
                complete = False
            index += 1
            continue
        if char.isdigit() or char == ".":
            # A leading phase coefficient cannot be applied reliably without a
            # full phase decomposition, so preserve elements but lower confidence.
            number_match = NUMBER_RE.match(text, index)
            index = number_match.end() if number_match else index + 1
            complete = False
            continue
        complete = False
        index += 1
    if stop_at_parenthesis:
        complete = False
    return totals, index, complete, saw_token


def parse_formula_stoichiometry(value: Any) -> tuple[dict[str, float], bool]:
    text = normalize_text(value).translate(SUBSCRIPT_TRANSLATION)
    if not text or looks_like_prose(text):
        return {}, False
    text = re.sub(r"\s+", "", text)
    totals, _, complete, saw_token = _parse_sequence(text, 0, False)
    totals = Counter({element: amount for element, amount in totals.items() if amount > 0 and math.isfinite(amount)})
    if not saw_token or not totals:
        return {}, False
    return dict(totals), complete


def _elements_from_name_or_family(*values: Any) -> tuple[set[str], str]:
    for label, value in zip(("material_name_raw", "material_family_raw"), values):
        text = normalize_text(value)
        if not text or text.casefold() in {"unknown", "undefined", "other"}:
            continue
        elements = extract_elements_from_text(text)
        if elements:
            return elements, label
    return set(), ""


def parse_material_row(
    formula_raw: Any,
    material_name_raw: Any,
    material_family_raw: Any,
    dopant_threshold: float,
) -> dict[str, Any]:
    formula_text = normalize_text(formula_raw)
    amounts, complete = parse_formula_stoichiometry(formula_text)
    parse_source = "formula_raw" if amounts else ""
    fractions: dict[str, float] = {}
    if amounts and complete:
        total = sum(amounts.values())
        fractions = {element: amount / total for element, amount in amounts.items()}
        elements = set(fractions)
        parse_status = "stoichiometry_parsed"
    elif amounts and not looks_like_prose(formula_text):
        elements = set(amounts)
        parse_status = "element_presence_fallback"
    else:
        elements, parse_source = _elements_from_name_or_family(
            material_name_raw, material_family_raw
        )
        if elements:
            parse_status = "material_name_fallback"
        else:
            elements = set()
            parse_status = "parse_failed"

    if fractions:
        major = {element for element, fraction in fractions.items() if fraction >= dopant_threshold}
        dopants = set(fractions) - major
    else:
        major = set(elements)
        dopants = set()
    if elements and not major:
        major = set(elements)
        dopants = set()
        parse_status = "element_presence_fallback"

    parsed_elements = sorted(elements)
    major_elements = sorted(major)
    dopant_elements = sorted(dopants)
    fraction_json = (
        json.dumps({element: round(fractions[element], 8) for element in sorted(fractions)}, ensure_ascii=False)
        if fractions
        else ""
    )
    system_key = "system::" + "-".join(major_elements) if major_elements else ""
    host_key, candidates, host_reason = classify_host_family(set(major_elements))
    multiple = len(candidates) > 1
    reason_parts = [host_reason]
    if dopant_elements:
        reason_parts.append(
            f"{', '.join(dopant_elements)} below dopant threshold {dopant_threshold:.3g}"
        )
    if parse_status == "element_presence_fallback":
        reason_parts.append("stoichiometry uncertain; element-presence fallback used")
    elif parse_status == "material_name_fallback":
        reason_parts.append(f"elements inferred from {parse_source}")
    elif parse_status == "parse_failed":
        reason_parts = ["formula and material-name parse failed"]
    return {
        "parsed_elements": ";".join(parsed_elements),
        "element_fraction": fraction_json,
        "major_elements": ";".join(major_elements),
        "dopant_elements": ";".join(dopant_elements),
        "major_element_system_key": system_key,
        "host_family_key": host_key,
        "multiple_host_candidate": multiple,
        "host_candidate_keys": ";".join(candidates),
        "parse_status": parse_status,
        "parse_source": parse_source,
        "classification_reason": "; ".join(part for part in reason_parts if part),
    }


def classify_host_family(major: set[str]) -> tuple[str, list[str], str]:
    specific: list[str] = []
    rules = [
        ("host::BiSbTe", {"Bi", "Sb", "Te"}.issubset(major)),
        ("host::PbTe", {"Pb", "Te"}.issubset(major)),
        ("host::SnTe", {"Sn", "Te"}.issubset(major)),
        ("host::GeTe", {"Ge", "Te"}.issubset(major)),
        ("host::BiTe", {"Bi", "Te"}.issubset(major)),
        ("host::SbTe", {"Sb", "Te"}.issubset(major)),
        ("host::SiGe", {"Si", "Ge"}.issubset(major)),
        ("host::Mg2SiSn", "Mg" in major and bool({"Si", "Sn"} & major)),
        ("host::CoSb_skutterudite", {"Co", "Sb"}.issubset(major)),
    ]
    specific = [key for key, matched in rules if matched]
    if specific:
        chosen = specific[0]
        reason = f"{', '.join(sorted(major))} are major elements"
        if len(specific) > 1:
            reason += f"; multiple host candidates {', '.join(specific)}; priority selected {chosen}"
        return chosen, specific, reason

    broad_rules = [
        ("host::oxide", "O" in major),
        ("host::sulfide", "S" in major),
        ("host::selenide", "Se" in major),
        ("host::telluride", "Te" in major),
    ]
    broad = [key for key, matched in broad_rules if matched]
    if broad:
        chosen = broad[0]
        reason = f"{', '.join(sorted(major))} are major elements"
        if len(broad) > 1:
            reason += f"; multiple broad host candidates {', '.join(broad)}; priority selected {chosen}"
        return chosen, broad, reason
    if major:
        return (
            "host::other_formula_system",
            ["host::other_formula_system"],
            f"{', '.join(sorted(major))} are major elements; no named host rule matched",
        )
    return "unknown_material_group", [], "formula parse failed"


def _paper_key(row: pd.Series) -> str:
    paper_id = normalize_text(row.get("paper_id", ""))
    if paper_id:
        return paper_id
    sample_key = normalize_text(row.get("sample_key", ""))
    if "__sample_" in sample_key:
        return sample_key.split("__sample_", 1)[0]
    return sample_key


def add_material_group_keys(
    data: pd.DataFrame,
    dopant_threshold: float,
    min_rows_per_group: int,
    min_samples_per_group: int,
    min_papers_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = data.copy()
    required_raw = ["formula_raw", "material_name_raw", "material_family_raw"]
    for column in required_raw:
        if column not in output.columns:
            output[column] = ""
    unique_raw = output[required_raw].drop_duplicates().reset_index(drop=True)
    parsed_records = [
        parse_material_row(
            row.formula_raw,
            row.material_name_raw,
            row.material_family_raw,
            dopant_threshold,
        )
        for row in unique_raw.itertuples(index=False)
    ]
    unique_classified = pd.concat([unique_raw, pd.DataFrame(parsed_records)], axis=1)
    output = output.merge(unique_classified, on=required_raw, how="left", validate="many_to_one")
    output["paper_key_step31"] = output.apply(_paper_key, axis=1)

    valid_system = output["major_element_system_key"].ne("")
    system_counts = (
        output.loc[valid_system]
        .groupby("major_element_system_key", sort=True)
        .agg(
            row_count=("major_element_system_key", "size"),
            sample_count=("sample_key", "nunique"),
            paper_count=("paper_key_step31", lambda values: len({value for value in values if normalize_text(value)})),
        )
        .reset_index()
    )
    system_counts["meets_group_thresholds"] = (
        system_counts["row_count"].ge(min_rows_per_group)
        & system_counts["sample_count"].ge(min_samples_per_group)
        & system_counts["paper_count"].ge(min_papers_per_group)
    )
    threshold_map = system_counts.set_index("major_element_system_key")["meets_group_thresholds"].to_dict()

    final_keys: list[str] = []
    levels: list[str] = []
    reasons: list[str] = []
    for row in output.itertuples(index=False):
        system_key = normalize_text(row.major_element_system_key)
        host_key = normalize_text(row.host_family_key)
        base_reason = normalize_text(row.classification_reason)
        if system_key and threshold_map.get(system_key, False):
            final_key = system_key
            level = "major_element_system_used"
            extra_reason = "major-element system met row/sample/paper thresholds"
        elif host_key == "unknown_material_group" or not host_key:
            final_key = "unknown_material_group"
            level = "unknown"
            extra_reason = "no usable host family"
        elif host_key in BROAD_HOST_KEYS:
            final_key = host_key
            level = "backed_off_to_broad_family"
            extra_reason = "major-element system count too small; backed off to broad family"
        else:
            final_key = host_key
            level = "backed_off_to_host_family"
            extra_reason = "major-element system count too small; backed off to host family"
        final_keys.append(final_key)
        levels.append(level)
        reasons.append(f"{base_reason}; {extra_reason}" if base_reason else extra_reason)
    output["final_material_group_key"] = final_keys
    output["group_backoff_level"] = levels
    output["classification_reason"] = reasons
    return output, system_counts


def _joined_examples(values: Iterable[Any], limit: int = 5) -> str:
    unique: list[str] = []
    for value in values:
        text = normalize_text(value)
        if text and text not in unique:
            unique.append(text)
        if len(unique) >= limit:
            break
    return " | ".join(unique)


def build_material_group_counts(data: pd.DataFrame) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for key_type in ["host_family_key", "major_element_system_key", "final_material_group_key"]:
        subset = data[data[key_type].map(normalize_text).ne("")].copy()
        if subset.empty:
            continue
        block = (
            subset.groupby(key_type, sort=True, dropna=False)
            .agg(
                row_count=(key_type, "size"),
                sample_count=("sample_key", "nunique"),
                paper_count=("paper_key_step31", lambda values: len({value for value in values if normalize_text(value)})),
                carrier_type_values=("carrier_type", lambda values: ";".join(sorted({normalize_text(value) for value in values if normalize_text(value)}))),
                T_min_K=("T_K", "min"),
                T_max_K=("T_K", "max"),
                formula_examples=("formula_raw", _joined_examples),
                material_name_examples=("material_name_raw", _joined_examples),
            )
            .reset_index()
            .rename(columns={key_type: "material_group_key"})
        )
        block.insert(0, "classification_key_type", key_type)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
