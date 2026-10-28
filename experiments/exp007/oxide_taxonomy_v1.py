"""Reusable composition-only oxide taxonomy v1 for exp007.

The taxonomy is deliberately limited to information available in the existing
material labels and parsed element sets.  It does not assign crystal structure
classes such as perovskite, delafossite, spinel, or layered cobaltite.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


TAXONOMY_VERSION = "oxide_v1"
DEFINITION_DATE = "2026-10-13"

ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
}

TRANSITION_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}

ORGANIC_PATTERNS = (
    r"poly\(",
    r"\bpolymer\b",
    r"\bpolyaniline\b",
    r"\bpedot\b",
    r"\bpss\b",
    r"\bpani\b",
    r"thiophene",
    r"\bswcnts?\b",
    r"\bmwcnts?\b",
    r"\bdmso\b",
    r"\bpedc\b",
    r"buopv",
)


RULES = [
    {
        "priority": 10,
        "rule_id": "OXV1_R010",
        "group_key": "oxide_v1::excluded_organic_polymer_or_carbon_composite",
        "display_name_en": "Excluded organic/polymer or carbon composite",
        "display_name_ja": "除外：有機・高分子・炭素複合系",
        "taxonomy_scope": "excluded_nonoxide_or_composite",
        "default_model_action": "exclude_from_inorganic_oxide_v1",
        "required_elements_all": "O",
        "required_elements_any": "",
        "label_condition": "organic/polymer/CNT keyword or C-H-O molecular notation",
        "rule_description": "Exclude O-containing organic, polymer, and carbon-composite labels before inorganic oxide classification.",
    },
    {
        "priority": 20,
        "rule_id": "OXV1_R020",
        "group_key": "oxide_v1::mixed_anion_O_S_Se_Te",
        "display_name_en": "Mixed-anion O + S/Se/Te",
        "display_name_ja": "混合アニオン：O＋S・Se・Te",
        "taxonomy_scope": "mixed_anion_O_containing",
        "default_model_action": "keep_as_separate_group",
        "required_elements_all": "O",
        "required_elements_any": "S;Se;Te",
        "label_condition": "none",
        "rule_description": "Separate oxychalcogenide-like compositions from ordinary oxides.",
    },
    {
        "priority": 30,
        "rule_id": "OXV1_R030",
        "group_key": "oxide_v1::mixed_anion_O_As_P",
        "display_name_en": "Mixed-anion O + As/P",
        "display_name_ja": "混合アニオン：O＋As・P",
        "taxonomy_scope": "mixed_anion_O_containing",
        "default_model_action": "keep_as_separate_group",
        "required_elements_all": "O",
        "required_elements_any": "As;P",
        "label_condition": "none",
        "rule_description": "Separate O-containing arsenide/phosphide compositions after organic/polymer exclusion.",
    },
    {
        "priority": 40,
        "rule_id": "OXV1_R040",
        "group_key": "oxide_v1::mixed_anion_O_N_or_halogen",
        "display_name_en": "Mixed-anion O + N/halogen",
        "display_name_ja": "混合アニオン：O＋N・ハロゲン",
        "taxonomy_scope": "mixed_anion_O_containing",
        "default_model_action": "manual_review_before_model",
        "required_elements_all": "O",
        "required_elements_any": "N;F;Cl;Br;I",
        "label_condition": "none",
        "rule_description": "Isolate O plus N/halogen entries because labels may represent oxynitrides, oxyhalides, dopants, or composites.",
    },
    {
        "priority": 50,
        "rule_id": "OXV1_R050",
        "group_key": "oxide_v1::Co_Mn_mixed_oxide",
        "display_name_en": "Co-Mn mixed oxide",
        "display_name_ja": "Co-Mn混合酸化物系",
        "taxonomy_scope": "inorganic_oxide",
        "default_model_action": "include_as_separate_group",
        "required_elements_all": "O;Co;Mn",
        "required_elements_any": "",
        "label_condition": "none",
        "rule_description": "Keep Co-Mn compositions separate before single-marker oxide rules.",
    },
    {
        "priority": 60,
        "rule_id": "OXV1_R060",
        "group_key": "oxide_v1::Co_containing_oxide",
        "display_name_en": "Co-containing oxide",
        "display_name_ja": "Co含有酸化物系",
        "taxonomy_scope": "inorganic_oxide",
        "default_model_action": "include",
        "required_elements_all": "O;Co",
        "required_elements_any": "",
        "label_condition": "none",
        "rule_description": "Assign remaining Co-containing compositions after mixed-anion and Co-Mn rules.",
    },
    {
        "priority": 70,
        "rule_id": "OXV1_R070",
        "group_key": "oxide_v1::Mn_containing_oxide",
        "display_name_en": "Mn-containing oxide",
        "display_name_ja": "Mn含有酸化物系",
        "taxonomy_scope": "inorganic_oxide",
        "default_model_action": "include",
        "required_elements_all": "O;Mn",
        "required_elements_any": "",
        "label_condition": "none",
        "rule_description": "Assign remaining Mn-containing compositions after mixed-anion and Co-Mn rules.",
    },
    {
        "priority": 80,
        "rule_id": "OXV1_R080",
        "group_key": "oxide_v1::Zn_containing_oxide",
        "display_name_en": "Zn-containing oxide",
        "display_name_ja": "Zn含有酸化物系",
        "taxonomy_scope": "inorganic_oxide",
        "default_model_action": "include",
        "required_elements_all": "O;Zn",
        "required_elements_any": "",
        "label_condition": "none",
        "rule_description": "Assign remaining Zn-containing compositions.",
    },
    {
        "priority": 90,
        "rule_id": "OXV1_R090",
        "group_key": "oxide_v1::Ti_containing_oxide",
        "display_name_en": "Ti-containing oxide",
        "display_name_ja": "Ti含有酸化物系",
        "taxonomy_scope": "inorganic_oxide",
        "default_model_action": "include",
        "required_elements_all": "O;Ti",
        "required_elements_any": "",
        "label_condition": "none",
        "rule_description": "Assign remaining Ti-containing compositions.",
    },
    {
        "priority": 100,
        "rule_id": "OXV1_R100",
        "group_key": "oxide_v1::Cu_containing_oxide",
        "display_name_en": "Cu-containing oxide",
        "display_name_ja": "Cu含有酸化物系",
        "taxonomy_scope": "inorganic_oxide",
        "default_model_action": "include",
        "required_elements_all": "O;Cu",
        "required_elements_any": "",
        "label_condition": "none",
        "rule_description": "Assign remaining Cu-containing compositions.",
    },
    {
        "priority": 110,
        "rule_id": "OXV1_R110",
        "group_key": "oxide_v1::In_Sn_Cd_containing_oxide",
        "display_name_en": "In/Sn/Cd-containing oxide",
        "display_name_ja": "In・Sn・Cd含有酸化物系",
        "taxonomy_scope": "inorganic_oxide",
        "default_model_action": "include",
        "required_elements_all": "O",
        "required_elements_any": "In;Sn;Cd",
        "label_condition": "none",
        "rule_description": "Pool remaining In-, Sn-, or Cd-containing compositions.",
    },
    {
        "priority": 120,
        "rule_id": "OXV1_R120",
        "group_key": "oxide_v1::Mo_W_V_Nb_containing_oxide",
        "display_name_en": "Mo/W/V/Nb-containing oxide",
        "display_name_ja": "Mo・W・V・Nb含有酸化物系",
        "taxonomy_scope": "inorganic_oxide",
        "default_model_action": "include",
        "required_elements_all": "O",
        "required_elements_any": "Mo;W;V;Nb",
        "label_condition": "none",
        "rule_description": "Pool remaining Mo-, W-, V-, or Nb-containing compositions.",
    },
    {
        "priority": 130,
        "rule_id": "OXV1_R130",
        "group_key": "oxide_v1::Fe_Ni_Cr_containing_oxide",
        "display_name_en": "Fe/Ni/Cr-containing oxide",
        "display_name_ja": "Fe・Ni・Cr含有酸化物系",
        "taxonomy_scope": "inorganic_oxide",
        "default_model_action": "include",
        "required_elements_all": "O",
        "required_elements_any": "Fe;Ni;Cr",
        "label_condition": "none",
        "rule_description": "Pool remaining Fe-, Ni-, or Cr-containing compositions.",
    },
    {
        "priority": 140,
        "rule_id": "OXV1_R140",
        "group_key": "oxide_v1::other_inorganic_O_containing",
        "display_name_en": "Other inorganic O-containing",
        "display_name_ja": "その他の無機O含有系",
        "taxonomy_scope": "inorganic_oxide",
        "default_model_action": "include_with_review_if_label_is_ambiguous",
        "required_elements_all": "O",
        "required_elements_any": "",
        "label_condition": "earlier rules did not match",
        "rule_description": "Catch remaining inorganic O-containing entries without making a structure claim.",
    },
    {
        "priority": 999,
        "rule_id": "OXV1_R999",
        "group_key": "oxide_v1::unclassified_outside_scope",
        "display_name_en": "Unclassified outside oxide-v1 scope",
        "display_name_ja": "分類不能：oxide-v1対象外",
        "taxonomy_scope": "outside_scope",
        "default_model_action": "exclude_and_investigate",
        "required_elements_all": "",
        "required_elements_any": "",
        "label_condition": "parsed elements do not contain O",
        "rule_description": "Fail-safe rule; no current broad-oxide row is expected here.",
    },
]


RULE_BY_ID = {rule["rule_id"]: rule for rule in RULES}


def rules_sha256() -> str:
    payload = json.dumps(RULES, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def parse_element_set(value: Any) -> set[str]:
    if value is None:
        return set()
    return {item for item in str(value).split(";") if item in ELEMENTS}


def organic_or_polymer_reason(label: str, elements: set[str]) -> str:
    lowered = label.casefold()
    for pattern in ORGANIC_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return f"label matches organic/polymer pattern: {pattern}"
    molecular = re.search(
        r"(?:^|\()C(?:\d+(?:\.\d+)?)?H(?:\d+(?:\.\d+)?)?",
        label,
    )
    if molecular and {"C", "H", "O"}.issubset(elements):
        return "C-H-O molecular notation"
    return ""


def is_formula_like(label: str) -> bool:
    if not label or re.search(r"\s|:|/|%", label):
        return False
    if not re.fullmatch(r"[A-Za-z0-9().+\-·∙×xδ]+", label):
        return False
    tokens = re.findall(r"[A-Z][a-z]?", label)
    if not tokens or any(token not in ELEMENTS for token in tokens):
        return False
    residual = re.sub(r"[A-Z][a-z]?", "", label)
    residual = re.sub(r"[0-9().+\-·∙×xδ]", "", residual)
    return residual == ""


def has_multicomponent_notation(label: str) -> bool:
    lowered = label.casefold()
    return bool(
        "/" in label
        or "+" in label
        or ":" in label
        or "wt%" in lowered
        or "vol%" in lowered
        or " composite" in lowered
        or "mixture" in lowered
        or re.search(r"\)\s*\d*(?:\.\d+)?\s*\(", label)
    )


def choose_rule(label: str, elements: set[str]) -> tuple[dict[str, Any], str]:
    organic_reason = organic_or_polymer_reason(label, elements)
    if "O" not in elements:
        return RULE_BY_ID["OXV1_R999"], "parsed element set does not contain O"
    if organic_reason:
        return RULE_BY_ID["OXV1_R010"], organic_reason
    if elements & {"S", "Se", "Te"}:
        return RULE_BY_ID["OXV1_R020"], "contains O and S/Se/Te"
    if elements & {"As", "P"}:
        return RULE_BY_ID["OXV1_R030"], "contains O and As/P"
    if elements & {"N", "F", "Cl", "Br", "I"}:
        return RULE_BY_ID["OXV1_R040"], "contains O and N/halogen"
    if {"Co", "Mn"}.issubset(elements):
        return RULE_BY_ID["OXV1_R050"], "contains O, Co, and Mn"
    if "Co" in elements:
        return RULE_BY_ID["OXV1_R060"], "contains O and Co after higher-priority rules"
    if "Mn" in elements:
        return RULE_BY_ID["OXV1_R070"], "contains O and Mn after higher-priority rules"
    if "Zn" in elements:
        return RULE_BY_ID["OXV1_R080"], "contains O and Zn after higher-priority rules"
    if "Ti" in elements:
        return RULE_BY_ID["OXV1_R090"], "contains O and Ti after higher-priority rules"
    if "Cu" in elements:
        return RULE_BY_ID["OXV1_R100"], "contains O and Cu after higher-priority rules"
    if elements & {"In", "Sn", "Cd"}:
        return RULE_BY_ID["OXV1_R110"], "contains O and In/Sn/Cd after higher-priority rules"
    if elements & {"Mo", "W", "V", "Nb"}:
        return RULE_BY_ID["OXV1_R120"], "contains O and Mo/W/V/Nb after higher-priority rules"
    if elements & {"Fe", "Ni", "Cr"}:
        return RULE_BY_ID["OXV1_R130"], "contains O and Fe/Ni/Cr after higher-priority rules"
    return RULE_BY_ID["OXV1_R140"], "contains O; no earlier v1 rule matched"


def classify_oxide_v1(label_value: Any, parsed_elements_value: Any) -> dict[str, Any]:
    label = normalize_label(label_value)
    elements = parse_element_set(parsed_elements_value)
    rule, match_basis = choose_rule(label, elements)
    formula_like = is_formula_like(label)
    multicomponent = has_multicomponent_notation(label)
    transition_elements = sorted(elements & TRANSITION_ELEMENTS)

    review_reasons: list[str] = []
    if rule["rule_id"] == "OXV1_R010":
        inclusion_status = "exclude"
        confidence = "low"
        review_priority = "high"
        review_reasons.append("confirm exclusion from inorganic oxide taxonomy")
    elif rule["rule_id"] == "OXV1_R999":
        inclusion_status = "exclude"
        confidence = "low"
        review_priority = "high"
        review_reasons.append("unexpected absence of parsed O")
    elif rule["rule_id"] == "OXV1_R040":
        inclusion_status = "review"
        confidence = "medium"
        review_priority = "medium"
        review_reasons.append("O+N/halogen may represent an oxynitride, oxyhalide, dopant, or composite")
    elif not formula_like:
        inclusion_status = "review"
        confidence = "medium"
        review_priority = "medium"
        review_reasons.append("label is not a strict formula-like string")
    elif multicomponent:
        inclusion_status = "review"
        confidence = "medium"
        review_priority = "low"
        review_reasons.append("multi-component notation requires manual confirmation")
    else:
        inclusion_status = "include"
        confidence = "high"
        review_priority = "none"

    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_rules_sha256": rules_sha256(),
        "oxide_v1_rule_id": rule["rule_id"],
        "oxide_v1_rule_priority": rule["priority"],
        "oxide_v1_group_key": rule["group_key"],
        "oxide_v1_display_name_en": rule["display_name_en"],
        "oxide_v1_display_name_ja": rule["display_name_ja"],
        "oxide_v1_scope": rule["taxonomy_scope"],
        "oxide_v1_default_model_action": rule["default_model_action"],
        "oxide_v1_match_basis": match_basis,
        "oxide_v1_inclusion_status": inclusion_status,
        "oxide_v1_default_model_eligible": inclusion_status == "include",
        "oxide_v1_classification_confidence": confidence,
        "oxide_v1_review_priority": review_priority,
        "oxide_v1_review_reasons": "; ".join(review_reasons),
        "oxide_v1_formula_like_label": formula_like,
        "oxide_v1_multicomponent_notation": multicomponent,
        "oxide_v1_transition_elements": ";".join(transition_elements),
        "oxide_v1_transition_element_count": len(transition_elements),
        "oxide_v1_structural_class": "not_assigned_formula_only_v1",
    }

