"""Conservative, evidence-located host candidates from paper titles/abstracts.

No file/network access, classification decisions, coefficient thresholds, or
composition edits occur here. Offsets refer to the original input string, with
Python's zero-based, end-exclusive indexing. Each result is a candidate only.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass


ELEMENTS = set("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split())
NAMES = dict(zip(
    "hydrogen helium lithium beryllium boron carbon nitrogen oxygen fluorine neon sodium magnesium aluminium silicon phosphorus sulfur chlorine argon potassium calcium scandium titanium vanadium chromium manganese iron cobalt nickel copper zinc gallium germanium arsenic selenium bromine krypton rubidium strontium yttrium zirconium niobium molybdenum technetium ruthenium rhodium palladium silver cadmium indium tin antimony tellurium iodine xenon caesium barium lanthanum cerium praseodymium neodymium promethium samarium europium gadolinium terbium dysprosium holmium erbium thulium ytterbium lutetium hafnium tantalum tungsten rhenium osmium iridium platinum gold mercury thallium lead bismuth polonium astatine radon francium radium actinium thorium protactinium uranium".split(),
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U".split(),
))
NAMES.update({"aluminum": "Al", "sulphur": "S", "cesium": "Cs", "oxide": "O", "sulfide": "S", "sulphide": "S", "selenide": "Se", "telluride": "Te", "nitride": "N", "phosphide": "P", "arsenide": "As", "antimonide": "Sb", "silicide": "Si", "germanide": "Ge", "boride": "B", "carbide": "C", "fluoride": "F", "chloride": "Cl", "bromide": "Br", "iodide": "I"})
ANION_NAMES = {n for n in NAMES if n.endswith("ide")}
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓᵧ", "0123456789xy")
NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
VARIABLE = r"[xyzδεϵΔηαβγ]"
COEFFICIENT = rf"(?:(?:{NUMBER})?{VARIABLE}|{NUMBER})?(?:[+±-](?:{NUMBER})?{VARIABLE})*"
UNIT = rf"[A-Z][a-z]?{COEFFICIENT}"
GROUP = rf"\((?:{UNIT})+(?:\s*,\s*(?:{UNIT})+)*\){COEFFICIENT}"
FORMULA_BODY = rf"(?:{UNIT}|{GROUP})+"
FORMULA = re.compile(rf"(?<![A-Za-z0-9]){FORMULA_BODY}(?![A-Za-z0-9])")
TOKEN = re.compile(rf"([A-Z][a-z]?)({COEFFICIENT})")
ALLOY = re.compile(rf"(?<![A-Za-z0-9]){UNIT}(?:-{UNIT})+(?![A-Za-z0-9])")
NAME_WORD = "(?:" + "|".join(sorted(NAMES, key=len, reverse=True)) + ")"
NAME_PATTERN = re.compile(rf"\b{NAME_WORD}(?:[-\s]+{NAME_WORD})*\b", re.I)


@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    formula: str | None
    elements: tuple[str, ...]
    kind: str
    raw: str


def _normalize(value: str) -> tuple[str, list[int], list[int]]:
    """Remove markup and normalize typography, preserving original offsets."""
    chars, starts, ends = [], [], []
    for match in re.finditer(r"<[^>]*>|&(?:#\d+|#x[0-9a-fA-F]+|\w+);|[^<&]|[<&]", value):
        raw = match.group()
        if raw.startswith("<") and raw.endswith(">"):
            continue
        decoded = html.unescape(raw).translate(SUBSCRIPTS)
        for char in decoded:
            chars.append("-" if char in "−–—‐‑" else char)
            starts.append(match.start())
            ends.append(match.end())
    return "".join(chars), starts, ends


def _formula_elements(value: str) -> tuple[str, ...] | None:
    tokens = list(TOKEN.finditer(value))
    if not tokens or not re.fullmatch(FORMULA_BODY, value):
        return None
    if any(m.group(1) not in ELEMENTS for m in tokens):
        return None
    return tuple(dict.fromkeys(m.group(1) for m in tokens))


def _unparsed_coefficient_suffix(text: str, end: int) -> bool:
    """Reject partial formulas rather than silently making a fixed composition.

    Unsupported variable names, coefficient arithmetic, and fractional variable
    coefficients must not turn ``TiO2-delta`` or ``TiO2-delta/2`` into ``TiO2``.
    A coefficient separated by markup whitespace, as in ``Sr(Ti,Fe)O 3- δ``,
    also invalidates the partial mention instead of inventing oxygen ratio 1.
    Ordinary prose suffixes such as ``-based`` remain allowed.
    """
    tail = text[end:].lstrip()
    return bool(re.match(
        r"(?:\d|\.\d)|(?:[+±-]\s*(?:\d|\.\d|[α-ωΑ-Ωϵ]|(?:delta|epsilon)\b|[a-z](?![a-z])|\(\s*(?:\d|[a-zα-ωΑ-Ω])))"
        r"|(?:/\s*(?:\d|[xyzδεϵΔ]))|(?:[α-ωΑ-Ωϵ])",
        tail,
    ))


def _mentions(text: str) -> list[Mention]:
    result = []
    for match in ALLOY.finditer(text):
        if _unparsed_coefficient_suffix(text, match.end()):
            continue
        parts = match.group().split("-")
        elements = [_formula_elements(part) for part in parts]
        # An elemental series such as Si-Ge, not PbTe-PbS or variable formulas.
        if all(part and len(part) == 1 for part in elements):
            unique = tuple(dict.fromkeys(e for part in elements for e in part))
            if len(unique) > 1:
                result.append(Mention(match.start(), match.end(), match.group(), unique, "element_family", match.group()))
    for match in FORMULA.finditer(text):
        if any(m.start <= match.start() and m.end >= match.end() for m in result):
            continue
        raw = match.group()
        if _unparsed_coefficient_suffix(text, match.end()):
            continue
        elements = _formula_elements(raw)
        if not elements:
            continue
        # Reject acronyms such as SPS, BCS, PSS, and English In/As/At in prose.
        if len(elements) > 1 and not re.search(r"[a-z0-9]", raw):
            continue
        if len(elements) == 1 and raw not in ELEMENTS and not re.search(r"\d", raw):
            continue
        kind = "variable_formula" if re.search(VARIABLE, raw) else "grouped_formula" if "(" in raw else "formula"
        result.append(Mention(match.start(), match.end(), raw, elements, kind, raw))
    for match in NAME_PATTERN.finditer(text):
        words = re.findall(r"[a-z]+", match.group().lower())
        if len(words) == 1 and words[0] in ANION_NAMES:
            continue
        if words == ["lead"] and re.match(r"\s+to\b", text[match.end():], re.I):
            continue
        elements = tuple(dict.fromkeys(NAMES[word] for word in words))
        result.append(Mention(match.start(), match.end(), None, elements, "chemical_name", match.group()))
    return sorted(result, key=lambda m: (m.start, -m.end))


def candidates_from_text(title: str | None, abstract_text: str | None) -> list[dict]:
    """Return all supported candidates; never choose a paper's single host.

    Explicit ``X-doped Y``, ``Y doped with X``, ``Y-based``, and ``Y alloys`` /
    ``Y solid solutions`` relations are sought in both sources. Other chemical
    mentions in a title are weaker candidates. An abstract-only bare mention
    is not enough. Chemical names without stoichiometry have host_formula=None.
    Empty lists mean no supported candidate, not absence of a host material.
    """
    output = []
    for source, original in (("title", title or ""), ("abstract", abstract_text or "")):
        text, starts, ends = _normalize(original)
        if not text:
            continue
        mentions = _mentions(text)
        proposed = []
        explicit_dopants = set()
        common_cautions = []
        if re.search(r"\b(?:(?:nano)?composites?|heterostructures?|multiphase|(?:nano)?inclusions?|nanoparticles?|nanostructures?|interfaces?|superlattices?|hybrids?|templates?)\b", text, re.I):
            common_cautions.append("composite_or_multiphase_context")
        if re.search(r"\b(?:review|perspective|overview|comparison|comparative|versus|vs\.?|survey)\b", text, re.I):
            common_cautions.append("review_or_comparison_context")
        for event in re.finditer(r"\b(?:co-?)?doped\b", text, re.I):
            before = [m for m in mentions if m.end <= event.start() and event.start() - m.end < 35]
            after = [m for m in mentions if m.start >= event.end() and m.start - event.end() < 80]
            if not before or not after:
                continue
            left, right = before[-1], after[0]
            gap_before, gap_after = text[left.end:event.start()], text[event.end():right.start]
            if re.fullmatch(r"[\s-]*", gap_before) and re.fullmatch(r"\s+with\s+", gap_after, re.I):
                host, dopant, method = left, right, "host_doped_with_element"
            elif re.fullmatch(r"[\s-]*", gap_before) and re.fullmatch(r"[\s-]*(?:(?:n-type|p-type|nanostructured|nanocrystalline|polycrystalline|single-crystal|bulk)\s+)*", gap_after, re.I):
                host, dopant, method = right, left, "element_doped_host"
            else:
                continue
            # Multiple-element additives/compounds require a separate phase model.
            dopants = list(dopant.elements) if len(dopant.elements) == 1 else []
            cautions = [] if dopants else ["dopant_or_additive_is_compound"]
            proposed.append((host, dopants, method, left.start, right.end, cautions))
            explicit_dopants.add((dopant.start, dopant.end))
        for mention in mentions:
            tail = text[mention.end:mention.end + 50]
            match = re.match(r"[\s-]+based\b", tail, re.I)
            if match:
                proposed.append((mention, [], "host_based", mention.start, mention.end + match.end(), []))
            match = re.match(r"[\s-]+(?:alloys?|solid[\s-]+solutions?)\b", tail, re.I)
            if match:
                proposed.append((mention, [], "alloy_or_solid_solution", mention.start, mention.end + match.end(), []))
        if source == "title":
            explicit_hosts = {(m.start, m.end) for m, *_ in proposed}
            for mention in mentions:
                key = (mention.start, mention.end)
                if key in explicit_hosts or key in explicit_dopants:
                    continue
                # Bare single-symbol titles are often prepositions or acronyms.
                if mention.kind == "formula" and len(mention.elements) == 1 and not re.search(r"\d", mention.raw):
                    continue
                proposed.append((mention, [], "title_material_mention", mention.start, mention.end, ["material_role_not_explicit"]))
        distinct_hosts = {(m.formula, m.elements) for m, *_ in proposed}
        for mention, dopants, method, start, end, cautions in proposed:
            notes = list(common_cautions) + cautions
            if len(distinct_hosts) > 1:
                notes.append("multiple_material_candidates")
            if mention.kind == "variable_formula":
                notes.append("variable_composition")
            if mention.kind == "element_family":
                notes.append("element_family_not_stoichiometry")
            if mention.kind == "chemical_name":
                notes.append("name_stoichiometry_unspecified")
            if set(dopants) & set(mention.elements):
                notes.append("explicit_dopant_also_in_host_formula")
            a, b = starts[start], ends[end - 1]
            ha, hb = starts[mention.start], ends[mention.end - 1]
            output.append({
                "candidate_only": True,
                "source": source,
                "method": method,
                "host_formula": mention.formula,
                "host_name": mention.raw,
                "host_elements": list(mention.elements),
                "dopant_elements": dopants,
                "formula_kind": mention.kind,
                "span": [a, b],
                "host_span": [ha, hb],
                "matched_text": original[a:b],
                "cautions": sorted(set(notes)),
            })
    return output
