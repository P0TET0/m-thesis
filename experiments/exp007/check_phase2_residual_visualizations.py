"""Independently verify exp007 Phase 2 figures and summary tables."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
FIGURES = EXP007 / "figures" / "phase2"
REPORTS = EXP007 / "reports"
SOURCE = DATA / "phase1_sigma_residual_rows.csv"

STEMS = [
    "phase2_figure1_global_residual_overview",
    "phase2_figure2_material_group_profiles",
    "phase2_figure3_temperature_profiles",
    "phase2_figure4_oxide_focus",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        SOURCE,
        DATA / "phase2_material_visual_summary.csv",
        DATA / "phase2_temperature_visual_summary.csv",
        DATA / "phase2_oxide_visual_summary.csv",
        DATA / "phase2_visual_scope_summary.csv",
        DATA / "phase2_figure_manifest.csv",
        DATA / "phase2_input_manifest.csv",
        DATA / "phase2_validation_checks.csv",
        REPORTS / "phase2_residual_visualization.md",
        REPORTS / "phase2_validation_log.txt",
        *[FIGURES / f"{stem}{suffix}" for stem in STEMS for suffix in (".png", ".pdf")],
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 2 outputs: {missing}")

    source = pd.read_csv(SOURCE, low_memory=False)
    evaluable = source.loc[source["is_evaluable"].eq(True)]
    materials = pd.read_csv(DATA / "phase2_material_visual_summary.csv")
    temperatures = pd.read_csv(DATA / "phase2_temperature_visual_summary.csv")
    oxides = pd.read_csv(DATA / "phase2_oxide_visual_summary.csv")
    scope = pd.read_csv(DATA / "phase2_visual_scope_summary.csv")
    manifest = pd.read_csv(DATA / "phase2_figure_manifest.csv")
    validations = pd.read_csv(DATA / "phase2_validation_checks.csv")
    input_manifest = pd.read_csv(DATA / "phase2_input_manifest.csv")

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append((name, bool(passed), detail))

    check(
        "evaluable_source_count",
        len(evaluable) == 18968,
        f"rows={len(evaluable)}",
    )
    check(
        "material_rows_reconcile",
        int(materials["n_rows"].sum()) == len(evaluable),
        f"sum={int(materials['n_rows'].sum())}",
    )
    check(
        "temperature_rows_reconcile",
        int(temperatures["n_rows"].sum()) == len(evaluable),
        f"sum={int(temperatures['n_rows'].sum())}",
    )
    check(
        "oxide_rows_reconcile",
        int(oxides["n_rows"].sum()) == len(evaluable)
        and set(oxides["scope"]) == {"broad oxide", "non-oxide"},
        f"sum={int(oxides['n_rows'].sum())}",
    )
    scope_map = dict(zip(scope["metric"], scope["value"]))
    check(
        "scope_counts_match_source",
        int(scope_map["n_evaluable_rows"]) == len(evaluable)
        and int(scope_map["n_factor_10_outliers"])
        == int((evaluable["abs_residual_log10"] > 1).sum()),
        f"scope_rows={int(scope_map['n_evaluable_rows'])}",
    )
    check(
        "builder_validation_all_pass",
        validations["status"].eq("PASS").all(),
        f"failed={int(validations['status'].ne('PASS').sum())}",
    )

    expected_files = {
        (FIGURES / f"{stem}{suffix}").relative_to(ROOT).as_posix()
        for stem in STEMS
        for suffix in (".png", ".pdf")
    }
    check(
        "manifest_has_all_figures",
        set(manifest["figure_file"]) == expected_files and len(manifest) == 8,
        f"manifest_files={len(manifest)}",
    )
    manifest_hashes_ok = True
    for row in manifest.itertuples(index=False):
        path = ROOT / row.figure_file
        manifest_hashes_ok &= sha256(path) == row.sha256
        manifest_hashes_ok &= path.stat().st_size == int(row.size_bytes)
    check(
        "figure_hashes_and_sizes_match_manifest",
        manifest_hashes_ok,
        "all files checked",
    )

    png_checks = []
    for stem in STEMS:
        path = FIGURES / f"{stem}.png"
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            thumbnail = rgb.resize((240, 160))
            array = np.asarray(thumbnail, dtype=float)
            nonwhite = float(np.mean(np.min(array, axis=2) < 245))
            color_std = float(np.std(array))
        png_checks.append((width >= 2500 and height >= 1500, nonwhite > 0.03, color_std > 10))
    check(
        "png_dimensions_and_content",
        all(all(values) for values in png_checks),
        f"figures_checked={len(png_checks)}",
    )

    pdf_pages = []
    for stem in STEMS:
        pdf_pages.append(len(PdfReader(FIGURES / f"{stem}.pdf").pages))
    check(
        "pdf_files_open_and_have_one_page",
        pdf_pages == [1, 1, 1, 1],
        f"pages={pdf_pages}",
    )

    current_source_hash = sha256(SOURCE)
    check(
        "phase1_source_hash_matches_phase2_manifest",
        len(input_manifest) == 1
        and bool(input_manifest.iloc[0]["source_unchanged_during_run"])
        and current_source_hash == input_manifest.iloc[0]["sha256_before"]
        and current_source_hash == input_manifest.iloc[0]["sha256_after"],
        f"current={current_source_hash}",
    )
    report = (REPORTS / "phase2_residual_visualization.md").read_text(encoding="utf-8")
    check(
        "report_describes_all_four_figures",
        all(str(number) + "." in report for number in range(1, 5)),
        "figure list 1-4 present",
    )

    failed = [name for name, passed, _ in checks if not passed]
    lines = [
        "exp007 Phase 2 independent check",
        "=====================================",
        *[
            f"{'PASS' if passed else 'FAIL'} {name}: {detail}"
            for name, passed, detail in checks
        ],
        "",
        f"evaluable_rows={len(evaluable)}",
        f"figure_png_files={len(STEMS)}",
        f"figure_pdf_files={len(STEMS)}",
        f"Final status: {'PASS' if not failed else 'FAIL'}",
    ]
    output = "\n".join(lines) + "\n"
    (REPORTS / "phase2_check_log.txt").write_text(output, encoding="utf-8")
    print(output)
    if failed:
        raise RuntimeError(f"Phase 2 independent checks failed: {failed}")


if __name__ == "__main__":
    main()
