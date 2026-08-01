import argparse
import re
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = EXP_DIR / "data" / "processed" / "step9b_ct_vs_pred_25k_np_split"
DEFAULT_PREDICTIONS = DEFAULT_INPUT_DIR / "step9b_prediction_rows_used.csv"
DEFAULT_OLD_CT = DEFAULT_INPUT_DIR / "step9b_old_ct_curves_no_pn.csv"
DEFAULT_OUTPUT = (
    EXP_DIR / "data" / "processed" / "step9b_ct_vs_pred_25k_np_split_custom_xlim"
)
ORIGINAL_FIGURE_DIR = EXP_DIR / "figures" / "step9b_ct_vs_pred_25k_np_split"
DEFAULT_FIGURES = EXP_DIR / "figures" / "step9b_ct_vs_pred_25k_np_split_custom_xlim"
DEFAULT_REPORT = (
    EXP_DIR
    / "reports"
    / "step9b_ct_vs_pred_25k_np_split_custom_xlim"
    / "step9b_ct_vs_pred_25k_np_split_custom_xlim_report.md"
)

TARGET_GROUPS = [
    "broad::SnTe_like",
    "broad::PbTe_like",
    "broad::BiTe_like",
    "broad::SbTe_like",
    "broad::SiGe_like",
    "broad::oxide",
    "broad::sulfide",
]

CUSTOM_X_LIMITS = {
    ("broad::SiGe_like", "n"): (50.0, 900.0),
    ("broad::SnTe_like", "p"): (0.0, 1000.0),
    ("broad::PbTe_like", "n"): (0.0, 1000.0),
    ("broad::PbTe_like", "p"): (100.0, 1000.0),
    ("broad::BiTe_like", "n"): (0.0, 800.0),
    ("broad::SbTe_like", "n"): (0.0, 1000.0),
    ("broad::SbTe_like", "p"): (0.0, 900.0),
    ("broad::sulfide", "n"): (0.0, 1300.0),
    ("broad::sulfide", "p"): (0.0, 1200.0),
}

UNCHANGED_AUTO_COMBINATIONS = {
    ("broad::SiGe_like", "p"),
    ("broad::SnTe_like", "n"),
    ("broad::BiTe_like", "p"),
    ("broad::oxide", "n"),
    ("broad::oxide", "p"),
}

INDEX_COLUMNS = [
    "figure_id",
    "material_group_key",
    "carrier_type",
    "xlim_mode",
    "requested_x_min_K",
    "requested_x_max_K",
    "applied_x_min_K",
    "applied_x_max_K",
    "figure_path_png",
    "figure_path_pdf",
    "n_prediction_points",
    "n_old_ct_points",
    "title",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create separate Step9B figures with requested custom x-axis ranges."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--old-ct", type=Path, default=DEFAULT_OLD_CT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step9b-custom-xlim] {message}", flush=True)


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or "unknown"


def directory_manifest(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not directory.exists():
        return pd.DataFrame(columns=["relative_path", "size", "mtime_ns"])
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            stat = path.stat()
            rows.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return pd.DataFrame(rows)


def compare_manifests(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    merged = before.merge(
        after,
        on="relative_path",
        how="outer",
        suffixes=("_before", "_after"),
        indicator=True,
    )
    merged["unchanged"] = (
        merged["_merge"].eq("both")
        & merged["size_before"].eq(merged["size_after"])
        & merged["mtime_ns_before"].eq(merged["mtime_ns_after"])
    )
    return merged


def validate_inputs(predictions: pd.DataFrame, old_ct: pd.DataFrame) -> None:
    required_prediction = {
        "material_group_key",
        "carrier_type",
        "T_K",
        "sigma_pred_S_per_m",
    }
    required_old = {
        "material_group_key_mapped",
        "T_K",
        "old_C_T_S_per_m",
    }
    missing_prediction = sorted(required_prediction - set(predictions.columns))
    missing_old = sorted(required_old - set(old_ct.columns))
    if missing_prediction:
        raise ValueError(f"Prediction CSV missing columns: {missing_prediction}")
    if missing_old:
        raise ValueError(f"Old C(T) CSV missing columns: {missing_old}")
    sigma_pred = pd.to_numeric(predictions["sigma_pred_S_per_m"], errors="coerce")
    old_values = pd.to_numeric(old_ct["old_C_T_S_per_m"], errors="coerce")
    if not (np.isfinite(sigma_pred) & sigma_pred.gt(0)).all():
        raise ValueError("Prediction CSV contains invalid sigma_pred")
    if not (np.isfinite(old_values) & old_values.gt(0)).all():
        raise ValueError("Old C(T) CSV contains invalid values")


def plot_one(
    group: str,
    carrier: str,
    predictions: pd.DataFrame,
    old_curve: pd.DataFrame,
    x_limits: tuple[float, float] | None,
    png_path: Path,
    pdf_path: Path,
) -> tuple[str, tuple[float, float]]:
    color = "#d95f02" if carrier == "p" else "#1b6ca8"
    title = (
        f"{group}, {carrier}: predicted sigma vs old C(T), "
        "25K bins (custom x-range)"
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.scatter(
        predictions["T_K"],
        predictions["sigma_pred_S_per_m"],
        s=16,
        alpha=0.48,
        color=color,
        edgecolors="none",
        label=f"Predicted sigma ({carrier})",
        zorder=2,
    )
    ax.plot(
        old_curve["T_K"],
        old_curve["old_C_T_S_per_m"],
        color="#222222",
        linewidth=2.5,
        label="Old C(T) from SS2026 (no p/n split)",
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("Electrical conductivity sigma [S/m]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="best")
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    applied = tuple(float(value) for value in ax.get_xlim())
    fig.tight_layout()
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return title, applied


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "n/a"
    text = frame.copy()
    for column in text.columns:
        text[column] = text[column].map(
            lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|")
        )
    header = "| " + " | ".join(text.columns) + " |"
    separator = "| " + " | ".join("---" for _ in text.columns) + " |"
    rows = [
        "| " + " | ".join(row[column] for column in text.columns) + " |"
        for _, row in text.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def write_report(
    path: Path,
    predictions_path: Path,
    old_ct_path: Path,
    index: pd.DataFrame,
    protection: pd.DataFrame,
    elapsed: float,
) -> None:
    lines = [
        "# Step9B custom x-axis figures",
        "",
        "## Summary",
        "",
        "- These are newly created figures; the original Step9B figures were not overwritten.",
        f"- Prediction rows: `{predictions_path}`",
        f"- p/n-unsplit old C(T): `{old_ct_path}`",
        "- Only the horizontal axis range was changed.",
        "- Point data, old C(T) data, y-axis scaling, colors, and legends follow the original Step9B figures.",
        "- Combinations not explicitly assigned a custom range retain matplotlib automatic x-axis limits.",
        f"- Original-figure files unchanged: {bool(len(protection) and protection['unchanged'].all())}",
        f"- PNG files: {len(index)}",
        f"- PDF files: {len(index)}",
        f"- elapsed_seconds: {elapsed:.2f}",
        "",
        "## Figure index and x-axis ranges",
        "",
        dataframe_to_markdown(
            index[
                [
                    "material_group_key",
                    "carrier_type",
                    "xlim_mode",
                    "requested_x_min_K",
                    "requested_x_max_K",
                    "applied_x_min_K",
                    "applied_x_max_K",
                    "figure_path_png",
                ]
            ]
        ),
        "",
        "## Notes",
        "",
        "- No new sigma_pred values were calculated.",
        "- Existing Step9B CSVs were read without modification.",
        "- Existing Step9B PNG/PDF files were not modified.",
        "- Measured sigma and sigma0_ref are not plotted.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_checks(
    index: pd.DataFrame,
    protection: pd.DataFrame,
    report: Path,
) -> None:
    failures: list[str] = []
    expected_combinations = {
        (group, carrier) for group in TARGET_GROUPS for carrier in ["p", "n"]
    }
    found = set(zip(index["material_group_key"], index["carrier_type"]))
    if found != expected_combinations:
        failures.append("not all 14 material/carrier figures were created")
    for _, row in index.iterrows():
        key = (row["material_group_key"], row["carrier_type"])
        png = Path(row["figure_path_png"])
        pdf = Path(row["figure_path_pdf"])
        if not png.exists() or png.stat().st_size == 0:
            failures.append(f"missing PNG: {png}")
        if not pdf.exists() or pdf.stat().st_size == 0:
            failures.append(f"missing PDF: {pdf}")
        if key in CUSTOM_X_LIMITS:
            expected = CUSTOM_X_LIMITS[key]
            applied = (float(row["applied_x_min_K"]), float(row["applied_x_max_K"]))
            if not np.allclose(applied, expected, rtol=0.0, atol=1e-9):
                failures.append(f"x-axis mismatch for {key}: {applied} != {expected}")
            if row["xlim_mode"] != "custom_fixed":
                failures.append(f"custom range not marked custom_fixed for {key}")
        elif key in UNCHANGED_AUTO_COMBINATIONS:
            if row["xlim_mode"] != "unchanged_auto":
                failures.append(f"unchanged combination not marked unchanged_auto for {key}")
        else:
            failures.append(f"combination has no x-axis policy: {key}")
    if protection.empty or not protection["unchanged"].all():
        failures.append("one or more original Step9B figure files changed")
    if not report.exists() or report.stat().st_size == 0:
        failures.append("custom-xlim report was not created")
    if failures:
        for failure in failures:
            print(f"[step9b-custom-xlim] FAIL: {failure}", flush=True)
        raise SystemExit(1)


def main() -> None:
    started = time.time()
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(args.predictions)
    if not args.old_ct.exists():
        raise FileNotFoundError(args.old_ct)
    if args.figures.resolve() == ORIGINAL_FIGURE_DIR.resolve():
        raise ValueError("Custom figures must not use the original Step9B figure directory")

    original_before = directory_manifest(ORIGINAL_FIGURE_DIR)
    predictions = pd.read_csv(args.predictions, low_memory=False)
    old_ct = pd.read_csv(args.old_ct, low_memory=False)
    validate_inputs(predictions, old_ct)

    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    figure_id = 1
    for group in TARGET_GROUPS:
        old_curve = old_ct[
            old_ct["material_group_key_mapped"].eq(group)
        ].sort_values("T_K")
        for carrier in ["p", "n"]:
            key = (group, carrier)
            pred = predictions[
                predictions["material_group_key"].eq(group)
                & predictions["carrier_type"].eq(carrier)
            ].sort_values("T_K")
            x_limits = CUSTOM_X_LIMITS.get(key)
            mode = "custom_fixed" if x_limits is not None else "unchanged_auto"
            log(
                f"creating {group} / {carrier}: "
                f"{x_limits if x_limits is not None else 'unchanged auto range'}"
            )
            stem = (
                f"{safe_name(group)}_{carrier}_sigma_pred_vs_oldCT_25k_custom_xlim"
            )
            png = args.figures / f"{stem}.png"
            pdf = args.figures / f"{stem}.pdf"
            title, applied = plot_one(
                group,
                carrier,
                pred,
                old_curve,
                x_limits,
                png,
                pdf,
            )
            rows.append(
                {
                    "figure_id": f"CUSTOM_XLIM_{figure_id:03d}",
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "xlim_mode": mode,
                    "requested_x_min_K": x_limits[0] if x_limits else np.nan,
                    "requested_x_max_K": x_limits[1] if x_limits else np.nan,
                    "applied_x_min_K": applied[0],
                    "applied_x_max_K": applied[1],
                    "figure_path_png": str(png.resolve()),
                    "figure_path_pdf": str(pdf.resolve()),
                    "n_prediction_points": len(pred),
                    "n_old_ct_points": len(old_curve),
                    "title": title,
                }
            )
            figure_id += 1
    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    index_path = args.output / "step9b_custom_xlim_figure_index.csv"
    index.to_csv(index_path, index=False, encoding="utf-8-sig")

    original_after = directory_manifest(ORIGINAL_FIGURE_DIR)
    protection = compare_manifests(original_before, original_after)
    protection_path = (
        args.output / "step9b_custom_xlim_original_figures_protection_manifest.csv"
    )
    protection.to_csv(protection_path, index=False, encoding="utf-8-sig")

    write_report(
        args.report,
        args.predictions,
        args.old_ct,
        index,
        protection,
        time.time() - started,
    )
    run_checks(index, protection, args.report)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")
    print(f"PNG figures: {len(index)}")
    print(f"PDF figures: {len(index)}")
    print(f"figure directory: {args.figures}")
    print(f"figure index: {index_path}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
