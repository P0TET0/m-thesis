import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from plot_starrydata2_sigma_temperature_comparison_step27 import (
    DEFAULT_INPUT,
    PROJECT_ROOT,
    RHO_PROPERTY,
    SIGMA_PROPERTY,
    TARGET_PROPERTIES,
    normalize_sigma,
    normalize_text,
    parse_elements,
    parse_numeric_list,
    plot_curves_by_label,
    write_curve_map,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "starrydata2_step28_sige_vs_similar_count_material"
)
SIGE_SYSTEM = "Ge-Si"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a SiGe comparison plot against a non-PbTe material system "
            "with a similar number of Starrydata2 conductivity curves."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--comparison-system",
        default="",
        help="Optional element system such as Sb-Te. If omitted, the closest count to SiGe is used.",
    )
    parser.add_argument(
        "--no-log-y",
        action="store_true",
        help="Use a linear conductivity axis instead of the default log scale.",
    )
    parser.add_argument("--y-quantile-low", type=float, default=0.01)
    parser.add_argument("--y-quantile-high", type=float, default=0.995)
    return parser.parse_args()


def element_system(composition: Any) -> str:
    elements = parse_elements(composition)
    if not elements:
        return "unknown"
    return "-".join(sorted(elements.keys()))


def load_normalized_curves(input_path: Path) -> pd.DataFrame:
    use_columns = [
        "curve_id",
        "curve_key",
        "sample_key",
        "SID",
        "DOI",
        "sample_id",
        "paper_title",
        "year",
        "composition",
        "material_system",
        "n_or_p",
        "figure_id",
        "prop_x",
        "property_step5",
        "unit_y",
        "x_values_json",
        "y_values_json",
    ]
    df = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        usecols=lambda column: column in use_columns,
        low_memory=False,
    )
    df = df[
        df["property_step5"].isin(TARGET_PROPERTIES)
        & df["prop_x"].map(normalize_text).eq("Temperature")
    ].copy()
    df["element_system"] = df["composition"].map(element_system)

    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        x_values = parse_numeric_list(getattr(row, "x_values_json"))
        y_values = parse_numeric_list(getattr(row, "y_values_json"))
        t_values, sigma_values, status = normalize_sigma(
            x_values,
            y_values,
            getattr(row, "property_step5"),
            getattr(row, "unit_y", ""),
        )
        if status != "ok":
            continue
        row_dict = row._asdict()
        row_dict.pop("x_values_json", None)
        row_dict.pop("y_values_json", None)
        records.append(
            {
                **row_dict,
                "source_property": getattr(row, "property_step5"),
                "temperature_K": t_values.tolist(),
                "sigma_S_per_m": sigma_values.tolist(),
                "T_min_K": float(t_values[0]),
                "T_max_K": float(t_values[-1]),
                "sigma_min_S_per_m": float(sigma_values.min()),
                "sigma_max_S_per_m": float(sigma_values.max()),
                "n_points_normalized": int(len(t_values)),
            }
        )
    if not records:
        raise SystemExit("no usable conductivity curves were found")
    return pd.DataFrame(records)


def choose_comparison_system(curves: pd.DataFrame, explicit_system: str) -> str:
    if explicit_system:
        if explicit_system not in set(curves["element_system"]):
            raise SystemExit(f"comparison system not found: {explicit_system}")
        return explicit_system

    counts = curves["element_system"].value_counts()
    sige_count = int(counts.get(SIGE_SYSTEM, 0))
    if sige_count == 0:
        raise SystemExit("SiGe curves were not found")

    candidates = counts.drop(labels=[SIGE_SYSTEM, "unknown"], errors="ignore")
    # Avoid selecting Pb-Te again; the request is specifically for a non-PbTe comparison.
    candidates = candidates[~candidates.index.map(lambda system: {"Pb", "Te"}.issubset(set(system.split("-"))))]
    if candidates.empty:
        raise SystemExit("no non-PbTe comparison systems were found")
    return str((candidates - sige_count).abs().sort_values(kind="mergesort").index[0])


def safe_system_filename(system: str) -> str:
    return system.lower().replace("-", "_")


def write_report(curves: pd.DataFrame, output_dir: Path, comparison_system: str) -> None:
    counts = curves["element_system"].value_counts()
    rows = [
        ("sige_system", SIGE_SYSTEM),
        ("comparison_system", comparison_system),
        ("sige_curves", int(counts.get(SIGE_SYSTEM, 0))),
        ("comparison_curves", int(counts.get(comparison_system, 0))),
    ]
    report = pd.DataFrame(rows, columns=["metric", "value"])
    report.to_csv(output_dir / "step28_sige_vs_similar_count_material_report.csv", index=False)
    text = "\n".join(f"{metric}: {value}" for metric, value in rows) + "\n"
    (output_dir / "step28_sige_vs_similar_count_material_report.txt").write_text(
        text, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_y = not args.no_log_y

    curves = load_normalized_curves(args.input)
    comparison_system = choose_comparison_system(curves, args.comparison_system)

    selected = curves[curves["element_system"].isin([SIGE_SYSTEM, comparison_system])].copy()
    selected["family_label"] = selected["element_system"].map(
        {SIGE_SYSTEM: "SiGe", comparison_system: comparison_system}
    )
    if selected["family_label"].isna().any():
        raise SystemExit("failed to label selected curves")

    write_curve_map(selected, args.output_dir)
    write_report(selected, args.output_dir, comparison_system)

    labels = ["SiGe", comparison_system]
    system_slug = safe_system_filename(comparison_system)
    title = (
        "Starrydata2 Si-Ge vs "
        f"{comparison_system} electrical conductivity curves "
        "(similar record count)"
    )

    common_kwargs = {
        "log_y": log_y,
        "dpi": args.dpi,
        "y_quantile_low": args.y_quantile_low,
        "y_quantile_high": args.y_quantile_high,
        "line_styles": {"SiGe": "-", comparison_system: "--"},
        "line_colors": {"SiGe": "#6baed6", comparison_system: "#74c476"},
        "median_colors": {"SiGe": "#08519c", comparison_system: "#006d2c"},
        "line_alphas": {"SiGe": 0.18, comparison_system: 0.18},
        "line_widths": {"SiGe": 0.75, comparison_system: 0.75},
        "median_widths": {"SiGe": 3.6, comparison_system: 3.8},
    }
    plot_curves_by_label(
        selected,
        args.output_dir / f"figure_01_sige_vs_{system_slug}_sigma.png",
        title,
        "family_label",
        labels,
        **common_kwargs,
    )
    plot_curves_by_label(
        selected,
        args.output_dir / f"figure_01_sige_vs_{system_slug}_sigma_without_trend_lines.png",
        title,
        "family_label",
        labels,
        show_median=False,
        **common_kwargs,
    )

    counts = selected["family_label"].value_counts()
    print(f"output_dir: {args.output_dir}")
    print(f"comparison_system: {comparison_system}")
    print(f"SiGe curves: {int(counts.get('SiGe', 0))}")
    print(f"{comparison_system} curves: {int(counts.get(comparison_system, 0))}")
    print("figures:")
    print(f"- figure_01_sige_vs_{system_slug}_sigma.png")
    print(f"- figure_01_sige_vs_{system_slug}_sigma_without_trend_lines.png")


if __name__ == "__main__":
    main()
