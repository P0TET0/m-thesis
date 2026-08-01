import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check old C(T) vs sigma0_ref-only comparison outputs.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure-index", type=Path, required=True)
    parser.add_argument("--script-parse-summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], label: str, failures: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [args.summary, args.figure_index, args.script_parse_summary, args.report]:
        if not path.exists():
            failures.append(f"missing file: {path}")
    if failures:
        raise SystemExit("\n".join(failures))

    summary = pd.read_csv(args.summary)
    figures = pd.read_csv(args.figure_index)
    parse = pd.read_csv(args.script_parse_summary)

    require_columns(
        parse,
        [
            "old_ct_script_path",
            "detected_output_file",
            "detected_old_ct_column",
            "detected_temperature_column",
            "detected_material_column",
            "detected_carrier_column",
            "detection_status",
            "notes",
        ],
        "script parse summary",
        failures,
    )
    require_columns(
        summary,
        [
            "material_group_key",
            "carrier_type",
            "old_ct_points",
            "sigma0_ref_points",
            "comparison_points",
            "median_log10_sigma0ref_over_oldCT",
            "warning",
        ],
        "summary",
        failures,
    )
    require_columns(
        figures,
        [
            "figure_id",
            "material_group_key",
            "carrier_type",
            "figure_type",
            "figure_path_png",
            "figure_path_pdf",
            "n_old_ct_points",
            "n_sigma0_ref_points",
        ],
        "figure index",
        failures,
    )
    if parse.empty or not parse["detection_status"].astype(str).eq("selected").any():
        failures.append("script parse summary has no selected old C(T) output")
    if summary.empty:
        failures.append("summary is empty")
    if figures.empty or not figures["figure_type"].astype(str).eq("overlay").any():
        failures.append("no overlay figure listed")
    for _, row in figures.iterrows():
        for column in ["figure_path_png", "figure_path_pdf"]:
            path = Path(str(row[column]))
            if not path.exists():
                failures.append(f"missing figure file: {path}")
    report_text = args.report.read_text(encoding="utf-8", errors="replace")
    for phrase in [
        "No new sigma_pred is calculated",
        "Step4 full-data reference curves are not used",
        "Starrydata2 raw data are not read",
        "old C(T) source script",
    ]:
        if phrase not in report_text:
            failures.append(f"report missing required phrase: {phrase}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("focus_ct_vs_sigma0ref_only_from_script output checks passed")
    print(f"summary rows: {len(summary)}")
    print(f"figures: {len(figures)}")
    print(f"overlay figures: {int(figures['figure_type'].eq('overlay').sum())}")


if __name__ == "__main__":
    main()
