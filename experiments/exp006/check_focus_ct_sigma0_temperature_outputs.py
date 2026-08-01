import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check focus C(T) vs sigma0 temperature comparison outputs.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure-index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], label: str, failures: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [args.summary, args.figure_index, args.report]:
        if not path.exists():
            failures.append(f"missing file: {path}")
    if failures:
        raise SystemExit("\n".join(failures))

    summary = pd.read_csv(args.summary)
    figures = pd.read_csv(args.figure_index)
    require_columns(
        summary,
        [
            "material_group_key",
            "carrier_type",
            "sigma_row_count",
            "old_ct_points",
            "current_sigma0_ref_points",
            "comparison_points",
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
            "n_sigma_points",
            "n_old_ct_points",
            "n_sigma0_ref_points",
        ],
        "figure index",
        failures,
    )
    if summary.empty:
        failures.append("summary is empty")
    if figures.empty:
        failures.append("figure index is empty")
    if "figure_type" in figures and not figures["figure_type"].eq("two_panel").any():
        failures.append("no two-panel figure listed")
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
    ]:
        if phrase not in report_text:
            failures.append(f"report missing required phrase: {phrase}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("focus_ct_sigma0_temperature output checks passed")
    print(f"summary rows: {len(summary)}")
    print(f"figures: {len(figures)}")
    print(f"two-panel figures: {int(figures['figure_type'].eq('two_panel').sum())}")


if __name__ == "__main__":
    main()
