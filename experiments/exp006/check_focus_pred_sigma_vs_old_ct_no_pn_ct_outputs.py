import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check predicted sigma vs no-p/n old C(T) outputs.")
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
            "prediction_points",
            "p_prediction_points",
            "n_prediction_points",
            "old_ct_points",
            "median_log10_pred_over_oldCT_nearest",
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
            "figure_type",
            "figure_path_png",
            "figure_path_pdf",
            "n_prediction_points",
            "n_old_ct_points",
        ],
        "figure index",
        failures,
    )
    if summary.empty:
        failures.append("summary is empty")
    if figures.empty:
        failures.append("figure index is empty")
    for _, row in figures.iterrows():
        for column in ["figure_path_png", "figure_path_pdf"]:
            path = Path(str(row[column]))
            if not path.exists():
                failures.append(f"missing figure: {path}")

    report = args.report.read_text(encoding="utf-8", errors="replace")
    for phrase in [
        "Old C(T) is aggregated without p/n splitting",
        "material group x temperature",
        "Points are current predicted sigma",
        "sigma0_ref is not included",
        "Experimental sigma points are not included in the main figures",
        "No new sigma_pred is calculated",
        "Step4 full-data reference curves are not used",
        "Starrydata2 raw data are not read",
    ]:
        if phrase not in report:
            failures.append(f"report missing phrase: {phrase}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("focus_pred_sigma_vs_old_ct_no_pn_ct output checks passed")
    print(f"summary rows: {len(summary)}")
    print(f"figures: {len(figures)}")


if __name__ == "__main__":
    main()
