import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "output" / "starrydata2_prepared_for_relaxation_time.xlsx"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "output" / "starrydata2_prepared_for_relaxation_time_csv"
)
TARGET_SHEETS = ("sample_master", "property_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the relaxation-time workbook sheets to CSV files."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="source workbook path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory where CSV files will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        excel = pd.ExcelFile(args.input_path)
        missing_sheets = [sheet for sheet in TARGET_SHEETS if sheet not in excel.sheet_names]
        if missing_sheets:
            raise KeyError(f"missing sheets in workbook: {missing_sheets}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for sheet_name in TARGET_SHEETS:
            df = excel.parse(sheet_name=sheet_name)
            output_path = args.output_dir / f"{sheet_name}.csv"
            df.to_csv(output_path, index=False)
            print(f"saved: {output_path}")
            print(f"rows_{sheet_name}: {len(df)}")
    except Exception as exc:
        raise SystemExit(f"failed to export workbook to CSV: {exc}") from exc


if __name__ == "__main__":
    main()
