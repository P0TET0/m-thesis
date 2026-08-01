# Build a paper_curve_meta.csv template from curve_master.csv.
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_DIR = PROJECT_ROOT / "data" / "output" / "master"
INPUT_CSV = MASTER_DIR / "curve_master.csv"
OUTPUT_CSV = MASTER_DIR / "paper_curve_meta.csv"


def build_paper_curve_meta(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "curve_id" not in df.columns:
        raise KeyError("missing column: curve_id")

    curve_ids = df["curve_id"].drop_duplicates(keep="first").reset_index(drop=True)
    df_out = pd.DataFrame(
        {
            "curve_id": curve_ids,
            "dopant_element": "unknown",
            "carrier_conc_cm3": "",
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    return df_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper_curve_meta.csv from curve_master.csv")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV, help="input curve master csv")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV, help="output paper curve meta csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        df_out = build_paper_curve_meta(args.input_csv, args.output_csv)
    except Exception as exc:
        raise SystemExit(f"failed to build paper curve meta: {exc}") from exc

    print(f"saved: {args.output_csv}")
    print(f"rows: {len(df_out)}")


if __name__ == "__main__":
    main()
