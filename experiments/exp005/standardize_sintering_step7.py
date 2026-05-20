import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP6_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step6_np_classification"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step7_sintering_unknown"

INPUT_OUTPUT_FILES = [
    (
        "sample_np_classification",
        "sample_np_classification_step6.csv",
        "sample_np_classification_step7.csv",
    ),
    (
        "sample_property_availability",
        "sample_property_availability_step6.csv",
        "sample_property_availability_step7.csv",
    ),
    (
        "candidate_samples_np",
        "candidate_samples_np_step6.csv",
        "candidate_samples_np_step7.csv",
    ),
    (
        "property_core_curves",
        "property_core_curves_step6.csv",
        "property_core_curves_step7.csv",
    ),
    (
        "candidate_core_curves",
        "candidate_core_curves_step6.csv",
        "candidate_core_curves_step7.csv",
    ),
]

SINTERING_COLUMNS = ["sintering_method", "sintering_checked", "record_checked"]
SINTERING_STATUS = "not_checked"
SINTERING_NOTE = (
    "not checked at step7; to be checked only for high-error, high-ZT, "
    "or final-paper samples"
)
NP_COLUMNS = [
    "n_or_p",
    "n_or_p_basis",
    "n_or_p_step6",
    "n_or_p_basis_step6",
    "n_or_p_confidence_step6",
]
LEARNING_COLUMNS = [
    "is_learning_candidate_step5",
    "learning_candidate_reason_step5",
    "has_sigma_or_rho",
    "has_seebeck",
    "has_kappa_or_zt",
    "sigma_or_rho_point_count",
    "seebeck_point_count",
    "kappa_or_zt_point_count",
]
EXCEL_MAX_ROWS = 1_048_576
EXCEL_PREVIEW_ROWS = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize sintering metadata to unknown/no for Step7."
    )
    parser.add_argument("--step6_dir", type=Path, default=DEFAULT_STEP6_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "null"}:
        return ""
    return text


def validate_sample_key(df: pd.DataFrame, filename: str) -> None:
    if "sample_key" not in df.columns:
        raise KeyError(f"{filename} missing required column: sample_key")


def prior_non_default_count(values: pd.Series, allowed: set[str]) -> int:
    normalized = values.map(lambda value: normalize_text(value).casefold())
    return int(normalized.map(lambda value: bool(value) and value not in allowed).sum())


def standardize_sintering(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in SINTERING_COLUMNS:
        prior_column = f"{column}_prior_step7"
        if column in output.columns:
            output[prior_column] = output[column]
        else:
            output[prior_column] = ""

    output["sintering_method"] = "unknown"
    output["sintering_checked"] = "no"
    output["record_checked"] = "no"
    output["sintering_status_step7"] = SINTERING_STATUS
    output["sintering_note_step7"] = SINTERING_NOTE
    return output


def changed_rows(before: pd.DataFrame, after: pd.DataFrame, columns: list[str]) -> int:
    change_mask = pd.Series(False, index=before.index)
    for column in columns:
        if column not in before.columns or column not in after.columns:
            continue
        before_values = before[column].map(normalize_text)
        after_values = after[column].map(normalize_text)
        change_mask = change_mask | before_values.ne(after_values)
    return int(change_mask.sum())


def status_failures(df: pd.DataFrame) -> dict[str, int]:
    return {
        "sintering_method_not_unknown": int(
            df["sintering_method"].map(lambda value: normalize_text(value).casefold()).ne("unknown").sum()
        ),
        "sintering_checked_not_no": int(
            df["sintering_checked"].map(lambda value: normalize_text(value).casefold()).ne("no").sum()
        ),
        "record_checked_not_no": int(
            df["record_checked"].map(lambda value: normalize_text(value).casefold()).ne("no").sum()
        ),
        "sintering_status_not_not_checked": int(
            df["sintering_status_step7"]
            .map(lambda value: normalize_text(value).casefold())
            .ne(SINTERING_STATUS)
            .sum()
        ),
    }


def build_report(
    inputs: dict[str, pd.DataFrame],
    outputs: dict[str, pd.DataFrame],
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    rows: list[tuple[str, str]] = []
    total_unknown = 0
    total_checked_no = 0
    total_record_no = 0
    total_prior_method = 0
    total_prior_checked = 0
    total_prior_record = 0
    total_np_changed = 0
    total_learning_changed = 0

    for label, input_name, output_name in INPUT_OUTPUT_FILES:
        before = inputs[label]
        after = outputs[label]
        rows.append((f"input_{input_name}_rows", str(len(before))))
        rows.append((f"output_{output_name}_rows", str(len(after))))
        rows.append((f"{label}_row_count_changed", str(len(before) != len(after))))

        method_unknown = int(after["sintering_method"].eq("unknown").sum())
        checked_no = int(after["sintering_checked"].eq("no").sum())
        record_no = int(after["record_checked"].eq("no").sum())
        total_unknown += method_unknown
        total_checked_no += checked_no
        total_record_no += record_no
        rows.append((f"{label}_sintering_method_unknown_rows", str(method_unknown)))
        rows.append((f"{label}_sintering_checked_no_rows", str(checked_no)))
        rows.append((f"{label}_record_checked_no_rows", str(record_no)))

        prior_method = prior_non_default_count(
            after["sintering_method_prior_step7"], {"unknown"}
        )
        prior_checked = prior_non_default_count(
            after["sintering_checked_prior_step7"], {"no"}
        )
        prior_record = prior_non_default_count(after["record_checked_prior_step7"], {"no"})
        total_prior_method += prior_method
        total_prior_checked += prior_checked
        total_prior_record += prior_record
        rows.append((f"{label}_prior_sintering_method_non_unknown_rows", str(prior_method)))
        rows.append((f"{label}_prior_sintering_checked_non_no_rows", str(prior_checked)))
        rows.append((f"{label}_prior_record_checked_non_no_rows", str(prior_record)))

        np_changed = changed_rows(before, after, NP_COLUMNS)
        learning_changed = changed_rows(before, after, LEARNING_COLUMNS)
        total_np_changed += np_changed
        total_learning_changed += learning_changed
        rows.append((f"{label}_np_changed_rows", str(np_changed)))
        rows.append((f"{label}_learning_flag_changed_rows", str(learning_changed)))

        for failure_name, count in status_failures(after).items():
            rows.append((f"{label}_{failure_name}", str(count)))

    rows.extend(
        [
            ("total_sintering_method_unknown_rows", str(total_unknown)),
            ("total_sintering_checked_no_rows", str(total_checked_no)),
            ("total_record_checked_no_rows", str(total_record_no)),
            ("total_prior_sintering_method_non_unknown_rows", str(total_prior_method)),
            ("total_prior_sintering_checked_non_no_rows", str(total_prior_checked)),
            ("total_prior_record_checked_non_no_rows", str(total_prior_record)),
            ("total_np_changed_rows", str(total_np_changed)),
            ("total_learning_flag_changed_rows", str(total_learning_changed)),
        ]
    )

    candidate = outputs["candidate_samples_np"]
    if "n_or_p" in candidate.columns:
        for value, count in candidate["n_or_p"].value_counts(dropna=False).sort_index().items():
            rows.append((f"candidate_samples_np_step7_n_or_p_{value}_rows", str(int(count))))
    if "n_or_p_confidence_step6" in candidate.columns:
        for value, count in (
            candidate["n_or_p_confidence_step6"].value_counts(dropna=False).sort_index().items()
        ):
            rows.append(
                (f"candidate_samples_np_step7_confidence_{value}_rows", str(int(count)))
            )

    for label in ["property_core_curves", "candidate_core_curves"]:
        frame = outputs[label]
        if "property_step5" in frame.columns:
            for value, count in frame["property_step5"].value_counts(dropna=False).sort_index().items():
                rows.append((f"{label}_step7_property_{value}_curve_count", str(int(count))))

    for note in excel_notes:
        rows.append(("excel_note", note))

    report_df = pd.DataFrame(rows, columns=["metric", "value"])
    report_text = "\n".join(f"{metric}: {value}" for metric, value in rows) + "\n"
    return report_text, report_df


def assert_acceptance(outputs: dict[str, pd.DataFrame]) -> None:
    for label, frame in outputs.items():
        if "sample_key" not in frame.columns:
            raise KeyError(f"{label} missing sample_key")
        for column, expected in (
            ("sintering_method", "unknown"),
            ("sintering_checked", "no"),
            ("record_checked", "no"),
            ("sintering_status_step7", SINTERING_STATUS),
        ):
            if column not in frame.columns:
                raise KeyError(f"{label} missing {column}")
            bad = frame[column].map(lambda value: normalize_text(value).casefold()).ne(expected)
            if bad.any():
                raise ValueError(f"{label} has non-standard {column} values")

    for label in ["sample_np_classification", "sample_property_availability"]:
        frame = outputs[label]
        if frame["sample_key"].duplicated().any():
            raise ValueError(f"{label} is not one row per sample_key")

    if "is_learning_candidate_step5" in outputs["candidate_samples_np"].columns:
        if not outputs["candidate_samples_np"]["is_learning_candidate_step5"].map(
            lambda value: normalize_text(value).casefold() == "true"
        ).all():
            raise ValueError("candidate_samples_np_step7 contains non-learning-candidate rows")

    for label in ["property_core_curves", "candidate_core_curves"]:
        frame = outputs[label]
        for column in ["x_values_json", "y_values_json"]:
            if column not in frame.columns:
                raise KeyError(f"{label} missing {column}")


def write_csv_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame], report_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, _, output_name in INPUT_OUTPUT_FILES:
        outputs[label].to_csv(output_dir / output_name, index=False)
    (output_dir / "step7_sintering_unknown_report.txt").write_text(
        report_text, encoding="utf-8"
    )


def excel_frame(df: pd.DataFrame, sheet_name: str, excel_notes: list[str]) -> pd.DataFrame:
    if len(df) <= EXCEL_MAX_ROWS - 1:
        return df
    excel_notes.append(
        f"{sheet_name} exceeded Excel row limit; wrote first {EXCEL_PREVIEW_ROWS} rows to workbook"
    )
    return df.head(EXCEL_PREVIEW_ROWS)


def fit_worksheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
    preview = df.head(200)
    for index, column in enumerate(df.columns, start=1):
        max_length = len(str(column))
        if not preview.empty:
            max_length = max(max_length, int(preview[column].astype(str).map(len).max()))
        worksheet.column_dimensions[worksheet.cell(row=1, column=index).column_letter].width = min(
            max(max_length + 2, 12), 60
        )


def write_excel_output(
    output_dir: Path,
    outputs: dict[str, pd.DataFrame],
    report_df: pd.DataFrame,
    excel_notes: list[str],
) -> None:
    path = output_dir / "starrydata2_step7_sintering_unknown.xlsx"
    sheets = {
        "sample_np_classification": outputs["sample_np_classification"],
        "sample_property_availability": outputs["sample_property_availability"],
        "candidate_samples_np": outputs["candidate_samples_np"],
        "property_core_curves": excel_frame(
            outputs["property_core_curves"], "property_core_curves", excel_notes
        ),
        "candidate_core_curves": excel_frame(
            outputs["candidate_core_curves"], "candidate_core_curves", excel_notes
        ),
        "sintering_report": report_df,
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            fit_worksheet(writer, sheet_name, frame)


def main() -> None:
    args = parse_args()
    inputs: dict[str, pd.DataFrame] = {}
    outputs: dict[str, pd.DataFrame] = {}

    for label, input_name, _ in INPUT_OUTPUT_FILES:
        path = args.step6_dir / input_name
        frame = read_csv_text(path)
        validate_sample_key(frame, input_name)
        inputs[label] = frame
        outputs[label] = standardize_sintering(frame)

    assert_acceptance(outputs)

    excel_notes: list[str] = []
    report_text, report_df = build_report(inputs, outputs, excel_notes)
    write_csv_outputs(args.output_dir, outputs, report_text)
    write_excel_output(args.output_dir, outputs, report_df, excel_notes)
    if excel_notes:
        report_text, report_df = build_report(inputs, outputs, excel_notes)
        (args.output_dir / "step7_sintering_unknown_report.txt").write_text(
            report_text, encoding="utf-8"
        )

    total_unknown = sum(int(df["sintering_method"].eq("unknown").sum()) for df in outputs.values())
    total_checked_no = sum(int(df["sintering_checked"].eq("no").sum()) for df in outputs.values())
    total_record_no = sum(int(df["record_checked"].eq("no").sum()) for df in outputs.values())
    total_np_changed = sum(changed_rows(inputs[label], outputs[label], NP_COLUMNS) for label in inputs)
    total_learning_changed = sum(
        changed_rows(inputs[label], outputs[label], LEARNING_COLUMNS) for label in inputs
    )

    print("Done.")
    print("Created:")
    print("- sample_np_classification_step7.csv")
    print("- sample_property_availability_step7.csv")
    print("- candidate_samples_np_step7.csv")
    print("- property_core_curves_step7.csv")
    print("- candidate_core_curves_step7.csv")
    print("- step7_sintering_unknown_report.txt")
    print("- starrydata2_step7_sintering_unknown.xlsx")
    print("")
    print("Summary:")
    print(f"sample_np_classification_step7 rows: {len(outputs['sample_np_classification'])}")
    print(f"sample_property_availability_step7 rows: {len(outputs['sample_property_availability'])}")
    print(f"candidate_samples_np_step7 rows: {len(outputs['candidate_samples_np'])}")
    print(f"property_core_curves_step7 rows: {len(outputs['property_core_curves'])}")
    print(f"candidate_core_curves_step7 rows: {len(outputs['candidate_core_curves'])}")
    print(f"sintering_method unknown rows: {total_unknown}")
    print(f"sintering_checked no rows: {total_checked_no}")
    print(f"record_checked no rows: {total_record_no}")
    print(f"n/p changed rows: {total_np_changed}")
    print(f"learning candidate flag changed rows: {total_learning_changed}")


if __name__ == "__main__":
    main()
