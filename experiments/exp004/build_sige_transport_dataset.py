import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "sige_transport_dataset"
SIGE_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "sige"
INPUT_ALL_CURVES_CSV = SIGE_OUTPUT_DIR / "sige_all_curves.csv"
INPUT_COMPLETE_SAMPLES_CSV = (
    SIGE_OUTPUT_DIR / "sige_transport_allowed_complete_samples.csv"
)
INPUT_DOI_SUMMARY_CSV = (
    SIGE_OUTPUT_DIR / "sige_transport_allowed_complete_doi_summary.csv"
)

PROPERTIES = {
    "transport": {"Electrical conductivity", "Electrical resistivity"},
    "seebeck": {"Seebeck coefficient"},
    "kappa": {"Thermal conductivity"},
}
EXPECTED_KEYS = {"transport", "seebeck", "kappa"}
DOI_PLOT_DIRNAME = "doi_plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build aligned SiGe transport dataset on the common temperature domain "
            "for DOIs with multiple complete samples."
        )
    )
    parser.add_argument("--all-curves-csv", type=Path, default=INPUT_ALL_CURVES_CSV)
    parser.add_argument(
        "--complete-samples-csv", type=Path, default=INPUT_COMPLETE_SAMPLES_CSV
    )
    parser.add_argument("--doi-summary-csv", type=Path, default=INPUT_DOI_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def safe_parse_list(raw_value: Any) -> list[Any]:
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, tuple):
        return list(raw_value)
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return []
    if isinstance(raw_value, str):
        try:
            parsed = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    return []


def to_float_array(values: list[Any]) -> np.ndarray:
    parsed: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            parsed.append(numeric)
    return np.asarray(parsed, dtype=float)


def normalize_prop_key(prop_y: Any) -> str | None:
    text = "" if prop_y is None else str(prop_y).strip()
    for key, names in PROPERTIES.items():
        if text in names:
            return key
    return None


def normalize_curve(
    x_values: list[Any], y_values: list[Any], prop_y: str
) -> tuple[np.ndarray, np.ndarray]:
    x_array = to_float_array(x_values)
    y_array = to_float_array(y_values)
    if len(x_array) != len(y_array):
        n = min(len(x_array), len(y_array))
        x_array = x_array[:n]
        y_array = y_array[:n]

    if len(x_array) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    mask = np.isfinite(x_array) & np.isfinite(y_array) & (x_array > 0.0)
    x_array = x_array[mask]
    y_array = y_array[mask]

    if prop_y == "Electrical resistivity":
        nonzero = y_array != 0.0
        x_array = x_array[nonzero]
        y_array = y_array[nonzero]
        y_array = 1.0 / y_array

    if len(x_array) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    order = np.argsort(x_array)
    x_array = x_array[order]
    y_array = y_array[order]

    unique_x: list[float] = []
    unique_y: list[float] = []
    for x_value in np.unique(x_array):
        same_x = x_array == x_value
        unique_x.append(float(x_value))
        unique_y.append(float(np.mean(y_array[same_x])))

    return np.asarray(unique_x, dtype=float), np.asarray(unique_y, dtype=float)


def build_common_grid(curves: dict[str, dict[str, Any]]) -> np.ndarray:
    t_min = max(float(curves[key]["x"][0]) for key in EXPECTED_KEYS)
    t_max = min(float(curves[key]["x"][-1]) for key in EXPECTED_KEYS)
    if not math.isfinite(t_min) or not math.isfinite(t_max) or t_max <= t_min:
        return np.asarray([], dtype=float)

    grid_values: list[float] = []
    for key in EXPECTED_KEYS:
        x_values = curves[key]["x"]
        mask = (x_values >= t_min) & (x_values <= t_max)
        grid_values.extend([float(value) for value in x_values[mask]])

    if not grid_values:
        return np.asarray([t_min, t_max], dtype=float)

    grid = np.asarray(sorted(set(grid_values)), dtype=float)
    if len(grid) == 1:
        grid = np.asarray(sorted({float(grid[0]), t_min, t_max}), dtype=float)
    return grid


def interpolate_curve(x_values: np.ndarray, y_values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(x_values) < 2 or len(grid) == 0:
        return np.asarray([], dtype=float)
    return np.interp(grid, x_values, y_values)


def sign_label(values: np.ndarray, tolerance: float = 1e-12) -> tuple[int, str, bool]:
    if len(values) == 0:
        return 0, "empty", False
    positive = bool(np.all(values > tolerance))
    negative = bool(np.all(values < -tolerance))
    if positive:
        return 1, "positive", True
    if negative:
        return -1, "negative", True
    if np.all(np.abs(values) <= tolerance):
        return 0, "zero", False
    return 0, "mixed", False


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def target_dois(doi_summary_csv: Path) -> list[str]:
    df = pd.read_csv(doi_summary_csv)
    if "DOI" not in df.columns or "complete_samples" not in df.columns:
        raise KeyError("DOI summary CSV must contain DOI and complete_samples columns")
    df["complete_samples"] = pd.to_numeric(df["complete_samples"], errors="coerce")
    subset = df[df["complete_samples"] > 1].copy()
    return subset["DOI"].astype(str).drop_duplicates().tolist()


def target_samples(complete_samples_csv: Path, allowed_dois: set[str]) -> pd.DataFrame:
    df = pd.read_csv(complete_samples_csv)
    required = {"DOI", "sample_id", "composition", "transport_source"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"complete sample CSV missing columns: {sorted(missing)}")
    df["DOI"] = df["DOI"].astype(str)
    df = df[df["DOI"].isin(allowed_dois)].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def load_target_curves(all_curves_csv: Path, sample_ids: set[str]) -> pd.DataFrame:
    df = pd.read_csv(all_curves_csv)
    required = {
        "DOI",
        "composition",
        "sample_id",
        "prop_x",
        "prop_y",
        "x_list",
        "y_list",
        "si_frac",
        "ge_frac",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"all curves CSV missing columns: {sorted(missing)}")
    df["sample_id"] = df["sample_id"].astype(str)
    df = df[df["sample_id"].isin(sample_ids)].copy()
    df = df[df["prop_x"].astype(str).str.strip() == "Temperature"].copy()
    df["prop_key"] = df["prop_y"].apply(normalize_prop_key)
    df = df[df["prop_key"].notna()].copy()
    return df


def sample_records(curve_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (doi, composition, sample_id), group in curve_df.groupby(
        ["DOI", "composition", "sample_id"], sort=True
    ):
        curves: dict[str, dict[str, Any]] = {}
        for row in group.itertuples(index=False):
            prop_key = getattr(row, "prop_key")
            x_values = safe_parse_list(getattr(row, "x_list"))
            y_values = safe_parse_list(getattr(row, "y_list"))
            x_curve, y_curve = normalize_curve(x_values, y_values, getattr(row, "prop_y"))
            if len(x_curve) < 2:
                continue
            curves[prop_key] = {
                "x": x_curve,
                "y": y_curve,
                "prop_y_original": getattr(row, "prop_y"),
                "unit_y": getattr(row, "unit_y", ""),
                "si_frac": getattr(row, "si_frac"),
                "ge_frac": getattr(row, "ge_frac"),
            }

        if set(curves.keys()) != EXPECTED_KEYS:
            continue

        grid = build_common_grid(curves)
        if len(grid) < 2:
            continue

        sigma = interpolate_curve(curves["transport"]["x"], curves["transport"]["y"], grid)
        seebeck = interpolate_curve(curves["seebeck"]["x"], curves["seebeck"]["y"], grid)
        kappa = interpolate_curve(curves["kappa"]["x"], curves["kappa"]["y"], grid)
        valid_mask = np.isfinite(sigma) & np.isfinite(seebeck) & np.isfinite(kappa) & (sigma > 0.0) & (kappa > 0.0)
        if valid_mask.sum() < 2:
            continue

        grid = grid[valid_mask]
        sigma = sigma[valid_mask]
        seebeck = seebeck[valid_mask]
        kappa = kappa[valid_mask]

        pf = (seebeck ** 2) * sigma
        zt_calc = pf * grid / kappa
        s_sign, s_label, s_stable = sign_label(seebeck)

        record: dict[str, Any] = {
            "DOI": str(doi),
            "composition": str(composition),
            "sample_id": str(sample_id),
            "si_frac": pd.to_numeric(curves["transport"]["si_frac"], errors="coerce"),
            "ge_frac": pd.to_numeric(curves["transport"]["ge_frac"], errors="coerce"),
            "transport_source": curves["transport"]["prop_y_original"],
            "sigma_unit": "S/m",
            "seebeck_unit": curves["seebeck"]["unit_y"],
            "kappa_unit": curves["kappa"]["unit_y"],
            "sigma_T_min_K": float(curves["transport"]["x"][0]),
            "sigma_T_max_K": float(curves["transport"]["x"][-1]),
            "sigma_n_raw_points": int(len(curves["transport"]["x"])),
            "S_T_min_K": float(curves["seebeck"]["x"][0]),
            "S_T_max_K": float(curves["seebeck"]["x"][-1]),
            "S_n_raw_points": int(len(curves["seebeck"]["x"])),
            "kappa_T_min_K": float(curves["kappa"]["x"][0]),
            "kappa_T_max_K": float(curves["kappa"]["x"][-1]),
            "kappa_n_raw_points": int(len(curves["kappa"]["x"])),
            "common_T_min_K": float(grid[0]),
            "common_T_max_K": float(grid[-1]),
            "common_T_span_K": float(grid[-1] - grid[0]),
            "common_grid_n_points": int(len(grid)),
            "S_sign": int(s_sign),
            "seebeck_sign_label": s_label,
            "seebeck_sign_stable": bool(s_stable),
            "PF_min_W_mK2": float(np.min(pf)),
            "PF_max_W_mK2": float(np.max(pf)),
            "ZT_calc_min": float(np.min(zt_calc)),
            "ZT_calc_max": float(np.max(zt_calc)),
            "T_grid_K": grid.tolist(),
            "sigma_S_per_m": sigma.tolist(),
            "S_V_per_K": seebeck.tolist(),
            "kappa_W_per_mK": kappa.tolist(),
            "PF_W_per_mK2": pf.tolist(),
            "ZT_calc": zt_calc.tolist(),
        }
        records.append(record)
    return records


def build_point_table(curve_table: pd.DataFrame) -> pd.DataFrame:
    point_records: list[dict[str, Any]] = []
    for row in curve_table.itertuples(index=False):
        temperatures = safe_parse_list(getattr(row, "T_grid_K"))
        sigma = safe_parse_list(getattr(row, "sigma_S_per_m"))
        seebeck = safe_parse_list(getattr(row, "S_V_per_K"))
        kappa = safe_parse_list(getattr(row, "kappa_W_per_mK"))
        pf = safe_parse_list(getattr(row, "PF_W_per_mK2"))
        zt_calc = safe_parse_list(getattr(row, "ZT_calc"))
        n_points = min(len(temperatures), len(sigma), len(seebeck), len(kappa), len(pf), len(zt_calc))
        for point_index in range(n_points):
            point_records.append(
                {
                    "DOI": getattr(row, "DOI"),
                    "composition": getattr(row, "composition"),
                    "sample_id": getattr(row, "sample_id"),
                    "si_frac": getattr(row, "si_frac"),
                    "ge_frac": getattr(row, "ge_frac"),
                    "transport_source": getattr(row, "transport_source"),
                    "S_sign": getattr(row, "S_sign"),
                    "seebeck_sign_label": getattr(row, "seebeck_sign_label"),
                    "seebeck_sign_stable": getattr(row, "seebeck_sign_stable"),
                    "point_index": point_index,
                    "curve_n_points": n_points,
                    "point_weight": 1.0 / n_points,
                    "T_K": float(temperatures[point_index]),
                    "sigma_S_per_m": float(sigma[point_index]),
                    "S_V_per_K": float(seebeck[point_index]),
                    "kappa_W_per_mK": float(kappa[point_index]),
                    "PF_W_per_mK2": float(pf[point_index]),
                    "ZT_calc": float(zt_calc[point_index]),
                }
            )
    return pd.DataFrame(point_records)


def build_doi_priority(sample_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for doi, group in sample_table.groupby("DOI", sort=False):
        compositions = sorted(group["composition"].astype(str).unique().tolist())
        rows.append(
            {
                "DOI": doi,
                "usable_samples": int(len(group)),
                "unique_compositions": len(compositions),
                "compositions": ", ".join(compositions),
                "total_aligned_points": int(group["common_grid_n_points"].sum()),
                "mean_aligned_points": float(group["common_grid_n_points"].mean()),
                "median_overlap_span_K": float(group["common_T_span_K"].median()),
                "min_overlap_span_K": float(group["common_T_span_K"].min()),
                "max_overlap_span_K": float(group["common_T_span_K"].max()),
                "stable_sign_samples": int(group["seebeck_sign_stable"].sum()),
            }
        )

    priority_df = pd.DataFrame(rows)
    priority_df = priority_df.sort_values(
        by=["usable_samples", "total_aligned_points", "median_overlap_span_K", "DOI"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    priority_df["priority_rank"] = np.arange(1, len(priority_df) + 1)
    priority_df = priority_df[
        [
            "priority_rank",
            "DOI",
            "usable_samples",
            "unique_compositions",
            "compositions",
            "total_aligned_points",
            "mean_aligned_points",
            "median_overlap_span_K",
            "min_overlap_span_K",
            "max_overlap_span_K",
            "stable_sign_samples",
        ]
    ]
    return priority_df


def plot_by_doi(sample_table: pd.DataFrame, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    y_columns = [
        ("sigma_S_per_m", "sigma [S/m]"),
        ("S_V_per_K", "S [V/K]"),
        ("kappa_W_per_mK", "kappa [W/mK]"),
        ("PF_W_per_mK2", "PF [W/mK^2]"),
        ("ZT_calc", "ZT_calc [-]"),
    ]
    cmap = plt.get_cmap("tab10")

    for doi, group in sample_table.groupby("DOI", sort=False):
        fig, axes = plt.subplots(
            nrows=len(y_columns),
            ncols=1,
            figsize=(11, 15),
            sharex=True,
            constrained_layout=True,
        )
        if len(y_columns) == 1:
            axes = [axes]

        legend_handles = []
        legend_labels = []

        for sample_index, row in enumerate(group.itertuples(index=False)):
            color = cmap(sample_index % 10)
            t_grid = np.asarray(safe_parse_list(getattr(row, "T_grid_K")), dtype=float)
            label = f"{getattr(row, 'sample_id')} | {getattr(row, 'composition')}"
            for axis, (column, ylabel) in zip(axes, y_columns):
                values = np.asarray(safe_parse_list(getattr(row, column)), dtype=float)
                handle = axis.plot(t_grid, values, linewidth=1.8, color=color, label=label)[0]
                axis.set_ylabel(ylabel)
                axis.grid(True, alpha=0.25)
                if column == "sigma_S_per_m":
                    legend_handles.append(handle)
                    legend_labels.append(label)

        axes[-1].set_xlabel("Temperature [K]")
        compositions = ", ".join(sorted(group["composition"].astype(str).unique().tolist()))
        fig.suptitle(f"{doi}\ncompositions: {compositions}")
        axes[0].legend(legend_handles, legend_labels, fontsize=8, loc="best")
        fig.savefig(plot_dir / f"{slugify(doi)}.png", dpi=180)
        plt.close(fig)


def stringify_list_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        output[column] = output[column].apply(json.dumps)
    return output


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_dois = set(target_dois(args.doi_summary_csv))
    samples_df = target_samples(args.complete_samples_csv, allowed_dois)
    curves_df = load_target_curves(args.all_curves_csv, set(samples_df["sample_id"].tolist()))

    records = sample_records(curves_df)
    if not records:
        raise SystemExit("no aligned samples were generated")

    sample_table = pd.DataFrame(records).sort_values(
        by=["DOI", "composition", "sample_id"], kind="mergesort"
    )
    point_table = build_point_table(sample_table)
    priority_table = build_doi_priority(sample_table)

    rank_map = priority_table.set_index("DOI")["priority_rank"].to_dict()
    sample_table["doi_priority_rank"] = sample_table["DOI"].map(rank_map)
    point_table["doi_priority_rank"] = point_table["DOI"].map(rank_map)

    plot_by_doi(sample_table, output_dir / DOI_PLOT_DIRNAME)

    list_columns = [
        "T_grid_K",
        "sigma_S_per_m",
        "S_V_per_K",
        "kappa_W_per_mK",
        "PF_W_per_mK2",
        "ZT_calc",
    ]
    sample_table_csv = stringify_list_columns(sample_table, list_columns)

    sample_summary_columns = [
        "doi_priority_rank",
        "DOI",
        "composition",
        "sample_id",
        "si_frac",
        "ge_frac",
        "transport_source",
        "sigma_T_min_K",
        "sigma_T_max_K",
        "sigma_n_raw_points",
        "S_T_min_K",
        "S_T_max_K",
        "S_n_raw_points",
        "kappa_T_min_K",
        "kappa_T_max_K",
        "kappa_n_raw_points",
        "common_T_min_K",
        "common_T_max_K",
        "common_T_span_K",
        "common_grid_n_points",
        "S_sign",
        "seebeck_sign_label",
        "seebeck_sign_stable",
        "PF_min_W_mK2",
        "PF_max_W_mK2",
        "ZT_calc_min",
        "ZT_calc_max",
    ]
    sample_summary = sample_table[sample_summary_columns].copy()

    sample_summary.to_csv(output_dir / "sige_transport_sample_summary.csv", index=False)
    sample_table_csv.to_csv(output_dir / "sige_transport_aligned_curves.csv", index=False)
    point_table.to_csv(output_dir / "sige_transport_aligned_points.csv", index=False)
    priority_table.to_csv(output_dir / "sige_transport_doi_priority.csv", index=False)

    print(f"saved_dir: {output_dir}")
    print(f"target_dois: {len(allowed_dois)}")
    print(f"usable_samples: {len(sample_table)}")
    print(f"aligned_points: {len(point_table)}")
    print(f"plots: {len(list((output_dir / DOI_PLOT_DIRNAME).glob('*.png')))}")


if __name__ == "__main__":
    main()
