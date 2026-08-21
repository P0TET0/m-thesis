#!/usr/bin/env python3
"""Compute PF and ZT from existing exp006 sigma predictions.

This script is deliberately downstream-only: it does not refit tau_eff, retrain an
ML model, or recompute exp006 sigma predictions.  It reconstructs kappa and observed
ZT points from the original property curves with the exp005 Step10/11 rules, then
matches them to each exp006 prediction temperature within a configurable tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.exp005.build_training_dataset_step10 import (
    KAPPA_PROPERTY,
    ZT_PROPERTY,
    aggregate_points,
    build_lookup,
    expand_curves_to_points,
    nearest_property_row,
)
from experiments.exp005.normalize_units_step11 import convert_quantity


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "output" / "exp006_pf_zt_from_sigma_prediction_full_curves"
DEFAULT_PREDICTION = (
    ROOT
    / "experiments"
    / "exp006"
    / "data"
    / "processed"
    / "step5b_test_predictions_default.csv"
)
DEFAULT_CURVE_FILE = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step5_core_properties"
    / "candidate_core_curves_step5.csv"
)
DEFAULT_ANNOTATION = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step9_literature_annotations"
    / "sample_literature_annotations_step9.csv"
)
DEFAULT_METADATA = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step3_fixed"
    / "sample_master_fixed.csv"
)

PREDICTION_REQUIRED = {
    "sample_key",
    "T_K",
    "sigma_pred_S_per_m",
    "sigma_S_per_m",
    "S_uV_per_K",
}
CURVE_REQUIRED = {"sample_key", "x_values_json", "y_values_json"}
DEFAULT_TEMPERATURE_TOLERANCE_K = 2.0
DEFAULT_TEMPERATURE_ROUND_DECIMALS = 3
THRESHOLDS = (0.5, 1.0, 1.5)
EXTREME_ERROR_FACTOR = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-file", type=Path, default=None)
    parser.add_argument("--curve-file", type=Path, default=None)
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument("--metadata-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--temperature-tolerance-K",
        type=float,
        default=DEFAULT_TEMPERATURE_TOLERANCE_K,
        help="Maximum same-sample nearest-temperature difference for kappa and ZT.",
    )
    parser.add_argument(
        "--temperature-round-decimals",
        type=int,
        default=DEFAULT_TEMPERATURE_ROUND_DECIMALS,
        help="Temperature rounding used before duplicate-point median aggregation.",
    )
    parser.add_argument(
        "--skip-workbook",
        action="store_true",
        help="Skip the artifact-tool workbook step (CSV/report/figures are still written).",
    )
    parser.add_argument("--artifact-node", type=Path, default=None)
    parser.add_argument("--artifact-node-modules", type=Path, default=None)
    return parser.parse_args()


def has_columns(path: Path, required: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        return required.issubset(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False


def choose_prediction_file(explicit: Path | None) -> tuple[Path, list[Path]]:
    if explicit is not None:
        path = explicit.resolve()
        if not has_columns(path, PREDICTION_REQUIRED):
            raise ValueError(f"Prediction file lacks required columns: {path}")
        return path, [path]

    patterns = [
        ROOT / "experiments" / "exp006",
        ROOT / "data" / "output",
    ]
    candidates: list[Path] = []
    for base in patterns:
        if base.exists():
            candidates.extend(
                p.resolve()
                for p in base.rglob("*.csv")
                if not p.stem.lower().endswith("_test") and has_columns(p, PREDICTION_REQUIRED)
            )
    candidates = sorted(set(candidates), key=lambda p: str(p).lower())
    preferred = DEFAULT_PREDICTION.resolve()
    if preferred in candidates:
        return preferred, candidates
    default_named = [p for p in candidates if p.name == "step5b_test_predictions_default.csv"]
    if default_named:
        return default_named[0], candidates
    if not candidates:
        raise FileNotFoundError("No exp006 sigma prediction CSV with the required columns was found")
    raise RuntimeError(
        "Prediction candidates were found, but no primary/default file was identifiable. "
        "Pass --prediction-file explicitly."
    )


def choose_curve_file(explicit: Path | None) -> tuple[Path, list[Path]]:
    if explicit is not None:
        path = explicit.resolve()
        if not has_columns(path, CURVE_REQUIRED):
            raise ValueError(f"Curve file lacks required columns: {path}")
        return path, [path]

    preferred = [
        ROOT
        / "data"
        / "output"
        / "starrydata2_step9_literature_annotations"
        / "candidate_core_curves_step9.csv",
        DEFAULT_CURVE_FILE,
        ROOT
        / "data"
        / "output"
        / "starrydata2_step6_material_system"
        / "candidate_core_curves_step6.csv",
    ]
    candidates = [p.resolve() for p in preferred if has_columns(p, CURVE_REQUIRED)]
    if not candidates:
        raise FileNotFoundError("No original property-curve CSV with x/y JSON columns was found")
    return candidates[0], sorted(set(candidates), key=lambda p: str(p).lower())


def first_existing(explicit: Path | None, default: Path) -> Path | None:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    return default.resolve() if default.is_file() else None


def relative(path: Path | None) -> str:
    if path is None:
        return "not used"
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def existing_columns(path: Path, wanted: Sequence[str]) -> list[str]:
    header = set(pd.read_csv(path, nrows=0).columns)
    return [column for column in wanted if column in header]


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = PREDICTION_REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if "config_id" not in df:
        df["config_id"] = "unspecified"
    if "prediction_status" not in df:
        df["prediction_status"] = "unknown"
    if "row_id" not in df:
        df["row_id"] = np.arange(len(df), dtype=int)
    df["sample_key"] = df["sample_key"].astype(str)
    for column in [
        "T_K",
        "T_bin_center_K",
        "sigma_pred_S_per_m",
        "sigma_S_per_m",
        "S_uV_per_K",
    ]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_curve_observations(
    path: Path,
    prediction_sample_keys: set[str],
    temperature_round_decimals: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Expand and aggregate original curves with the exp005 Step10 rules."""
    curves = pd.read_csv(path, low_memory=False)
    missing = CURVE_REQUIRED.difference(curves.columns)
    if missing:
        raise ValueError(f"Missing curve columns: {sorted(missing)}")
    curves["sample_key"] = curves["sample_key"].astype(str)
    source_curve_rows = len(curves)
    source_curve_samples = curves["sample_key"].nunique()
    curves = curves.loc[curves["sample_key"].isin(prediction_sample_keys)].copy()

    points, _curve_audit = expand_curves_to_points(curves)
    aggregated = aggregate_points(points, temperature_round_decimals)
    aggregated = aggregated.loc[
        aggregated["property_step10"].isin([KAPPA_PROPERTY, ZT_PROPERTY])
    ].copy()
    stats = {
        "source_curve_rows": int(source_curve_rows),
        "source_curve_samples": int(source_curve_samples),
        "overlap_curve_rows": int(len(curves)),
        "overlap_curve_samples": int(curves["sample_key"].nunique()),
        "expanded_curve_point_rows": int(len(points)),
        "aggregated_kappa_zt_point_rows": int(len(aggregated)),
        "aggregated_kappa_point_rows": int(
            aggregated["property_step10"].eq(KAPPA_PROPERTY).sum()
        ),
        "aggregated_zt_point_rows": int(aggregated["property_step10"].eq(ZT_PROPERTY).sum()),
    }
    return aggregated, stats


def match_curve_observations(
    predictions: pd.DataFrame,
    aggregated: pd.DataFrame,
    tolerance_K: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Match kappa and ZT independently to exp006 temperatures without interpolation."""
    lookup = build_lookup(aggregated)
    records: list[dict[str, object]] = []
    for row in predictions[["sample_key", "T_K"]].itertuples(index=False):
        record: dict[str, object] = {}
        for property_name, prefix in [(KAPPA_PROPERTY, "kappa"), (ZT_PROPERTY, "zt")]:
            nearest, delta = nearest_property_row(
                lookup,
                str(row.sample_key),
                property_name,
                float(row.T_K),
                tolerance_K,
            )
            if nearest is None:
                record.update(
                    {
                        f"{prefix}_obs_raw": np.nan,
                        f"{prefix}_unit_raw": "",
                        f"{prefix}_matched_temperature_K": np.nan,
                        f"{prefix}_temperature_delta_K": np.nan,
                        f"{prefix}_source_point_count": np.nan,
                        f"{prefix}_source_curve_count": np.nan,
                        f"{prefix}_source_curve_ids": "",
                    }
                )
                continue
            record.update(
                {
                    f"{prefix}_obs_raw": nearest["value_raw_median"],
                    f"{prefix}_unit_raw": nearest["units_y_step10"],
                    f"{prefix}_matched_temperature_K": nearest["temperature_K_rounded"],
                    f"{prefix}_temperature_delta_K": delta,
                    f"{prefix}_source_point_count": nearest["source_point_count_step10"],
                    f"{prefix}_source_curve_count": nearest["source_curve_count_step10"],
                    f"{prefix}_source_curve_ids": nearest["curve_ids_step10"],
                }
            )
        records.append(record)

    matched = pd.DataFrame.from_records(records, index=predictions.index)
    kappa_conversion = convert_quantity(
        matched["kappa_obs_raw"], matched["kappa_unit_raw"], "kappa", True
    ).add_prefix("kappa_")
    zt_conversion = convert_quantity(
        matched["zt_obs_raw"], matched["zt_unit_raw"], "zt", False
    ).add_prefix("zt_")
    matched = pd.concat([matched, kappa_conversion, zt_conversion], axis=1)
    matched["kappa_obs_W_per_mK"] = matched["kappa_converted_value"]
    matched["ZT_obs"] = matched["zt_converted_value"]
    matched["kappa_join_method"] = np.where(
        matched["kappa_matched_temperature_K"].notna(),
        f"same sample nearest temperature within +/-{tolerance_K:g} K",
        f"unmatched within +/-{tolerance_K:g} K",
    )
    matched["zt_join_method"] = np.where(
        matched["zt_matched_temperature_K"].notna(),
        f"same sample nearest temperature within +/-{tolerance_K:g} K",
        f"unmatched within +/-{tolerance_K:g} K",
    )
    matched["observation_join_method"] = (
        "kappa: " + matched["kappa_join_method"] + "; ZT: " + matched["zt_join_method"]
    )

    kappa_raw = matched["kappa_matched_temperature_K"].notna()
    zt_raw = matched["zt_matched_temperature_K"].notna()
    kappa_valid = matched["kappa_obs_W_per_mK"].notna()
    zt_valid = matched["ZT_obs"].notna()
    both_valid = kappa_valid & zt_valid
    counts = {
        "kappa_nearest_match_rows_raw": int(kappa_raw.sum()),
        "kappa_nearest_match_samples_raw": int(
            predictions.loc[kappa_raw, "sample_key"].nunique()
        ),
        "zt_nearest_match_rows_raw": int(zt_raw.sum()),
        "zt_nearest_match_samples_raw": int(predictions.loc[zt_raw, "sample_key"].nunique()),
        "kappa_exact_temperature_match_rows": int(
            matched["kappa_temperature_delta_K"].eq(0).sum()
        ),
        "zt_exact_temperature_match_rows": int(matched["zt_temperature_delta_K"].eq(0).sum()),
        "kappa_unit_valid_rows": int(kappa_valid.sum()),
        "zt_unit_valid_rows": int(zt_valid.sum()),
        "kappa_zt_both_unit_valid_rows": int(both_valid.sum()),
        "kappa_zt_both_unit_valid_samples": int(
            predictions.loc[both_valid, "sample_key"].nunique()
        ),
    }
    return matched, counts


def load_sample_metadata(annotation_path: Path | None, metadata_path: Path | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    common = [
        "sample_key",
        "sample_id",
        "DOI",
        "paper_title",
        "composition",
        "material_system",
        "n_or_p",
        "sintering_method",
        "sintering_checked",
        "sintering_method_checked",
        "record_checked",
        "carrier_type_guess",
    ]
    for source_name, path in [("annotation", annotation_path), ("metadata", metadata_path)]:
        if path is None:
            continue
        cols = existing_columns(path, common)
        if "sample_key" not in cols:
            continue
        frame = pd.read_csv(path, usecols=cols, low_memory=False)
        frame["sample_key"] = frame["sample_key"].astype(str)
        if frame["sample_key"].duplicated().any():
            raise ValueError(f"{source_name} metadata has duplicate sample_key values: {path}")
        frame = frame.set_index("sample_key")
        frame.columns = [f"{c}__{source_name}" for c in frame.columns]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(index=pd.Index([], name="sample_key"))
    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, how="outer")
    return result.reset_index()


def coalesce(df: pd.DataFrame, candidates: Sequence[str], default: object = np.nan) -> pd.Series:
    series: pd.Series | None = None
    for column in candidates:
        if column in df:
            series = df[column] if series is None else series.combine_first(df[column])
    if series is None:
        return pd.Series(default, index=df.index)
    return series


def finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_mape(actual: pd.Series, predicted: pd.Series) -> tuple[float, int]:
    a = finite(actual)
    p = finite(predicted)
    mask = a.notna() & p.notna() & a.ne(0)
    if not mask.any():
        return math.nan, 0
    return float(((p[mask] - a[mask]).abs() / a[mask].abs()).mean() * 100.0), int(mask.sum())


def safe_mae(actual: pd.Series, predicted: pd.Series) -> tuple[float, int]:
    a = finite(actual)
    p = finite(predicted)
    mask = a.notna() & p.notna()
    return (float((p[mask] - a[mask]).abs().mean()), int(mask.sum())) if mask.any() else (math.nan, 0)


def safe_rmse(actual: pd.Series, predicted: pd.Series) -> tuple[float, int]:
    a = finite(actual)
    p = finite(predicted)
    mask = a.notna() & p.notna()
    return (
        float(np.sqrt(np.mean(np.square(p[mask] - a[mask])))),
        int(mask.sum()),
    ) if mask.any() else (math.nan, 0)


def safe_log_metrics(actual: pd.Series, predicted: pd.Series) -> tuple[float, float, int]:
    a = finite(actual)
    p = finite(predicted)
    mask = a.gt(0) & p.gt(0)
    if not mask.any():
        return math.nan, math.nan, 0
    error = np.log10(p[mask] / a[mask])
    return float(np.sqrt(np.mean(np.square(error)))), float(np.mean(np.abs(error))), int(mask.sum())


def confusion(actual: pd.Series, predicted: pd.Series, threshold: float) -> dict[str, float | int]:
    a = finite(actual)
    p = finite(predicted)
    valid = a.notna() & p.notna()
    truth = a[valid].ge(threshold)
    pred = p[valid].ge(threshold)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    return {
        "valid_rows": int(valid.sum()),
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def calculate_rows(
    predictions: pd.DataFrame,
    observation_matches: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    prediction_path: Path,
    curve_path: Path,
    curve_counts: dict[str, int],
    match_counts: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    df = predictions.copy()
    df["temperature_K"] = finite(df["T_K"])
    df["temp_bin_center_K"] = finite(df.get("T_bin_center_K", pd.Series(np.nan, index=df.index)))
    df["sigma_obs_S_per_m"] = finite(df["sigma_S_per_m"])
    df["sigma_pred_exp006_S_per_m"] = finite(df["sigma_pred_S_per_m"])
    df["seebeck_obs_V_per_K"] = finite(df["S_uV_per_K"]) * 1.0e-6
    if len(observation_matches) != len(df) or not observation_matches.index.equals(df.index):
        raise AssertionError("Curve observation matches are not aligned to prediction rows")
    overlapping = sorted(set(df.columns).intersection(observation_matches.columns))
    if overlapping:
        raise ValueError(f"Unexpected duplicate observation columns: {overlapping}")
    df = pd.concat([df, observation_matches], axis=1)

    if not sample_metadata.empty:
        before = len(df)
        df = df.merge(sample_metadata, on="sample_key", how="left", validate="many_to_one")
        if len(df) != before:
            raise AssertionError("Sample metadata merge changed the prediction row count")

    df["sample_id"] = coalesce(
        df,
        ["sample_id__observation", "sample_id__annotation", "sample_id__metadata", "sample_id"],
    )
    df["DOI"] = coalesce(
        df, ["DOI__observation", "DOI", "DOI__annotation", "DOI__metadata", "doi"]
    )
    df["paper_title"] = coalesce(
        df, ["paper_title__observation", "paper_title", "paper_title__annotation"]
    )
    df["composition"] = coalesce(
        df,
        [
            "composition__observation",
            "composition",
            "composition__annotation",
            "composition__metadata",
            "formula_raw",
        ],
    )
    df["material_system"] = coalesce(
        df,
        ["material_system__observation", "material_system", "material_system__annotation", "material_family_raw"],
    )
    df["n_or_p"] = coalesce(
        df,
        ["n_or_p__observation", "n_or_p", "n_or_p__annotation", "carrier_type", "carrier_type_guess__metadata"],
    )
    df["sintering_method"] = coalesce(
        df,
        [
            "sintering_method__observation",
            "sintering_method",
            "sintering_method__annotation",
            "sintering_method__metadata",
        ],
    )
    df["sintering_checked"] = coalesce(
        df,
        [
            "sintering_checked__observation",
            "sintering_checked",
            "sintering_checked__annotation",
            "sintering_method_checked__metadata",
        ],
    )
    df["record_checked"] = coalesce(
        df,
        [
            "record_checked__observation",
            "record_checked",
            "record_checked__annotation",
            "record_checked__metadata",
        ],
    )

    kappa = finite(df.get("kappa_obs_W_per_mK", pd.Series(np.nan, index=df.index)))
    zt_obs = finite(df.get("ZT_obs", pd.Series(np.nan, index=df.index)))
    df["kappa_obs_W_per_mK"] = kappa
    df["ZT_obs"] = zt_obs

    valid_pf_pred = df["seebeck_obs_V_per_K"].notna() & df["sigma_pred_exp006_S_per_m"].gt(0)
    valid_pf_obs = df["seebeck_obs_V_per_K"].notna() & df["sigma_obs_S_per_m"].gt(0)
    df["PF_pred_exp006_W_per_mK2"] = np.where(
        valid_pf_pred,
        np.square(df["seebeck_obs_V_per_K"]) * df["sigma_pred_exp006_S_per_m"],
        np.nan,
    )
    df["PF_obs_W_per_mK2"] = np.where(
        valid_pf_obs,
        np.square(df["seebeck_obs_V_per_K"]) * df["sigma_obs_S_per_m"],
        np.nan,
    )

    valid_zt_pred = valid_pf_pred & df["temperature_K"].gt(0) & kappa.gt(0)
    valid_zt_calc = valid_pf_obs & df["temperature_K"].gt(0) & kappa.gt(0)
    df["ZT_pred_exp006"] = np.where(
        valid_zt_pred,
        np.square(df["seebeck_obs_V_per_K"])
        * df["sigma_pred_exp006_S_per_m"]
        * df["temperature_K"]
        / kappa,
        np.nan,
    )
    df["ZT_calc_from_obs"] = np.where(
        valid_zt_calc,
        np.square(df["seebeck_obs_V_per_K"])
        * df["sigma_obs_S_per_m"]
        * df["temperature_K"]
        / kappa,
        np.nan,
    )

    def errors(prefix: str, actual: str, predicted: str) -> None:
        a = finite(df[actual])
        p = finite(df[predicted])
        valid = a.notna() & p.notna()
        df[f"{prefix}_signed_error"] = np.where(valid, p - a, np.nan)
        df[f"{prefix}_abs_error"] = np.where(valid, np.abs(p - a), np.nan)
        nonzero = valid & a.ne(0)
        df[f"{prefix}_relative_error"] = np.where(nonzero, (p - a) / a, np.nan)
        df[f"{prefix}_absolute_percentage_error_pct"] = np.where(
            nonzero, np.abs((p - a) / a) * 100.0, np.nan
        )
        positive = a.gt(0) & p.gt(0)
        df[f"{prefix}_log10_error"] = np.where(positive, np.log10(p / a), np.nan)

    errors("sigma", "sigma_obs_S_per_m", "sigma_pred_exp006_S_per_m")
    errors("PF", "PF_obs_W_per_mK2", "PF_pred_exp006_W_per_mK2")
    errors("ZT_vs_obs", "ZT_obs", "ZT_pred_exp006")
    errors("ZT_vs_calc", "ZT_calc_from_obs", "ZT_pred_exp006")

    df["prediction_source"] = f"{relative(prediction_path)}:sigma_pred_S_per_m"
    df["seebeck_obs_source"] = f"{relative(prediction_path)}:S_uV_per_K converted by 1e-6"
    df["kappa_zt_obs_source"] = (
        f"{relative(curve_path)}:exp005 Step10 curve expansion/aggregation + "
        "Step11 unit normalization"
    )

    counts = {
        **curve_counts,
        **match_counts,
        "prediction_rows": int(len(df)),
        "valid_pf_pred_rows": int(finite(df["PF_pred_exp006_W_per_mK2"]).notna().sum()),
        "valid_pf_comparison_rows": int(
            (finite(df["PF_pred_exp006_W_per_mK2"]).notna() & finite(df["PF_obs_W_per_mK2"]).notna()).sum()
        ),
        "valid_zt_pred_rows": int(finite(df["ZT_pred_exp006"]).notna().sum()),
        "zt_obs_nonmissing_rows": int(finite(df["ZT_obs"]).notna().sum()),
        "valid_zt_obs_comparison_rows": int(
            (finite(df["ZT_pred_exp006"]).notna() & finite(df["ZT_obs"]).notna()).sum()
        ),
        "valid_zt_calc_comparison_rows": int(
            (finite(df["ZT_pred_exp006"]).notna() & finite(df["ZT_calc_from_obs"]).notna()).sum()
        ),
    }
    return df, counts


def build_overall_metrics(rows: pd.DataFrame, counts: dict[str, int]) -> pd.DataFrame:
    sigma_log_rmse, sigma_mae_log, valid_sigma = safe_log_metrics(
        rows["sigma_obs_S_per_m"], rows["sigma_pred_exp006_S_per_m"]
    )
    sigma_mape, _ = safe_mape(rows["sigma_obs_S_per_m"], rows["sigma_pred_exp006_S_per_m"])
    pf_log_rmse, _, valid_pf = safe_log_metrics(
        rows["PF_obs_W_per_mK2"], rows["PF_pred_exp006_W_per_mK2"]
    )
    pf_mape, _ = safe_mape(rows["PF_obs_W_per_mK2"], rows["PF_pred_exp006_W_per_mK2"])
    zt_obs_mape, valid_zt_obs = safe_mape(rows["ZT_obs"], rows["ZT_pred_exp006"])
    zt_obs_mae, _ = safe_mae(rows["ZT_obs"], rows["ZT_pred_exp006"])
    zt_obs_rmse, _ = safe_rmse(rows["ZT_obs"], rows["ZT_pred_exp006"])
    zt_calc_mape, valid_zt_calc = safe_mape(rows["ZT_calc_from_obs"], rows["ZT_pred_exp006"])
    zt_calc_mae, _ = safe_mae(rows["ZT_calc_from_obs"], rows["ZT_pred_exp006"])
    zt_calc_rmse, _ = safe_rmse(rows["ZT_calc_from_obs"], rows["ZT_pred_exp006"])
    return pd.DataFrame(
        [
            {
                "config_id": rows["config_id"].dropna().iloc[0] if rows["config_id"].notna().any() else "unspecified",
                "prediction_source": rows["prediction_source"].iloc[0],
                **counts,
                "valid_sigma_rows": valid_sigma,
                "sigma_log_RMSE": sigma_log_rmse,
                "sigma_MAE_log": sigma_mae_log,
                "sigma_MAPE_pct": sigma_mape,
                "valid_PF_rows": valid_pf,
                "PF_MAPE_pct": pf_mape,
                "PF_log_RMSE": pf_log_rmse,
                "valid_ZT_obs_rows": valid_zt_obs,
                "ZT_pred_vs_ZT_obs_MAPE_pct": zt_obs_mape,
                "ZT_pred_vs_ZT_obs_MAE": zt_obs_mae,
                "ZT_pred_vs_ZT_obs_RMSE": zt_obs_rmse,
                "valid_ZT_calc_rows": valid_zt_calc,
                "ZT_pred_vs_ZT_calc_MAPE_pct": zt_calc_mape,
                "ZT_pred_vs_ZT_calc_MAE": zt_calc_mae,
                "ZT_pred_vs_ZT_calc_RMSE": zt_calc_rmse,
            }
        ]
    )


def build_sample_summary(rows: pd.DataFrame) -> pd.DataFrame:
    carry = [
        "sample_id",
        "DOI",
        "paper_title",
        "composition",
        "material_system",
        "n_or_p",
        "sintering_method",
        "sintering_checked",
        "record_checked",
        "config_id",
        "prediction_source",
    ]
    aggregations: dict[str, tuple[str, str | callable]] = {
        "max_ZT_obs": ("ZT_obs", "max"),
        "max_ZT_pred_exp006": ("ZT_pred_exp006", "max"),
        "max_ZT_calc_from_obs": ("ZT_calc_from_obs", "max"),
        "mean_ZT_pred_exp006": ("ZT_pred_exp006", "mean"),
        "number_of_temperature_points": ("temperature_K", "nunique"),
        "valid_ZT_pred_points": ("ZT_pred_exp006", "count"),
        "valid_ZT_obs_points": ("ZT_obs", "count"),
        "valid_ZT_calc_points": ("ZT_calc_from_obs", "count"),
    }
    for column in carry:
        aggregations[column] = (column, "first")
    summary = rows.groupby("sample_key", dropna=False).agg(**aggregations).reset_index()
    summary["high_ZT_obs_flag"] = summary["max_ZT_obs"].ge(1.0) & summary["max_ZT_obs"].notna()
    summary["high_ZT_pred_flag"] = (
        summary["max_ZT_pred_exp006"].ge(1.0) & summary["max_ZT_pred_exp006"].notna()
    )
    return summary


def build_classification(rows: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        output.append(
            {
                "evaluation_level": "row",
                "threshold": threshold,
                **confusion(rows["ZT_obs"], rows["ZT_pred_exp006"], threshold),
            }
        )
        output.append(
            {
                "evaluation_level": "sample_max",
                "threshold": threshold,
                **confusion(samples["max_ZT_obs"], samples["max_ZT_pred_exp006"], threshold),
            }
        )
    return pd.DataFrame(output)


def build_ranking(samples: pd.DataFrame) -> pd.DataFrame:
    valid = samples.loc[
        finite(samples["max_ZT_obs"]).notna() & finite(samples["max_ZT_pred_exp006"]).notna(),
        ["sample_key", "max_ZT_obs", "max_ZT_pred_exp006"],
    ].copy()
    rows: list[dict[str, object]] = []
    spearman = (
        float(valid["max_ZT_obs"].rank(method="average").corr(valid["max_ZT_pred_exp006"].rank(method="average")))
        if len(valid) >= 2
        else math.nan
    )
    rows.append(
        {
            "metric": "ranking_correlation",
            "requested_k": np.nan,
            "effective_k": len(valid),
            "intersection_count": np.nan,
            "value": spearman,
            "description": "Spearman rank correlation of sample-level max ZT",
        }
    )
    for requested_k in (50, 100, 300):
        effective_k = min(requested_k, len(valid))
        if effective_k:
            actual_top = set(
                valid.sort_values(["max_ZT_obs", "sample_key"], ascending=[False, True])
                .head(effective_k)["sample_key"]
            )
            predicted_top = set(
                valid.sort_values(["max_ZT_pred_exp006", "sample_key"], ascending=[False, True])
                .head(effective_k)["sample_key"]
            )
            intersection = len(actual_top & predicted_top)
            overlap = intersection / effective_k
        else:
            intersection = 0
            overlap = math.nan
        rows.append(
            {
                "metric": "top_k_overlap",
                "requested_k": requested_k,
                "effective_k": effective_k,
                "intersection_count": intersection,
                "value": overlap,
                "description": "Intersection / effective_k; k reduced when valid sample count is smaller",
            }
        )
    return pd.DataFrame(rows)


def build_problem_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reasons: list[str] = []
    for _, row in rows.iterrows():
        current: list[str] = []
        sigma_pred = row.get("sigma_pred_exp006_S_per_m")
        seebeck = row.get("seebeck_obs_V_per_K")
        kappa = row.get("kappa_obs_W_per_mK")
        temperature = row.get("temperature_K")
        zt_obs = row.get("ZT_obs")
        if pd.isna(sigma_pred):
            current.append("sigma_pred_missing")
        elif sigma_pred <= 0:
            current.append("sigma_pred_nonpositive")
        if pd.isna(seebeck):
            current.append("S_obs_missing")
        if pd.isna(kappa):
            current.append("kappa_obs_missing")
        elif kappa <= 0:
            current.append("kappa_nonpositive")
        if pd.isna(temperature):
            current.append("temperature_missing")
        elif temperature <= 0:
            current.append("temperature_nonpositive")
        if pd.isna(zt_obs):
            current.append("ZT_obs_missing")
        pf_log = row.get("PF_log10_error")
        if pd.notna(pf_log) and abs(pf_log) >= math.log10(EXTREME_ERROR_FACTOR):
            current.append("PF_error_factor_ge_10")
        for label, column in [
            ("ZT_vs_obs_error_factor_ge_10", "ZT_vs_obs_log10_error"),
            ("ZT_vs_calc_error_factor_ge_10", "ZT_vs_calc_log10_error"),
        ]:
            value = row.get(column)
            if pd.notna(value) and abs(value) >= math.log10(EXTREME_ERROR_FACTOR):
                current.append(label)
        reasons.append(";".join(current))
    flagged = rows.copy()
    flagged["problem_reasons"] = reasons
    flagged = flagged.loc[flagged["problem_reasons"].ne("")].copy()
    problem_columns = [
        "row_id",
        "sample_key",
        "sample_id",
        "temperature_K",
        "temp_bin_center_K",
        "material_system",
        "n_or_p",
        "sigma_obs_S_per_m",
        "sigma_pred_exp006_S_per_m",
        "seebeck_obs_V_per_K",
        "kappa_obs_W_per_mK",
        "PF_obs_W_per_mK2",
        "PF_pred_exp006_W_per_mK2",
        "ZT_obs",
        "ZT_calc_from_obs",
        "ZT_pred_exp006",
        "PF_log10_error",
        "ZT_vs_obs_log10_error",
        "ZT_vs_calc_log10_error",
        "prediction_status",
        "config_id",
        "observation_join_method",
        "problem_reasons",
    ]
    flagged = flagged[[c for c in problem_columns if c in flagged]]
    sample_problems = (
        flagged.groupby("sample_key", dropna=False)
        .agg(
            problem_row_count=("problem_reasons", "size"),
            problem_reasons=("problem_reasons", lambda s: ";".join(sorted(set(";".join(s).split(";"))))),
            temperature_min_K=("temperature_K", "min"),
            temperature_max_K=("temperature_K", "max"),
        )
        .reset_index()
    )
    return flagged, sample_problems


def scatter_plot(
    rows: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    output: Path,
    log_scale: bool,
) -> None:
    x = finite(rows[x_column])
    y = finite(rows[y_column])
    mask = x.notna() & y.notna()
    if log_scale:
        mask &= x.gt(0) & y.gt(0)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    if mask.any():
        ax.scatter(x[mask], y[mask], s=16, alpha=0.42, color="#176B87", edgecolors="none")
        low = float(min(x[mask].min(), y[mask].min()))
        high = float(max(x[mask].max(), y[mask].max()))
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1.8, color="#C0392B", label="y = x")
        ax.legend(frameon=False, fontsize=11)
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No valid matched rows", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, which="both", alpha=0.18)
    ax.text(0.02, 0.98, f"n = {int(mask.sum()):,}", transform=ax.transAxes, va="top", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def confusion_plot(classification: pd.DataFrame, output: Path) -> None:
    row = classification.loc[
        classification["evaluation_level"].eq("row") & classification["threshold"].eq(1.0)
    ].iloc[0]
    matrix = np.array(
        [[row["true_negative"], row["false_positive"]], [row["false_negative"], row["true_positive"]]],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    image = ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=17, color="#17202A")
    ax.set_xticks([0, 1], labels=["Predicted < 1", "Predicted ≥ 1"], fontsize=11)
    ax.set_yticks([0, 1], labels=["Observed < 1", "Observed ≥ 1"], fontsize=11)
    ax.set_xlabel("exp006 ZT prediction", fontsize=13)
    ax.set_ylabel("Observed ZT", fontsize=13)
    ax.set_title("High-ZT classification (threshold = 1.0)", fontsize=15, pad=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row count")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def find_artifact_runtime(args: argparse.Namespace) -> tuple[Path, Path]:
    node_candidates = [
        args.artifact_node,
        Path(os.environ["CODEX_ARTIFACT_NODE"]) if os.environ.get("CODEX_ARTIFACT_NODE") else None,
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "node"
        / "bin"
        / "node.exe",
    ]
    modules_candidates = [
        args.artifact_node_modules,
        Path(os.environ["CODEX_ARTIFACT_NODE_MODULES"])
        if os.environ.get("CODEX_ARTIFACT_NODE_MODULES")
        else None,
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "node"
        / "node_modules",
    ]
    node = next((p.resolve() for p in node_candidates if p is not None and p.is_file()), None)
    modules = next((p.resolve() for p in modules_candidates if p is not None and p.is_dir()), None)
    if node is None or modules is None:
        raise RuntimeError(
            "The bundled artifact-tool runtime was not found. Pass --artifact-node and "
            "--artifact-node-modules, or use --skip-workbook."
        )
    return node, modules


def build_workbook(stage_dir: Path, args: argparse.Namespace) -> None:
    node, modules = find_artifact_runtime(args)
    builder = ROOT / "build_exp006_pf_zt_workbook.mjs"
    if not builder.is_file():
        raise FileNotFoundError(builder)
    with tempfile.TemporaryDirectory(prefix="exp006_pfzt_artifact_") as temp_name:
        work = Path(temp_name)
        shutil.copy2(builder, work / builder.name)
        link = work / "node_modules"
        if os.name == "nt":
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(modules)],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Could not create artifact-tool junction: {completed.stderr}")
        else:
            link.symlink_to(modules, target_is_directory=True)
        subprocess.run(
            [
                str(node),
                "--max-old-space-size=8192",
                str(work / builder.name),
                "--input-dir",
                str(stage_dir),
                "--output",
                str(stage_dir / "starrydata2_exp006_pf_zt_from_sigma_prediction.xlsx"),
            ],
            check=True,
        )


def publish_stage(stage_dir: Path, output_dir: Path) -> None:
    """Rename a completed stage, tolerating short Windows post-process file locks."""
    for inspect_log in stage_dir.glob("*.inspect.ndjson"):
        for attempt in range(10):
            try:
                inspect_log.unlink()
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(1.0)
    for attempt in range(20):
        try:
            stage_dir.rename(output_dir)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(1.0)


def format_metric(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def write_report(
    path: Path,
    prediction_path: Path,
    curve_path: Path,
    annotation_path: Path | None,
    metadata_path: Path | None,
    prediction_candidates: Sequence[Path],
    curve_candidates: Sequence[Path],
    temperature_tolerance_K: float,
    temperature_round_decimals: int,
    counts: dict[str, int],
    metrics: pd.DataFrame,
    classification: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    metric = metrics.iloc[0]
    f1_row = classification.loc[
        classification["evaluation_level"].eq("row") & classification["threshold"].eq(1.0), "F1"
    ].iloc[0]
    spearman = ranking.loc[ranking["metric"].eq("ranking_correlation"), "value"].iloc[0]
    lines = [
        "exp006 PF/ZT from existing sigma prediction",
        "=" * 49,
        "",
        "Processing scope",
        "----------------",
        "- Downstream calculation only; exp006 sigma predictions were not recomputed.",
        "- No ML retraining and no tau_eff refitting were performed.",
        "- Existing input files were read-only; all outputs are new files in the requested directory.",
        "",
        "Selected inputs",
        "---------------",
        f"- Prediction file: {relative(prediction_path)}",
        "- Primary selection reason: exp006 Step5B default/recommended configuration file; it contains one config_id and the existing sigma_pred_S_per_m column.",
        f"- sigma prediction column: sigma_pred_S_per_m -> sigma_pred_exp006_S_per_m",
        f"- S observation column: S_uV_per_K -> seebeck_obs_V_per_K (multiplied by 1e-6)",
        f"- Temperature column: T_K -> temperature_K",
        f"- sigma observation column: sigma_S_per_m -> sigma_obs_S_per_m",
        f"- Original property-curve file: {relative(curve_path)}",
        "- Kappa and observed ZT are reconstructed from x_values_json/y_values_json.",
        "- Unit conversion uses the exp005 Step11 conversion rules.",
        f"- Annotation metadata: {relative(annotation_path)}",
        f"- Sample metadata: {relative(metadata_path)}",
        "",
        "Join rule and coverage",
        "----------------------",
        "- S_obs is already in the exp006 prediction file, so no S temperature join was needed.",
        f"- Curve temperatures were rounded to {temperature_round_decimals} decimals, and duplicate same-sample/property/temperature values were aggregated by median.",
        f"- Kappa and ZT were independently matched to each exp006 T_K by the same-sample nearest temperature within +/-{temperature_tolerance_K:g} K.",
        "- No interpolation or extrapolation was used; equal-distance ties select the lower temperature.",
        "- Sample-level descriptive fields were joined by sample_key only; existing source values were copied, not edited.",
        *[f"- {key}: {value:,}" for key, value in counts.items()],
        "",
        "Units and formulas",
        "------------------",
        "- sigma_pred_exp006_S_per_m and sigma_obs_S_per_m: S/m",
        "- seebeck_obs_V_per_K: V/K (converted from microV/K)",
        "- kappa_obs_W_per_mK: W/(m K)",
        "- temperature_K: K",
        "- PF_pred = S_obs^2 * sigma_pred_exp006 [W/(m K^2)]",
        "- ZT_pred = S_obs^2 * sigma_pred_exp006 * T / kappa_obs [dimensionless]",
        "- PF_obs and ZT_calc_from_obs use the same formulas with sigma_obs.",
        "",
        "Exclusions",
        "----------",
        "- PF prediction is not calculated when S_obs is missing or sigma_pred <= 0/missing.",
        "- ZT prediction is additionally not calculated when temperature <= 0/missing or kappa <= 0/missing.",
        "- Error metrics use only finite comparable pairs; MAPE excludes actual zero values and log metrics require both values > 0.",
        f"- Extreme-error flag: prediction/observation differs by a factor of at least {EXTREME_ERROR_FACTOR:g} on a positive pair.",
        "",
        "Primary metrics",
        "---------------",
        f"- sigma log RMSE: {format_metric(metric['sigma_log_RMSE'])}",
        f"- sigma MAE log10: {format_metric(metric['sigma_MAE_log'])}",
        f"- sigma MAPE (%): {format_metric(metric['sigma_MAPE_pct'])}",
        f"- PF MAPE (%): {format_metric(metric['PF_MAPE_pct'])}",
        f"- PF log RMSE: {format_metric(metric['PF_log_RMSE'])}",
        f"- ZT_pred vs ZT_obs MAPE (%): {format_metric(metric['ZT_pred_vs_ZT_obs_MAPE_pct'])}",
        f"- ZT_pred vs ZT_calc_from_obs MAPE (%): {format_metric(metric['ZT_pred_vs_ZT_calc_MAPE_pct'])}",
        f"- ZT >= 1 row-level F1: {format_metric(f1_row)}",
        f"- Sample max-ZT ranking correlation (Spearman): {format_metric(spearman)}",
        "",
        "Candidate search record",
        "-----------------------",
        f"- Prediction candidates satisfying required columns: {len(prediction_candidates)}",
        *[f"  - {relative(p)}" for p in prediction_candidates[:30]],
        f"- Original curve candidates considered: {len(curve_candidates)}",
        *[f"  - {relative(p)}" for p in curve_candidates],
        "",
        "Interpretation",
        "--------------",
        "- This analysis isolates how the already-generated exp006 sigma prediction propagates into PF/ZT when observed S and kappa are held fixed.",
        "- It is not a complete unknown-material ZT model because S and kappa are observed, not predicted.",
        "- Kappa/ZT coverage is expanded from the original curves while preserving the existing exp006 sigma prediction temperatures.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_notes(path: Path, counts: dict[str, int], temperature_tolerance_K: float) -> None:
    text = f"""# exp006 PF/ZT downstream calculation notes

This output reuses the existing exp006 `sigma_pred_S_per_m`. It does not retrain a model, refit `tau_eff`, or replace any source file.

## Equations

- `PF_pred_exp006_W_per_mK2 = seebeck_obs_V_per_K ** 2 * sigma_pred_exp006_S_per_m`
- `ZT_pred_exp006 = PF_pred_exp006_W_per_mK2 * temperature_K / kappa_obs_W_per_mK`
- Observed-reference PF/ZT use the same expressions with `sigma_obs_S_per_m`.

## Data handling

- Seebeck values came from exp006 `S_uV_per_K` and were converted to V/K by multiplying by `1e-6`.
- Original kappa and ZT curves were expanded with the exp005 Step10 rules.
- Kappa and ZT were independently joined to exp006 `T_K` by same-sample nearest temperature within `±{temperature_tolerance_K:g} K`.
- Interpolation and extrapolation were not used.
- Unit normalization and abnormal-ZT-unit rejection use the exp005 Step11 rules.
- PF-computable prediction rows: {counts['valid_pf_pred_rows']:,}.
- ZT-computable prediction rows: {counts['valid_zt_pred_rows']:,}.
- ZT-vs-observed comparable rows: {counts['valid_zt_obs_comparison_rows']:,}.

Use the report and row-level provenance columns to audit temperature matches and unit exclusions.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.temperature_tolerance_K < 0:
        raise ValueError("--temperature-tolerance-K must be nonnegative")
    if args.temperature_round_decimals < 0:
        raise ValueError("--temperature-round-decimals must be nonnegative")
    prediction_path, prediction_candidates = choose_prediction_file(args.prediction_file)
    curve_path, curve_candidates = choose_curve_file(args.curve_file)
    annotation_path = first_existing(args.annotation_file, DEFAULT_ANNOTATION)
    metadata_path = first_existing(args.metadata_file, DEFAULT_METADATA)

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite existing files: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".exp006_pf_zt_stage_", dir=output_dir.parent))
    try:
        figures_dir = stage_dir / "figures"
        figures_dir.mkdir()

        predictions = load_predictions(prediction_path)
        aggregated_observations, curve_counts = load_curve_observations(
            curve_path,
            set(predictions["sample_key"]),
            args.temperature_round_decimals,
        )
        observation_matches, match_counts = match_curve_observations(
            predictions,
            aggregated_observations,
            args.temperature_tolerance_K,
        )
        sample_metadata = load_sample_metadata(annotation_path, metadata_path)
        rows, counts = calculate_rows(
            predictions,
            observation_matches,
            sample_metadata,
            prediction_path,
            curve_path,
            curve_counts,
            match_counts,
        )
        metrics = build_overall_metrics(rows, counts)
        samples = build_sample_summary(rows)
        classification = build_classification(rows, samples)
        ranking = build_ranking(samples)
        problem_rows, problem_samples = build_problem_rows(rows)

        row_columns = [
            "row_id",
            "sample_key",
            "sample_id",
            "DOI",
            "paper_title",
            "composition",
            "temperature_K",
            "temp_bin_center_K",
            "material_system",
            "n_or_p",
            "sintering_method",
            "sintering_checked",
            "record_checked",
            "sigma_obs_S_per_m",
            "sigma_pred_exp006_S_per_m",
            "seebeck_obs_V_per_K",
            "kappa_obs_W_per_mK",
            "kappa_obs_raw",
            "kappa_unit_raw",
            "kappa_unit_norm",
            "kappa_conversion_status",
            "kappa_conversion_factor",
            "kappa_conversion_note",
            "kappa_matched_temperature_K",
            "kappa_temperature_delta_K",
            "kappa_source_point_count",
            "kappa_source_curve_count",
            "kappa_source_curve_ids",
            "kappa_join_method",
            "PF_obs_W_per_mK2",
            "PF_pred_exp006_W_per_mK2",
            "ZT_obs",
            "zt_obs_raw",
            "zt_unit_raw",
            "zt_unit_norm",
            "zt_conversion_status",
            "zt_conversion_factor",
            "zt_conversion_note",
            "zt_unit_abnormal_flag",
            "zt_matched_temperature_K",
            "zt_temperature_delta_K",
            "zt_source_point_count",
            "zt_source_curve_count",
            "zt_source_curve_ids",
            "zt_join_method",
            "ZT_calc_from_obs",
            "ZT_pred_exp006",
            "sigma_signed_error",
            "sigma_abs_error",
            "sigma_relative_error",
            "sigma_absolute_percentage_error_pct",
            "sigma_log10_error",
            "PF_signed_error",
            "PF_abs_error",
            "PF_relative_error",
            "PF_absolute_percentage_error_pct",
            "PF_log10_error",
            "ZT_vs_obs_signed_error",
            "ZT_vs_obs_abs_error",
            "ZT_vs_obs_relative_error",
            "ZT_vs_obs_absolute_percentage_error_pct",
            "ZT_vs_obs_log10_error",
            "ZT_vs_calc_signed_error",
            "ZT_vs_calc_abs_error",
            "ZT_vs_calc_relative_error",
            "ZT_vs_calc_absolute_percentage_error_pct",
            "ZT_vs_calc_log10_error",
            "prediction_status",
            "config_id",
            "prediction_source",
            "seebeck_obs_source",
            "kappa_zt_obs_source",
            "observation_join_method",
        ]
        row_output = rows[[c for c in row_columns if c in rows]].copy()

        write_csv(row_output, stage_dir / "exp006_pf_zt_row_predictions.csv")
        write_csv(samples, stage_dir / "exp006_pf_zt_sample_summary.csv")
        write_csv(metrics, stage_dir / "exp006_pf_zt_overall_metrics.csv")
        write_csv(classification, stage_dir / "exp006_high_zt_classification.csv")
        write_csv(ranking, stage_dir / "exp006_ranking_summary.csv")
        write_csv(problem_rows, stage_dir / "exp006_problem_rows.csv")
        write_csv(problem_samples, stage_dir / "exp006_problem_samples.csv")

        scatter_plot(
            rows,
            "sigma_obs_S_per_m",
            "sigma_pred_exp006_S_per_m",
            "Observed sigma (S/m)",
            "exp006 predicted sigma (S/m)",
            "exp006 sigma prediction vs observation",
            figures_dir / "figure_exp006_sigma_pred_vs_obs.png",
            True,
        )
        scatter_plot(
            rows,
            "PF_obs_W_per_mK2",
            "PF_pred_exp006_W_per_mK2",
            "Observed-reference PF (W/m K²)",
            "exp006 predicted PF (W/m K²)",
            "PF propagated from exp006 sigma prediction",
            figures_dir / "figure_exp006_pf_pred_vs_obs.png",
            True,
        )
        scatter_plot(
            rows,
            "ZT_obs",
            "ZT_pred_exp006",
            "Observed ZT",
            "exp006 predicted ZT",
            "exp006 ZT prediction vs observed ZT",
            figures_dir / "figure_exp006_zt_pred_vs_obs.png",
            False,
        )
        scatter_plot(
            rows,
            "ZT_calc_from_obs",
            "ZT_pred_exp006",
            "ZT calculated from observed sigma",
            "exp006 predicted ZT",
            "exp006 ZT prediction vs observation-derived ZT",
            figures_dir / "figure_exp006_zt_pred_vs_calc.png",
            False,
        )
        confusion_plot(classification, figures_dir / "figure_exp006_high_zt_confusion.png")

        write_report(
            stage_dir / "exp006_pf_zt_report.txt",
            prediction_path,
            curve_path,
            annotation_path,
            metadata_path,
            prediction_candidates,
            curve_candidates,
            args.temperature_tolerance_K,
            args.temperature_round_decimals,
            counts,
            metrics,
            classification,
            ranking,
        )
        write_notes(
            stage_dir / "exp006_pf_zt_notes.md", counts, args.temperature_tolerance_K
        )
        (stage_dir / "exp006_pf_zt_run_manifest.json").write_text(
            json.dumps(
                {
                    "prediction_file": relative(prediction_path),
                    "original_curve_file": relative(curve_path),
                    "annotation_file": relative(annotation_path),
                    "metadata_file": relative(metadata_path),
                    "sigma_prediction_column": "sigma_pred_S_per_m",
                    "seebeck_observation_column": "S_uV_per_K",
                    "temperature_column": "T_K",
                    "curve_x_column": "x_values_json",
                    "curve_y_column": "y_values_json",
                    "kappa_output_column": "kappa_obs_W_per_mK",
                    "zt_output_column": "ZT_obs",
                    "join_key": ["sample_key"],
                    "anchor_temperature_column": "T_K",
                    "temperature_tolerance_K": args.temperature_tolerance_K,
                    "temperature_round_decimals": args.temperature_round_decimals,
                    "duplicate_aggregation": "median",
                    "nearest_temperature_matching_used": True,
                    "interpolation_used": False,
                    "unit_normalization": "exp005 Step11 rules",
                    "counts": counts,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        if not args.skip_workbook:
            build_workbook(stage_dir, args)

        publish_stage(stage_dir, output_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    print(f"Created: {output_dir}")
    print(f"Prediction rows: {counts['prediction_rows']:,}")
    print(f"PF-computable rows: {counts['valid_pf_pred_rows']:,}")
    print(f"ZT-computable rows: {counts['valid_zt_pred_rows']:,}")
    print(f"ZT-vs-observed comparable rows: {counts['valid_zt_obs_comparison_rows']:,}")


if __name__ == "__main__":
    main()
