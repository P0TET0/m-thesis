# Train baseline models to predict log10_sigma from point_master.csv.
import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_DIR = PROJECT_ROOT / "data" / "output" / "master"
INPUT_CSV = MASTER_DIR / "point_master.csv"
OUTPUT_METRICS_CSV = MASTER_DIR / "metrics_baseline_sigma.csv"
OUTPUT_PLOT_PNG = MASTER_DIR / "baseline_sigma_oof_plot.png"
OUTPUT_CURVE_ERROR_CSV = MASTER_DIR / "curve_error_summary.csv"
OUTPUT_CARRIER_METRICS_CSV = MASTER_DIR / "metrics_with_carrier_subset.csv"
OUTPUT_CARRIER_PLOT_PNG = MASTER_DIR / "baseline_sigma_oof_plot_with_carrier.png"
N_SPLITS = 5
RIDGE_ALPHA = 1.0
HUBER_ALPHA = 0.0001
HUBER_EPSILON = 1.35
HUBER_MAX_ITER = 1000


def load_dataset(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "dopant_element" not in df.columns:
        df["dopant_element"] = "unknown"
    df["dopant_element"] = df["dopant_element"].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    df["T_K"] = pd.to_numeric(df["T_K"], errors="coerce")
    df["point_weight"] = pd.to_numeric(df["point_weight"], errors="coerce")
    df["log10_sigma"] = pd.to_numeric(df["log10_sigma"], errors="coerce")
    df["carrier_conc_cm3"] = pd.to_numeric(df.get("carrier_conc_cm3"), errors="coerce")
    df["inv_T"] = np.where(df["T_K"] > 0.0, 1.0 / df["T_K"], np.nan)
    df["log_T"] = np.where(df["T_K"] > 0.0, np.log(df["T_K"]), np.nan)
    df["log10_carrier_conc_cm3"] = np.where(df["carrier_conc_cm3"] > 0.0, np.log10(df["carrier_conc_cm3"]), np.nan)
    return df


def missing_report(df: pd.DataFrame) -> dict[str, int]:
    target_columns = [
        "curve_id",
        "point_weight",
        "T_K",
        "inv_T",
        "log_T",
        "log10_sigma",
        "dopant_element",
        "carrier_conc_cm3",
        "log10_carrier_conc_cm3",
    ]
    return {column: int(df[column].isna().sum()) for column in target_columns if column in df.columns}


def dopant_report(df: pd.DataFrame) -> dict[str, int]:
    counts = df["dopant_element"].fillna("unknown").astype(str).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def prepare_training_frame(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    frame = df.copy()
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["curve_id"] = frame["curve_id"].astype(str)
    frame = frame.dropna(subset=["curve_id", "point_weight", "log10_sigma"])
    frame = frame[frame["point_weight"] > 0.0].copy()
    return frame


def select_subset(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    if subset_name == "all_points":
        return df.copy()
    if subset_name == "carrier_subset":
        return df[df["carrier_conc_cm3"].notna() & (df["carrier_conc_cm3"] > 0.0)].copy()
    raise ValueError(f"unknown subset name: {subset_name}")


def build_pipeline(feature_set: str, regressor_name: str) -> tuple[Pipeline, list[str]]:
    numeric_features = ["T_K", "inv_T", "log_T"]
    transformers: list[tuple[str, Pipeline, list[str]]] = [
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        )
    ]
    feature_columns = numeric_features.copy()

    if feature_set in {"model_B", "model_C"}:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["dopant_element"],
            )
        )
        feature_columns.append("dopant_element")

    if feature_set == "model_C":
        transformers.append(
            (
                "carrier_numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["log10_carrier_conc_cm3"],
            )
        )
        feature_columns.append("log10_carrier_conc_cm3")
    elif feature_set not in {"model_A", "model_B"}:
        raise ValueError(f"unknown feature set: {feature_set}")

    if regressor_name == "Ridge":
        regressor = Ridge(alpha=RIDGE_ALPHA)
    elif regressor_name == "HuberRegressor":
        regressor = HuberRegressor(alpha=HUBER_ALPHA, epsilon=HUBER_EPSILON, max_iter=HUBER_MAX_ITER)
    else:
        raise ValueError(f"unknown regressor: {regressor_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocess", ColumnTransformer(transformers=transformers, sparse_threshold=0.0)),
            ("regressor", regressor),
        ]
    )
    return pipeline, feature_columns


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight))


def build_group_splits(frame: pd.DataFrame, n_splits_limit: int = N_SPLITS) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    n_groups = frame["curve_id"].nunique()
    if n_groups < 2:
        raise ValueError("at least two unique curve_id values are required for GroupKFold")
    n_splits = min(n_splits_limit, n_groups)
    splitter = GroupKFold(n_splits=n_splits)
    X_dummy = np.zeros((len(frame), 1))
    y_dummy = frame["log10_sigma"].to_numpy(dtype=float)
    groups = frame["curve_id"].to_numpy()
    splits = list(splitter.split(X_dummy, y_dummy, groups))
    return splits, n_splits


def evaluate_model(
    df: pd.DataFrame,
    feature_set: str,
    regressor_name: str,
    subset_name: str,
    predefined_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model_name = f"{regressor_name}_{feature_set}"
    pipeline, feature_columns = build_pipeline(feature_set, regressor_name)
    frame = select_subset(df, subset_name)
    frame = prepare_training_frame(frame, feature_columns)
    frame = frame.dropna(subset=["T_K", "inv_T", "log_T"]).copy()
    if feature_set in {"model_B", "model_C"}:
        frame["dopant_element"] = frame["dopant_element"].fillna("unknown").astype(str)
    if feature_set == "model_C":
        frame = frame.dropna(subset=["log10_carrier_conc_cm3"]).copy()

    if predefined_splits is None:
        splits, n_splits = build_group_splits(frame)
    else:
        splits = predefined_splits
        n_splits = len(predefined_splits)

    X = frame[feature_columns]
    y = frame["log10_sigma"].to_numpy(dtype=float)
    groups = frame["curve_id"].to_numpy()
    weights = frame["point_weight"].to_numpy(dtype=float)

    oof_pred = np.full(len(frame), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        w_train = weights[train_idx]
        w_test = weights[test_idx]

        pipeline.fit(X_train, y_train, regressor__sample_weight=w_train)
        y_pred = pipeline.predict(X_test)
        oof_pred[test_idx] = y_pred

        fold_rows.append(
            {
                "subset_name": subset_name,
                "model": model_name,
                "feature_set": feature_set,
                "regressor": regressor_name,
                "fold": fold_index,
                "rmse": weighted_rmse(y_test, y_pred, w_test),
                "mae": weighted_mae(y_test, y_pred, w_test),
                "n_test_points": int(len(test_idx)),
                "n_test_curves": int(pd.Series(groups[test_idx]).nunique()),
            }
        )

    valid_mask = np.isfinite(oof_pred)
    y_valid = y[valid_mask]
    pred_valid = oof_pred[valid_mask]
    weights_valid = weights[valid_mask]
    folds_df = pd.DataFrame(fold_rows)

    metrics = {
        "subset_name": subset_name,
        "model": model_name,
        "feature_set": feature_set,
        "regressor": regressor_name,
        "alpha": RIDGE_ALPHA if regressor_name == "Ridge" else HUBER_ALPHA,
        "epsilon": np.nan if regressor_name == "Ridge" else HUBER_EPSILON,
        "n_rows": int(len(frame)),
        "n_curves": int(frame["curve_id"].nunique()),
        "n_splits": int(n_splits),
        "rmse_mean": float(folds_df["rmse"].mean()),
        "rmse_std": float(folds_df["rmse"].std(ddof=0)),
        "mae_mean": float(folds_df["mae"].mean()),
        "mae_std": float(folds_df["mae"].std(ddof=0)),
        "oof_rmse": weighted_rmse(y_valid, pred_valid, weights_valid),
        "oof_mae": weighted_mae(y_valid, pred_valid, weights_valid),
    }

    predictions_df = frame[
        [
            "curve_id",
            "T_K",
            "log10_sigma",
            "point_weight",
            "dopant_element",
            "carrier_conc_cm3",
            "log10_carrier_conc_cm3",
        ]
    ].copy()
    predictions_df["subset_name"] = subset_name
    predictions_df["model"] = model_name
    predictions_df["feature_set"] = feature_set
    predictions_df["regressor"] = regressor_name
    predictions_df["log10_sigma_pred"] = oof_pred
    predictions_df["abs_error"] = np.abs(predictions_df["log10_sigma"] - predictions_df["log10_sigma_pred"])
    predictions_df["squared_error"] = (predictions_df["log10_sigma"] - predictions_df["log10_sigma_pred"]) ** 2
    return metrics, predictions_df


def build_curve_error_summary(predictions: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    def group_weighted_mae(group: pd.DataFrame) -> float:
        return weighted_mae(
            group["log10_sigma"].to_numpy(dtype=float),
            group["log10_sigma_pred"].to_numpy(dtype=float),
            group["point_weight"].to_numpy(dtype=float),
        )

    def group_weighted_rmse(group: pd.DataFrame) -> float:
        return weighted_rmse(
            group["log10_sigma"].to_numpy(dtype=float),
            group["log10_sigma_pred"].to_numpy(dtype=float),
            group["point_weight"].to_numpy(dtype=float),
        )

    rows: list[dict[str, Any]] = []
    group_columns = ["subset_name", "model", "feature_set", "regressor", "curve_id"]
    for keys, group in predictions.groupby(group_columns, sort=False):
        subset_name, model, feature_set, regressor_name, curve_id = keys
        rows.append(
            {
                "subset_name": subset_name,
                "model": model,
                "feature_set": feature_set,
                "regressor": regressor_name,
                "curve_id": curve_id,
                "n_points": int(len(group)),
                "curve_mae": group_weighted_mae(group),
                "curve_rmse": group_weighted_rmse(group),
            }
        )

    curve_error_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    curve_error_df.to_csv(output_csv, index=False)
    return curve_error_df


def save_metrics(metrics_rows: list[dict[str, Any]], output_csv: Path) -> pd.DataFrame:
    metrics_df = pd.DataFrame(metrics_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_csv, index=False)
    return metrics_df


def save_plot(predictions: pd.DataFrame, output_png: Path, title_prefix: str = "") -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    model_order = list(predictions["model"].drop_duplicates())
    color_map = {
        "P": "tab:blue",
        "B": "tab:orange",
        "unknown": "tab:gray",
    }
    fig, axes = plt.subplots(1, len(model_order), figsize=(6 * len(model_order), 5), squeeze=False)
    for axis, model_name in zip(axes[0], model_order):
        subset = predictions[predictions["model"] == model_name]
        for dopant_element, group in subset.groupby("dopant_element", dropna=False):
            label = str(dopant_element).strip() if pd.notna(dopant_element) and str(dopant_element).strip() else "unknown"
            axis.scatter(
                group["log10_sigma"],
                group["log10_sigma_pred"],
                s=22,
                alpha=0.78,
                color=color_map.get(label, "tab:green"),
                label=label,
            )
        lower = float(min(subset["log10_sigma"].min(), subset["log10_sigma_pred"].min()))
        upper = float(max(subset["log10_sigma"].max(), subset["log10_sigma_pred"].max()))
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1.0)
        axis.set_title(f"{title_prefix}{model_name}")
        axis.set_xlabel("Observed log10_sigma")
        axis.set_ylabel("Predicted log10_sigma")
        axis.grid(True, alpha=0.3)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def print_report(df: pd.DataFrame, metrics_df: pd.DataFrame, carrier_metrics_df: pd.DataFrame) -> None:
    print("data_summary")
    print(f"rows_used_source: {len(df)}")
    print(f"curve_id_count: {df['curve_id'].nunique()}")
    carrier_subset = select_subset(df, "carrier_subset")
    print(f"carrier_subset_rows: {len(carrier_subset)}")
    print(f"carrier_subset_curve_id_count: {carrier_subset['curve_id'].nunique()}")
    print("dopant_element_counts:")
    for key, value in dopant_report(df).items():
        print(f"  {key}: {value}")
    print("missing_values:")
    for key, value in missing_report(df).items():
        print(f"  {key}: {value}")
    print("model_metrics_all_points:")
    print(metrics_df.to_string(index=False))
    print("model_metrics_carrier_subset:")
    print(carrier_metrics_df.to_string(index=False))
    print("carrier_subset_comparison:")
    for regressor_name in carrier_metrics_df["regressor"].drop_duplicates():
        subset = carrier_metrics_df[carrier_metrics_df["regressor"] == regressor_name].set_index("feature_set")
        if {"model_B", "model_C"}.issubset(subset.index):
            delta_rmse = subset.loc["model_C", "oof_rmse"] - subset.loc["model_B", "oof_rmse"]
            delta_mae = subset.loc["model_C", "oof_mae"] - subset.loc["model_B", "oof_mae"]
            print(f"  {regressor_name}: delta_oof_rmse_model_C_minus_B={delta_rmse:.6f}")
            print(f"  {regressor_name}: delta_oof_mae_model_C_minus_B={delta_mae:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline models for log10_sigma prediction")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV, help="input point master csv")
    parser.add_argument("--output-metrics-csv", type=Path, default=OUTPUT_METRICS_CSV, help="output metrics csv for all-point models")
    parser.add_argument("--output-plot-png", type=Path, default=OUTPUT_PLOT_PNG, help="output oof plot png for all-point models")
    parser.add_argument("--output-curve-error-csv", type=Path, default=OUTPUT_CURVE_ERROR_CSV, help="output per-curve error summary csv")
    parser.add_argument("--output-carrier-metrics-csv", type=Path, default=OUTPUT_CARRIER_METRICS_CSV, help="output metrics csv on carrier subset")
    parser.add_argument("--output-carrier-plot-png", type=Path, default=OUTPUT_CARRIER_PLOT_PNG, help="output oof plot png on carrier subset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        df = load_dataset(args.input_csv)

        all_point_specs = [
            ("model_A", "Ridge"),
            ("model_B", "Ridge"),
            ("model_A", "HuberRegressor"),
            ("model_B", "HuberRegressor"),
        ]
        metrics_rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []
        for feature_set, regressor_name in all_point_specs:
            metrics, predictions = evaluate_model(df, feature_set, regressor_name, subset_name="all_points")
            metrics_rows.append(metrics)
            prediction_frames.append(predictions)
        metrics_df = save_metrics(metrics_rows, args.output_metrics_csv)
        predictions_df = pd.concat(prediction_frames, ignore_index=True)
        save_plot(predictions_df, args.output_plot_png)

        carrier_frame = prepare_training_frame(select_subset(df, "carrier_subset"), ["T_K", "inv_T", "log_T", "dopant_element", "log10_carrier_conc_cm3"])
        carrier_frame = carrier_frame.dropna(subset=["T_K", "inv_T", "log_T", "log10_carrier_conc_cm3"]).copy()
        carrier_splits, _ = build_group_splits(carrier_frame)

        carrier_specs = [
            ("model_B", "Ridge"),
            ("model_C", "Ridge"),
            ("model_B", "HuberRegressor"),
            ("model_C", "HuberRegressor"),
        ]
        carrier_metrics_rows: list[dict[str, Any]] = []
        carrier_prediction_frames: list[pd.DataFrame] = []
        for feature_set, regressor_name in carrier_specs:
            metrics, predictions = evaluate_model(
                df,
                feature_set,
                regressor_name,
                subset_name="carrier_subset",
                predefined_splits=carrier_splits,
            )
            carrier_metrics_rows.append(metrics)
            carrier_prediction_frames.append(predictions)
        carrier_metrics_df = save_metrics(carrier_metrics_rows, args.output_carrier_metrics_csv)
        carrier_predictions_df = pd.concat(carrier_prediction_frames, ignore_index=True)
        save_plot(carrier_predictions_df, args.output_carrier_plot_png, title_prefix="carrier_subset: ")

        curve_error_df = build_curve_error_summary(
            pd.concat([predictions_df, carrier_predictions_df], ignore_index=True),
            args.output_curve_error_csv,
        )
    except Exception as exc:
        raise SystemExit(f"failed to train baseline sigma models: {exc}") from exc

    print_report(df, metrics_df, carrier_metrics_df)
    print(f"curve_error_rows: {len(curve_error_df)}")
    print(f"saved_metrics: {args.output_metrics_csv}")
    print(f"saved_plot: {args.output_plot_png}")
    print(f"saved_curve_error_summary: {args.output_curve_error_csv}")
    print(f"saved_carrier_subset_metrics: {args.output_carrier_metrics_csv}")
    print(f"saved_carrier_subset_plot: {args.output_carrier_plot_png}")


if __name__ == "__main__":
    main()
