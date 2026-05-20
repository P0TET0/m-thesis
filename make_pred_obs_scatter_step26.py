import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_STEP22_DIR = "data/output/starrydata2_step22_fitting_vs_ml_comparison"
DEFAULT_STEP21_DIR = "data/output/starrydata2_step21_pf_zt_ml_prediction"
DEFAULT_STEP14_DIR = "data/output/starrydata2_step14_pf_zt_prediction"
DEFAULT_OUTPUT_DIR = "data/output/starrydata2_step26_pred_obs_scatter"
EXCEL_PREVIEW_ROWS = 100_000
EPS = 1e-12

STRING_COLUMNS = [
    "sample_key",
    "SID",
    "DOI",
    "doi_url",
    "sample_id",
    "composition",
    "material_system",
    "n_or_p",
]

PLOT_SPECS = [
    {
        "plot_id": "sigma_direct_fitting",
        "quantity": "sigma",
        "version": "direct_fitting",
        "obs_key": "sigma_obs",
        "pred_key": "sigma_fitting",
        "axis_scale": "log-log",
        "figure": "figure_01_sigma_pred_vs_obs_fitting.png",
        "label": "sigma direct fitting",
        "unit": "S/m",
        "positive_only": True,
    },
    {
        "plot_id": "sigma_ml_tau_prediction",
        "quantity": "sigma",
        "version": "ml_tau_prediction",
        "obs_key": "sigma_obs",
        "pred_key": "sigma_ml",
        "axis_scale": "log-log",
        "figure": "figure_02_sigma_pred_vs_obs_ml.png",
        "label": "sigma ML tau prediction",
        "unit": "S/m",
        "positive_only": True,
    },
    {
        "plot_id": "PF_direct_fitting",
        "quantity": "PF",
        "version": "direct_fitting",
        "obs_key": "pf_obs",
        "pred_key": "pf_fitting",
        "axis_scale": "log-log",
        "figure": "figure_03_pf_pred_vs_obs_fitting.png",
        "label": "PF direct fitting",
        "unit": "W/(m K^2)",
        "positive_only": True,
    },
    {
        "plot_id": "PF_ml_tau_prediction",
        "quantity": "PF",
        "version": "ml_tau_prediction",
        "obs_key": "pf_obs",
        "pred_key": "pf_ml",
        "axis_scale": "log-log",
        "figure": "figure_04_pf_pred_vs_obs_ml.png",
        "label": "PF ML tau prediction",
        "unit": "W/(m K^2)",
        "positive_only": True,
    },
    {
        "plot_id": "ZT_direct_fitting",
        "quantity": "ZT",
        "version": "direct_fitting",
        "obs_key": "zt_obs",
        "pred_key": "zt_fitting",
        "axis_scale": "linear",
        "figure": "figure_05_zt_pred_vs_obs_fitting.png",
        "label": "ZT direct fitting",
        "unit": "dimensionless",
        "positive_only": False,
    },
    {
        "plot_id": "ZT_ml_tau_prediction",
        "quantity": "ZT",
        "version": "ml_tau_prediction",
        "obs_key": "zt_obs",
        "pred_key": "zt_ml",
        "axis_scale": "linear",
        "figure": "figure_06_zt_pred_vs_obs_ml.png",
        "label": "ZT ML tau prediction",
        "unit": "dimensionless",
        "positive_only": False,
    },
]

ZT_LOG_SPECS = [
    {
        "plot_id": "ZT_direct_fitting_log_positive_only",
        "quantity": "ZT",
        "version": "direct_fitting",
        "obs_key": "zt_obs",
        "pred_key": "zt_fitting",
        "axis_scale": "log-log",
        "figure": "figure_08_zt_pred_vs_obs_fitting_log_positive_only.png",
        "label": "ZT direct fitting positive only",
        "unit": "dimensionless",
        "positive_only": True,
    },
    {
        "plot_id": "ZT_ml_tau_prediction_log_positive_only",
        "quantity": "ZT",
        "version": "ml_tau_prediction",
        "obs_key": "zt_obs",
        "pred_key": "zt_ml",
        "axis_scale": "log-log",
        "figure": "figure_09_zt_pred_vs_obs_ml_log_positive_only.png",
        "label": "ZT ML tau prediction positive only",
        "unit": "dimensionless",
        "positive_only": True,
    },
]

COLUMN_CANDIDATES = {
    "sigma_obs": ["sigma_obs_S_per_m_step11"],
    "sigma_fitting": ["sigma_pred_S_per_m_step12"],
    "sigma_ml": ["sigma_pred_ML_for_pf_zt_S_per_m_step21", "sigma_pred_ML_S_per_m_step20"],
    "pf_obs": [
        "power_factor_obs_W_per_mK2_step21",
        "power_factor_obs_W_per_mK2_step14",
        "power_factor_obs_W_per_mK2_step11",
    ],
    "pf_fitting": ["power_factor_pred_fitting_W_per_mK2_step21", "power_factor_pred_W_per_mK2_step14"],
    "pf_ml": ["power_factor_pred_ML_W_per_mK2_step21"],
    "zt_obs": ["zt_obs_dimensionless_step11"],
    "zt_fitting": ["zt_pred_fitting_step21", "zt_pred_from_sigma_step14"],
    "zt_ml": ["zt_pred_ML_step21"],
}

FUZZY_TOKENS = {
    "sigma_obs": ["sigma", "obs"],
    "sigma_fitting": ["sigma", "pred"],
    "sigma_ml": ["sigma", "pred", "ml"],
    "pf_obs": ["factor", "obs"],
    "pf_fitting": ["factor", "pred"],
    "pf_ml": ["factor", "pred", "ml"],
    "zt_obs": ["zt", "obs"],
    "zt_fitting": ["zt", "pred"],
    "zt_ml": ["zt", "pred", "ml"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create Step26 predicted-vs-observed scatter plots.")
    parser.add_argument("--step22_dir", default=DEFAULT_STEP22_DIR)
    parser.add_argument("--step21_dir", default=DEFAULT_STEP21_DIR)
    parser.add_argument("--step14_dir", default=DEFAULT_STEP14_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_points_per_plot", type=int, default=50_000)
    return parser.parse_args()


def dtype_for_existing(path):
    header = pd.read_csv(path, nrows=0)
    return {col: "string" for col in STRING_COLUMNS if col in header.columns}


def read_csv_if_exists(path, loaded, missing, required=False):
    path = Path(path)
    if not path.exists():
        missing.append(str(path))
        if required:
            raise FileNotFoundError(f"Required input file not found: {path}")
        return None
    loaded.append(str(path))
    return pd.read_csv(path, dtype=dtype_for_existing(path), low_memory=False)


def choose_input_df(args, loaded, missing):
    step22_row = Path(args.step22_dir) / "step22_row_level_comparison.csv"
    df = read_csv_if_exists(step22_row, loaded, missing, required=False)
    if df is not None:
        return df, str(step22_row)

    fallback_paths = [
        Path(args.step21_dir) / "thermoelectric_ml_primary_test_predictions_step21.csv",
        Path(args.step14_dir) / "thermoelectric_predictions_step14.csv",
    ]
    frames = [read_csv_if_exists(path, loaded, missing, required=False) for path in fallback_paths]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise FileNotFoundError("No Step22, Step21, or Step14 row-level input file was found.")
    return pd.concat(frames, ignore_index=True, sort=False), "fallback Step21/Step14 concatenation"


def find_column(df, key, report_lines):
    for col in COLUMN_CANDIDATES[key]:
        if col in df.columns:
            report_lines.append(f"{key}: selected exact column {col}")
            return col
    tokens = FUZZY_TOKENS[key]
    matches = [col for col in df.columns if all(token.lower() in col.lower() for token in tokens)]
    report_lines.append(f"{key}: exact candidates missing; fuzzy candidates={matches[:20]}")
    return matches[0] if matches else None


def numeric_series(df, col):
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def finite_mask(obs, pred, positive_only):
    mask = np.isfinite(obs) & np.isfinite(pred)
    if positive_only:
        mask &= (obs > 0) & (pred > 0)
    return mask


def r2_score(obs, pred):
    if len(obs) < 2:
        return np.nan
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((obs - pred) ** 2) / denom)


def corr_value(obs, pred, method="pearson"):
    if len(obs) < 2:
        return np.nan
    return float(pd.Series(obs).corr(pd.Series(pred), method=method))


def compute_metrics(data, plot_id, quantity, version, note=""):
    n_rows = int(len(data))
    n_samples = int(data["sample_key"].nunique()) if "sample_key" in data.columns and n_rows else 0
    if n_rows == 0:
        return {
            "plot_id": plot_id,
            "quantity": quantity,
            "version": version,
            "n_rows": 0,
            "n_samples": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "log_mae": np.nan,
            "log_rmse": np.nan,
            "r2": np.nan,
            "log_r2": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
            "within_25pct_rate": np.nan,
            "within_50pct_rate": np.nan,
            "within_factor_2_rate": np.nan,
            "note": note,
        }
    obs = data["obs_value"].to_numpy(dtype=float)
    pred = data["pred_value"].to_numpy(dtype=float)
    err = pred - obs
    rel = np.abs(err) / np.maximum(np.abs(obs), EPS)
    pos = (obs > 0) & (pred > 0) & np.isfinite(obs) & np.isfinite(pred)
    log_err = np.log(pred[pos]) - np.log(obs[pos]) if pos.any() else np.array([])
    ratio = pred[pos] / obs[pos] if pos.any() else np.array([])
    return {
        "plot_id": plot_id,
        "quantity": quantity,
        "version": version,
        "n_rows": n_rows,
        "n_samples": n_samples,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "mape": float(np.mean(rel)),
        "log_mae": float(np.mean(np.abs(log_err))) if len(log_err) else np.nan,
        "log_rmse": float(math.sqrt(np.mean(log_err**2))) if len(log_err) else np.nan,
        "r2": r2_score(obs, pred),
        "log_r2": r2_score(np.log(obs[pos]), np.log(pred[pos])) if int(pos.sum()) >= 2 else np.nan,
        "pearson": corr_value(obs, pred, "pearson"),
        "spearman": corr_value(pd.Series(obs).rank().to_numpy(), pd.Series(pred).rank().to_numpy(), "pearson"),
        "within_25pct_rate": float(np.mean(rel <= 0.25)),
        "within_50pct_rate": float(np.mean(rel <= 0.50)),
        "within_factor_2_rate": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))) if len(ratio) else np.nan,
        "note": note,
    }


def make_plot_data(df, spec, selected_cols):
    obs_col = selected_cols.get(spec["obs_key"])
    pred_col = selected_cols.get(spec["pred_key"])
    if obs_col is None or pred_col is None:
        return pd.DataFrame(), f"missing columns obs={obs_col}, pred={pred_col}", 0, 0

    obs = numeric_series(df, obs_col)
    pred = numeric_series(df, pred_col)
    finite_pair = np.isfinite(obs) & np.isfinite(pred)
    valid = finite_mask(obs, pred, spec["positive_only"])
    excluded = int(finite_pair.sum() - valid.sum())
    total_nonfinite_or_missing = int(len(df) - finite_pair.sum())

    meta_cols = [
        "sample_key",
        "temperature_K",
        "material_system",
        "n_or_p",
        "sintering_method",
        "sintering_checked",
        "record_checked",
        "DOI",
        "sample_id",
        "composition",
        "doi_url",
    ]
    out = pd.DataFrame(index=df.index[valid])
    out["plot_id"] = spec["plot_id"]
    out["quantity"] = spec["quantity"]
    out["version"] = spec["version"]
    out["obs_value"] = obs[valid].to_numpy()
    out["pred_value"] = pred[valid].to_numpy()
    for col in meta_cols:
        out[col] = df.loc[valid, col].to_numpy() if col in df.columns else pd.NA
    ordered = [
        "plot_id",
        "quantity",
        "version",
        "obs_value",
        "pred_value",
        "sample_key",
        "temperature_K",
        "material_system",
        "n_or_p",
        "sintering_method",
        "sintering_checked",
        "record_checked",
        "DOI",
        "sample_id",
        "composition",
        "doi_url",
    ]
    note = (
        f"obs_col={obs_col}; pred_col={pred_col}; excluded_by_scale={excluded}; "
        f"nonfinite_or_missing_pairs={total_nonfinite_or_missing}"
    )
    return out[ordered], note, excluded, total_nonfinite_or_missing


def sample_for_plot(data, max_points):
    if max_points is None or max_points <= 0 or len(data) <= max_points:
        return data, False
    return data.sample(n=max_points, random_state=26), True


def axis_limits(obs, pred, log_scale):
    vals = np.concatenate([np.asarray(obs, dtype=float), np.asarray(pred, dtype=float)])
    vals = vals[np.isfinite(vals)]
    if log_scale:
        vals = vals[vals > 0]
    if len(vals) == 0:
        return None
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if log_scale:
        if lo <= 0 or hi <= 0:
            return None
        pad = 10 ** (0.06 * max(math.log10(hi) - math.log10(lo), 1e-9))
        return lo / pad, hi * pad
    pad = 0.06 * max(hi - lo, 1e-9)
    return lo - pad, hi + pad


def draw_scatter(ax, data, spec, metric, title_prefix=""):
    log_scale = spec["axis_scale"] == "log-log"
    ax.scatter(data["obs_value"], data["pred_value"], s=12, alpha=0.38, linewidths=0, color="#2563eb")
    limits = axis_limits(data["obs_value"], data["pred_value"], log_scale)
    if limits is not None:
        ax.plot(limits, limits, color="#111827", linewidth=1.2, linestyle="--", label="y = x")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.7)
    title = f"{title_prefix}{spec['quantity']} {spec['version'].replace('_', ' ')}".strip()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"Observed {spec['quantity']} ({spec['unit']})", fontsize=9)
    ax.set_ylabel(f"Predicted {spec['quantity']} ({spec['unit']})", fontsize=9)
    text = (
        f"n={metric.get('n_rows', 0)}\n"
        f"MAPE={format_metric(metric.get('mape'))}\n"
        f"log RMSE={format_metric(metric.get('log_rmse'))}\n"
        f"Spearman={format_metric(metric.get('spearman'))}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.86},
    )


def format_metric(value):
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.2e}"
    return f"{value:.3g}"


def write_notes(path):
    text = """# Step26 Predicted vs Observed Scatter Notes

## Purpose

Step26 visualizes existing predicted values against experimental values for sigma, PF, and ZT.
It does not run new prediction, tau_eff refitting, or ML retraining.

## How to Read the Figures

The horizontal axis is the observed experimental value and the vertical axis is the predicted value.
The dashed diagonal line is y = x. Points closer to this line indicate better agreement.
Sigma and PF are shown on log-log axes because their ranges are wide. ZT is shown on linear axes,
with optional positive-only log-log figures.

## Direct Fitting Version

The direct fitting version uses the fitted tau_eff-derived predictions already produced in earlier steps.
Direct fitting uses sigma observations to fit tau_eff, so it is expected to perform better.

## ML Version

The ML version uses predictions based on the ML-predicted tau_eff columns already produced in earlier steps.
ML version predicts tau_eff from material features and is closer to unknown-material screening.

## Metrics

The metrics table reports MAE, RMSE, MAPE, log MAE, log RMSE, R2, log R2, Pearson, Spearman,
and rates within 25%, 50%, and a factor of 2. Log metrics are computed only where both observed
and predicted values are positive.

## Important Caveats

Direct fitting uses sigma observations to fit tau_eff, so it is expected to perform better.
ML version predicts tau_eff from material features and is closer to unknown-material screening.
S and kappa are not predicted; PF and ZT use observed S and kappa.
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.
"""
    path.write_text(text, encoding="utf-8")


def write_report(path, args, loaded, missing, selected_cols, plot_list, metrics, figures_created, figures_skipped, notes):
    lines = [
        "Step26 Predicted vs Observed Scatter Report",
        "",
        f"Input file used: {notes['input_used']}",
        f"Output directory: {args.output_dir}",
        "",
        "Loaded inputs:",
        *[f"- {item}" for item in loaded],
        "",
        "Missing optional inputs:",
        *[f"- {item}" for item in missing],
        "",
        "Column selection:",
        *[f"- {key}: {value}" for key, value in selected_cols.items()],
        "",
        "Column selection notes:",
        *[f"- {line}" for line in notes["column_notes"]],
        "",
        "Figures created:",
        *[f"- {item}" for item in figures_created],
        "",
        "Figures skipped and reason:",
        *[f"- {item}" for item in figures_skipped],
        "",
        "Rows used for each plot:",
    ]
    for _, row in plot_list.iterrows():
        lines.append(
            f"- {row['plot_id']}: n_rows={row['n_rows']}, n_plotted_rows={row['n_plotted_rows']}, "
            f"status={row['status']}, note={row['note']}"
        )
    lines.extend(["", "Metrics for each plot:"])
    for _, row in metrics.iterrows():
        lines.append(
            f"- {row['plot_id']}: n={row['n_rows']}, MAPE={format_metric(row['mape'])}, "
            f"log_RMSE={format_metric(row['log_rmse'])}, Spearman={format_metric(row['spearman'])}, "
            f"within_factor_2={format_metric(row['within_factor_2_rate'])}"
        )
    lines.extend(["", "Sampling information:"])
    lines.extend([f"- {line}" for line in notes["sampling_notes"]])
    lines.extend(
        [
            "",
            "Excluded rows:",
            *[f"- {line}" for line in notes["exclusion_notes"]],
            "",
            "Main interpretation:",
            "- The direct fitting version tends to be closer to the diagonal than the ML version when its error metrics are smaller.",
            "- The ML version is a harder task because tau_eff is predicted from material features.",
            "- See the metrics table to identify whether sigma, PF, or ZT has the largest errors.",
            "",
            "Important caveats:",
            "- Direct fitting uses sigma observations to fit tau_eff, so it is expected to perform better.",
            "- ML version predicts tau_eff from material features and is closer to unknown-material screening.",
            "- S and kappa are not predicted; PF and ZT use observed S and kappa.",
            "- tau_eff is a relative effective scalar, not a physical relaxation time in seconds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_lookup(metrics_df):
    return {row["plot_id"]: row.to_dict() for _, row in metrics_df.iterrows()}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures_step26"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    missing = []
    input_df, input_used = choose_input_df(args, loaded, missing)
    read_csv_if_exists(Path(args.step21_dir) / "thermoelectric_ml_primary_test_predictions_step21.csv", loaded, missing)
    read_csv_if_exists(Path(args.step14_dir) / "thermoelectric_predictions_step14.csv", loaded, missing)
    read_csv_if_exists(
        Path("data/output/starrydata2_step25_paper_outputs") / "paper_table_07_fitting_vs_ml_comparison_step25.csv",
        loaded,
        missing,
    )

    column_notes = []
    selected_cols = {key: find_column(input_df, key, column_notes) for key in COLUMN_CANDIDATES}

    scatter_frames = []
    metrics_rows = []
    plot_rows = []
    exclusion_notes = []
    sampling_notes = []

    for spec in PLOT_SPECS:
        data, note, excluded, nonfinite = make_plot_data(input_df, spec, selected_cols)
        exclusion_notes.append(
            f"{spec['plot_id']}: excluded_by_scale={excluded}; nonfinite_or_missing_pairs={nonfinite}; {note}"
        )
        metric = compute_metrics(data, spec["plot_id"], spec["quantity"], spec["version"], note=note)
        metrics_rows.append(metric)
        scatter_frames.append(data)
        n_plotted = min(len(data), args.max_points_per_plot) if len(data) else 0
        sampling = len(data) > args.max_points_per_plot
        sampling_notes.append(
            f"{spec['plot_id']}: original rows={len(data)}, plotted rows={n_plotted}, sampling used={sampling}"
        )
        status = "ready" if len(data) else "skipped"
        plot_rows.append(
            {
                "plot_id": spec["plot_id"],
                "quantity": spec["quantity"],
                "version": spec["version"],
                "figure_file": str(Path("figures_step26") / spec["figure"]),
                "axis_scale": spec["axis_scale"],
                "n_rows": len(data),
                "n_plotted_rows": n_plotted,
                "status": status,
                "note": note if len(data) else f"skipped: {note}",
            }
        )

    scatter_data = pd.concat(scatter_frames, ignore_index=True) if scatter_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metrics_rows)
    plot_list_df = pd.DataFrame(plot_rows)

    figures_created = []
    figures_skipped = []
    matplotlib_ok = True
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on environment
        matplotlib_ok = False
        figures_skipped.append(f"all PNG figures: matplotlib unavailable: {exc}")

    if matplotlib_ok:
        metrics_by_plot = metric_lookup(metrics_df)
        for spec in PLOT_SPECS + ZT_LOG_SPECS:
            data, note, _, _ = make_plot_data(input_df, spec, selected_cols)
            if data.empty:
                figures_skipped.append(f"{spec['figure']}: no valid rows; {note}")
                if spec["plot_id"] not in set(plot_list_df["plot_id"]):
                    plot_list_df.loc[len(plot_list_df)] = {
                        "plot_id": spec["plot_id"],
                        "quantity": spec["quantity"],
                        "version": spec["version"],
                        "figure_file": str(Path("figures_step26") / spec["figure"]),
                        "axis_scale": spec["axis_scale"],
                        "n_rows": 0,
                        "n_plotted_rows": 0,
                        "status": "skipped",
                        "note": f"skipped: {note}",
                    }
                continue
            plot_data, sampling = sample_for_plot(data, args.max_points_per_plot)
            metric = metrics_by_plot.get(spec["plot_id"]) or compute_metrics(
                data, spec["plot_id"], spec["quantity"], spec["version"], note=note
            )
            fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=180)
            draw_scatter(ax, plot_data, spec, metric)
            fig.tight_layout()
            out_path = figures_dir / spec["figure"]
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            figures_created.append(str(Path("figures_step26") / spec["figure"]))
            if spec["plot_id"] in set(plot_list_df["plot_id"]):
                plot_list_df.loc[plot_list_df["plot_id"] == spec["plot_id"], "status"] = "created"
                plot_list_df.loc[plot_list_df["plot_id"] == spec["plot_id"], "n_plotted_rows"] = len(plot_data)
                if sampling:
                    plot_list_df.loc[plot_list_df["plot_id"] == spec["plot_id"], "note"] += "; sampled for plotting"
            else:
                plot_list_df.loc[len(plot_list_df)] = {
                    "plot_id": spec["plot_id"],
                    "quantity": spec["quantity"],
                    "version": spec["version"],
                    "figure_file": str(Path("figures_step26") / spec["figure"]),
                    "axis_scale": spec["axis_scale"],
                    "n_rows": len(data),
                    "n_plotted_rows": len(plot_data),
                    "status": "created",
                    "note": note + ("; sampled for plotting" if sampling else ""),
                }

        combined_path = figures_dir / "figure_07_combined_pred_vs_obs_summary.png"
        if all((scatter_data["plot_id"] == spec["plot_id"]).any() for spec in PLOT_SPECS):
            fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), dpi=180)
            order = [
                "sigma_direct_fitting",
                "PF_direct_fitting",
                "ZT_direct_fitting",
                "sigma_ml_tau_prediction",
                "PF_ml_tau_prediction",
                "ZT_ml_tau_prediction",
            ]
            spec_by_id = {spec["plot_id"]: spec for spec in PLOT_SPECS}
            for ax, plot_id in zip(axes.ravel(), order):
                spec = spec_by_id[plot_id]
                data = scatter_data[scatter_data["plot_id"] == plot_id]
                plot_data, _ = sample_for_plot(data, args.max_points_per_plot)
                draw_scatter(ax, plot_data, spec, metrics_by_plot[plot_id])
            fig.tight_layout()
            fig.savefig(combined_path, bbox_inches="tight")
            plt.close(fig)
            figures_created.append(str(Path("figures_step26") / "figure_07_combined_pred_vs_obs_summary.png"))
            plot_list_df.loc[len(plot_list_df)] = {
                "plot_id": "combined_pred_vs_obs_summary",
                "quantity": "sigma/PF/ZT",
                "version": "direct_fitting_and_ml_tau_prediction",
                "figure_file": str(Path("figures_step26") / "figure_07_combined_pred_vs_obs_summary.png"),
                "axis_scale": "mixed",
                "n_rows": int(sum(len(scatter_data[scatter_data["plot_id"] == spec["plot_id"]]) for spec in PLOT_SPECS)),
                "n_plotted_rows": int(
                    sum(min(len(scatter_data[scatter_data["plot_id"] == spec["plot_id"]]), args.max_points_per_plot) for spec in PLOT_SPECS)
                ),
                "status": "created",
                "note": "2 x 3 summary figure; sigma and PF log-log, ZT linear",
            }
            combined_created = "yes"
        else:
            figures_skipped.append("figure_07_combined_pred_vs_obs_summary.png: one or more main plots had no valid rows")
            plot_list_df.loc[len(plot_list_df)] = {
                "plot_id": "combined_pred_vs_obs_summary",
                "quantity": "sigma/PF/ZT",
                "version": "direct_fitting_and_ml_tau_prediction",
                "figure_file": str(Path("figures_step26") / "figure_07_combined_pred_vs_obs_summary.png"),
                "axis_scale": "mixed",
                "n_rows": 0,
                "n_plotted_rows": 0,
                "status": "skipped",
                "note": "one or more main plots had no valid rows",
            }
            combined_created = "no"
    else:
        combined_created = "no"

    scatter_data.to_csv(output_dir / "pred_obs_scatter_data_step26.csv", index=False)
    metrics_df.to_csv(output_dir / "pred_obs_scatter_metrics_step26.csv", index=False)
    plot_list_df.to_csv(output_dir / "pred_obs_scatter_plot_list_step26.csv", index=False)
    write_notes(output_dir / "step26_pred_obs_scatter_notes.md")

    report_notes = {
        "input_used": input_used,
        "column_notes": column_notes,
        "sampling_notes": sampling_notes,
        "exclusion_notes": exclusion_notes,
    }
    write_report(
        output_dir / "step26_pred_obs_scatter_report.txt",
        args,
        loaded,
        missing,
        selected_cols,
        plot_list_df,
        metrics_df,
        figures_created,
        figures_skipped,
        report_notes,
    )

    report_sheet = pd.DataFrame(
        {"line": (output_dir / "step26_pred_obs_scatter_report.txt").read_text(encoding="utf-8").splitlines()}
    )
    with pd.ExcelWriter(output_dir / "starrydata2_step26_pred_obs_scatter.xlsx", engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="scatter_metrics", index=False)
        plot_list_df.to_excel(writer, sheet_name="plot_list", index=False)
        scatter_data.head(EXCEL_PREVIEW_ROWS).to_excel(writer, sheet_name="scatter_data_sample", index=False)
        report_sheet.to_excel(writer, sheet_name="report", index=False)

    row_lookup = {row["plot_id"]: row for _, row in metrics_df.iterrows()}

    def metric_line(plot_id, metric_name):
        row = row_lookup.get(plot_id)
        return "NA" if row is None else format_metric(row.get(metric_name))

    def rows_line(plot_id):
        row = row_lookup.get(plot_id)
        return "0" if row is None else str(int(row.get("n_rows", 0)))

    print("Done.")
    print("Created:")
    print("- pred_obs_scatter_data_step26.csv")
    print("- pred_obs_scatter_metrics_step26.csv")
    print("- pred_obs_scatter_plot_list_step26.csv")
    print("- step26_pred_obs_scatter_report.txt")
    print("- step26_pred_obs_scatter_notes.md")
    print("- starrydata2_step26_pred_obs_scatter.xlsx")
    print("- figures_step26/*.png")
    print("")
    print("Summary:")
    print(f"figures created: {len(figures_created)}")
    print(f"figures skipped: {len(figures_skipped)}")
    print(f"sigma fitting rows: {rows_line('sigma_direct_fitting')}")
    print(f"sigma ML rows: {rows_line('sigma_ml_tau_prediction')}")
    print(f"PF fitting rows: {rows_line('PF_direct_fitting')}")
    print(f"PF ML rows: {rows_line('PF_ml_tau_prediction')}")
    print(f"ZT fitting rows: {rows_line('ZT_direct_fitting')}")
    print(f"ZT ML rows: {rows_line('ZT_ml_tau_prediction')}")
    print(f"sigma fitting log RMSE: {metric_line('sigma_direct_fitting', 'log_rmse')}")
    print(f"sigma ML log RMSE: {metric_line('sigma_ml_tau_prediction', 'log_rmse')}")
    print(f"ZT fitting MAPE: {metric_line('ZT_direct_fitting', 'mape')}")
    print(f"ZT ML MAPE: {metric_line('ZT_ml_tau_prediction', 'mape')}")
    print(f"combined figure created: {combined_created}")


if __name__ == "__main__":
    main()
