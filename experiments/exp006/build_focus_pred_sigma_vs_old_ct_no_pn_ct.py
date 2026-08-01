import argparse
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_focus_pred_sigma_vs_old_ct as base


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed" / "focus_pred_sigma_vs_old_ct_no_pn_ct"
DEFAULT_FIGURES = EXP_DIR / "figures" / "focus_pred_sigma_vs_old_ct_no_pn_ct"
DEFAULT_REPORT = EXP_DIR / "reports" / "focus_pred_sigma_vs_old_ct_no_pn_ct" / "focus_pred_sigma_vs_old_ct_no_pn_ct_report.md"


def log(message: str) -> None:
    print(f"[pred_vs_ct_no_pn] {message}", flush=True)


base.log = log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare broad-family sigma_pred points with no-p/n old SS2026 C(T) curves.")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--old-ct-script", type=Path, required=True)
    parser.add_argument("--config-id", default=base.DEFAULT_CONFIG_ID)
    parser.add_argument("--target-groups", nargs="+", default=base.DEFAULT_TARGET_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-rows-per-group", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def join_examples(values: pd.Series) -> str:
    labels = [str(v) for v in values.dropna().astype(str).unique() if str(v).strip()]
    labels = sorted(labels)[:5]
    return "; ".join(labels)


def load_old_ct_no_pn(selected: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("loading old C(T) curves...")
    path = selected["path"]
    header = base.header_columns(path)
    keep = list(
        dict.fromkeys(
            [
                c
                for c in [
                    selected["ct_col"],
                    selected["temp_col"],
                    selected["material_col"],
                    "material_system",
                    "composition",
                    "prefactor_group_key_step12",
                ]
                if c and c in header
            ]
        )
    )
    raw = pd.read_csv(path, usecols=keep, low_memory=False)
    raw["source_file"] = str(path)
    raw["old_material_label"] = base.old_material_label(raw, selected["material_col"])
    mapped = raw["old_material_label"].map(base.map_old_material_label)
    raw["material_group_key_mapped"] = mapped.map(lambda x: x[0])
    raw["mapping_status"] = mapped.map(lambda x: x[1])
    raw["mapping_rule"] = mapped.map(lambda x: x[2])
    raw["T_K"] = pd.to_numeric(raw[selected["temp_col"]], errors="coerce")
    raw["old_C_T_S_per_m"] = pd.to_numeric(raw[selected["ct_col"]], errors="coerce")
    raw["old_ct_parse_status"] = np.where(
        raw["material_group_key_mapped"].eq("unmatched"),
        "unmatched_material",
        np.where(np.isfinite(raw["T_K"]) & base.finite_positive(raw["old_C_T_S_per_m"]), "ok", "invalid_numeric"),
    )
    mapping = (
        raw[["old_material_label", "material_group_key_mapped", "mapping_status", "mapping_rule"]]
        .drop_duplicates()
        .sort_values(["material_group_key_mapped", "old_material_label"])
    )
    unmatched = mapping[mapping["mapping_status"].eq("unmatched")].copy()
    ok = raw[raw["old_ct_parse_status"].eq("ok")].copy()
    log("aggregating old C(T) without p/n split...")
    normalized = (
        ok.groupby(["source_file", "material_group_key_mapped", "T_K"], dropna=False, sort=True)
        .agg(
            old_material_label_examples=("old_material_label", join_examples),
            old_C_T_S_per_m=("old_C_T_S_per_m", "median"),
            n_rows_aggregated=("old_C_T_S_per_m", "size"),
        )
        .reset_index()
    )
    normalized["log10_old_C_T_S_per_m"] = np.log10(normalized["old_C_T_S_per_m"])
    normalized["old_ct_parse_status"] = "ok"
    normalized = normalized[
        [
            "source_file",
            "old_material_label_examples",
            "material_group_key_mapped",
            "T_K",
            "old_C_T_S_per_m",
            "log10_old_C_T_S_per_m",
            "n_rows_aggregated",
            "old_ct_parse_status",
        ]
    ]
    line = normalized.copy()
    return normalized, line, mapping, unmatched


def nearest_comparison(pred: pd.DataFrame, old_line: pd.DataFrame, target_groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in target_groups:
        pred_sub = pred[pred["material_group_key"].eq(group)]
        old_sub = old_line[old_line["material_group_key_mapped"].eq(group)].sort_values("T_K")
        if pred_sub.empty or old_sub.empty:
            continue
        old_t = old_sub["T_K"].to_numpy(dtype=float)
        for _, row in pred_sub.iterrows():
            t_pred = float(row["T_K"])
            pos = int(np.nanargmin(np.abs(old_t - t_pred)))
            old = old_sub.iloc[pos]
            pred_sigma = float(row["sigma_pred_S_per_m"])
            old_ct = float(old["old_C_T_S_per_m"])
            rows.append(
                {
                    "material_group_key": group,
                    "row_id": row.get("row_id", ""),
                    "carrier_type": row.get("carrier_type", ""),
                    "T_K_pred": t_pred,
                    "sigma_pred_S_per_m": pred_sigma,
                    "log10_sigma_pred_S_per_m": math.log10(pred_sigma),
                    "T_K_old_ct": float(old["T_K"]),
                    "old_C_T_S_per_m": old_ct,
                    "log10_old_C_T_S_per_m": math.log10(old_ct),
                    "T_delta_K": float(old["T_K"] - t_pred),
                    "log10_pred_over_oldCT": math.log10(pred_sigma / old_ct),
                    "match_method": "nearest_no_pn_old_CT_temperature",
                }
            )
    return pd.DataFrame(rows)


def plot_main(group: str, pred: pd.DataFrame, old: pd.DataFrame, png: Path, pdf: Path) -> None:
    log("creating sigma_pred vs old C(T) figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "p": {"color": "#1f77b4", "marker": "o", "label": "Predicted sigma, p"},
        "n": {"color": "#2ca02c", "marker": "^", "label": "Predicted sigma, n"},
    }
    for carrier in ["p", "n"]:
        sub = pred[pred["carrier_type"].eq(carrier)]
        if sub.empty:
            continue
        style = styles[carrier]
        ax.scatter(
            sub["T_K"],
            sub["sigma_pred_S_per_m"],
            s=13,
            alpha=0.42,
            color=style["color"],
            marker=style["marker"],
            edgecolors="none",
            label=style["label"],
        )
    ax.plot(old["T_K"], old["old_C_T_S_per_m"], linewidth=2.6, color="#d62728", label="Old C(T) from SS2026, no p/n split")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("Electrical conductivity sigma [S/m]")
    ax.set_title(f"{group}: predicted sigma vs old C(T), no p/n split")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_summary(pred: pd.DataFrame, old_line: pd.DataFrame, comparison: pd.DataFrame, target_groups: list[str]) -> pd.DataFrame:
    rows = []
    for group in target_groups:
        p = pred[pred["material_group_key"].eq(group)]
        o = old_line[old_line["material_group_key_mapped"].eq(group)]
        c = comparison[comparison["material_group_key"].eq(group)]
        warnings = []
        if p.empty:
            warnings.append("no_prediction_points")
        if o.empty:
            warnings.append("no_old_ct")
        if c.empty:
            warnings.append("no_nearest_comparison")
        rows.append(
            {
                "material_group_key": group,
                "prediction_points": len(p),
                "p_prediction_points": int(p["carrier_type"].eq("p").sum()) if not p.empty else 0,
                "n_prediction_points": int(p["carrier_type"].eq("n").sum()) if not p.empty else 0,
                "old_ct_points": len(o),
                "T_pred_min_K": p["T_K"].min() if not p.empty else np.nan,
                "T_pred_max_K": p["T_K"].max() if not p.empty else np.nan,
                "T_old_ct_min_K": o["T_K"].min() if not o.empty else np.nan,
                "T_old_ct_max_K": o["T_K"].max() if not o.empty else np.nan,
                "sigma_pred_median_S_per_m": p["sigma_pred_S_per_m"].median() if not p.empty else np.nan,
                "old_C_T_median_S_per_m": o["old_C_T_S_per_m"].median() if not o.empty else np.nan,
                "median_log10_pred_over_oldCT_nearest": c["log10_pred_over_oldCT"].median() if not c.empty else np.nan,
                "warning": ";".join(warnings),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    pred_path: Path,
    old_script: Path,
    selected: dict[str, Any],
    target_groups: list[str],
    summary: pd.DataFrame,
    figure_index: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    ratio = comparison["log10_pred_over_oldCT"].dropna() if "log10_pred_over_oldCT" in comparison.columns else pd.Series(dtype=float)
    lines = [
        "# Predicted sigma vs no-p/n old C(T)",
        "",
        "This is a separate focus analysis outside the existing Step0-Step7C pipeline.",
        "",
        "## Inputs",
        f"- Prediction file: `{pred_path}`",
        f"- Old C(T) source script: `{old_script}`",
        f"- Detected old C(T) CSV: `{selected['path']}`",
        f"- Old C(T) column: `{selected['ct_col']}`",
        f"- Temperature column: `{selected['temp_col']}`",
        f"- Material column: `{selected['material_col']}`",
        f"- Ignored p/n column from old C(T), if present: `{selected.get('carrier_col', '')}`",
        "",
        "## Old C(T) Aggregation",
        "- Old C(T) is aggregated without p/n splitting.",
        "- Aggregation method: median over material group x temperature.",
        "- The n_or_p/carrier column is not used for the old C(T) curve.",
        "- Each material group has at most one old C(T) line.",
        "",
        "## Target Material Groups",
    ]
    lines.extend([f"- {group}" for group in target_groups])
    lines.extend(["", "## Group Summary"])
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['material_group_key']}: predictions={row['prediction_points']} "
            f"(p={row['p_prediction_points']}, n={row['n_prediction_points']}), "
            f"old_ct={row['old_ct_points']}, median_log10_pred_over_oldCT={row['median_log10_pred_over_oldCT_nearest']}, "
            f"warning={row['warning'] if isinstance(row['warning'], str) and row['warning'] else 'none'}"
        )
    lines.extend(["", "## Missing Data"])
    missing_old = summary[summary["old_ct_points"].eq(0)]
    missing_pred = summary[summary["prediction_points"].eq(0)]
    lines.append("- Material groups without old C(T): " + (", ".join(missing_old["material_group_key"].astype(str)) if not missing_old.empty else "none"))
    lines.append("- Material groups without prediction points: " + (", ".join(missing_pred["material_group_key"].astype(str)) if not missing_pred.empty else "none"))
    lines.extend(["", "## Ratio Overview"])
    if ratio.empty:
        lines.append("- No nearest comparison points were available.")
    else:
        lines.extend(
            [
                f"- count: {len(ratio)}",
                f"- min log10(sigma_pred / old C(T)): {ratio.min():.6g}",
                f"- median log10(sigma_pred / old C(T)): {ratio.median():.6g}",
                f"- max log10(sigma_pred / old C(T)): {ratio.max():.6g}",
            ]
        )
    lines.extend(["", "## Figures"])
    if figure_index.empty:
        lines.append("- No figures were created.")
    else:
        for _, row in figure_index.iterrows():
            lines.append(f"- {row['material_group_key']} / {row['figure_type']}: `{row['figure_path_png']}`")
    lines.extend(
        [
            "",
            "## How To Read The Figures",
            "- Points are current predicted sigma values from the broad_family prediction result.",
            "- The line is the SS2026 old C(T) curve aggregated without p/n splitting.",
            "- If the points lie near the old C(T) line, the prediction has a similar temperature-dependent scale to the old C(T) baseline.",
            "- If the points are far from the line, the S-input prediction differs from the old C(T) baseline.",
            "- Because old C(T) is not split by p/n, compare where p-type and n-type point clouds sit relative to the same line.",
            "",
            "## Notes",
            "- Points are current predicted sigma.",
            "- The line is the no-p/n SS2026 old C(T).",
            "- Experimental sigma points are not included in the main figures.",
            "- sigma0_ref is not included in the figures.",
            "- No new sigma_pred is calculated.",
            "- Step4 full-data reference curves are not used.",
            "- Starrydata2 raw data are not read.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity_checks(
    pred: pd.DataFrame,
    old_line: pd.DataFrame,
    figure_index: pd.DataFrame,
    summary: pd.DataFrame,
    report: Path,
    pred_path: Path,
    selected: dict[str, Any],
    target_groups: list[str],
) -> None:
    log("running sanity checks...")
    failures = []
    if not pred_path.exists():
        failures.append(f"prediction file missing: {pred_path}")
    if pred.empty:
        failures.append("prediction rows are empty after filtering")
    if not pred["config_id"].astype(str).eq(pred["config_id"].astype(str).iloc[0]).all():
        failures.append("multiple config_id values remained")
    if not pred["prediction_status"].astype(str).eq("ok").all():
        failures.append("non-ok prediction_status remained")
    if not base.finite_positive(pred["sigma_pred_S_per_m"]).all():
        failures.append("sigma_pred_S_per_m contains non-positive/non-finite values")
    if not Path(selected["path"]).exists():
        failures.append(f"old C(T) file missing: {selected['path']}")
    if old_line.empty or not base.finite_positive(old_line["old_C_T_S_per_m"]).all():
        failures.append("old_C_T_S_per_m contains non-positive/non-finite values")
    if "carrier_type" in old_line.columns:
        failures.append("old C(T) line unexpectedly contains carrier_type")
    duplicate_lines = old_line.duplicated(["material_group_key_mapped", "T_K"]).sum()
    if duplicate_lines:
        failures.append(f"old C(T) has duplicated material/temperature rows: {duplicate_lines}")
    if not target_groups:
        failures.append("target groups are empty")
    if figure_index.empty:
        failures.append("no figures were created")
    if summary.empty:
        failures.append("summary is empty")
    if not report.exists():
        failures.append(f"report missing: {report}")
    for forbidden in ["sigma0_ref", "step4_full", "starrydata2/raw"]:
        if forbidden in str(pred_path).casefold() or forbidden in str(selected["path"]).casefold():
            failures.append(f"forbidden input path detected: {forbidden}")
    if failures:
        raise RuntimeError("Sanity checks failed:\n" + "\n".join(failures))


def output_path(directory: Path, stem: str, suffix: str) -> Path:
    return directory / f"{stem}{suffix}.csv"


def main() -> None:
    start = time.time()
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    pred_path = args.predictions or base.resolve_first(base.DEFAULT_PREDICTIONS, "prediction")
    target_groups = list(args.target_groups)
    if args.max_groups is not None:
        target_groups = target_groups[: args.max_groups]

    pred = base.load_predictions(pred_path, args.config_id)
    selected = base.detect_old_ct(args.old_ct_script)
    old_normalized, old_line, mapping, unmatched = load_old_ct_no_pn(selected)

    log("mapping old material labels...")
    log("filtering target groups...")
    pred = pred[pred["material_group_key"].isin(target_groups)].copy()
    old_normalized = old_normalized[old_normalized["material_group_key_mapped"].isin(target_groups)].copy()
    old_line = old_line[old_line["material_group_key_mapped"].isin(target_groups)].copy()

    if args.max_rows_per_group is not None:
        limited_parts = []
        for _, group_df in pred.groupby(["material_group_key", "carrier_type"], sort=False):
            limited_parts.append(base.limited_for_plot(group_df.sort_values("T_K"), args.max_rows_per_group))
        pred = pd.concat(limited_parts, ignore_index=True) if limited_parts else pred.iloc[0:0].copy()

    comparison = nearest_comparison(pred, old_line, target_groups)
    summary = build_summary(pred, old_line, comparison, target_groups)

    figure_rows: list[dict[str, Any]] = []
    figure_id = 1
    for group in target_groups:
        log(f"processing material group {group}")
        p = pred[pred["material_group_key"].eq(group)].sort_values("T_K")
        o = old_line[old_line["material_group_key_mapped"].eq(group)].sort_values("T_K")
        if p.empty or o.empty:
            log(f"warning: missing prediction or old C(T) for {group}; skipping figure")
            continue
        safe = base.safe_name(group)
        png = args.figures / f"{safe}_sigma_pred_points_vs_oldCT_no_pn_line{args.output_suffix}.png"
        pdf = args.figures / f"{safe}_sigma_pred_points_vs_oldCT_no_pn_line{args.output_suffix}.pdf"
        plot_main(group, p, o, png, pdf)
        figure_rows.append(
            {
                "figure_id": figure_id,
                "material_group_key": group,
                "figure_type": "sigma_pred_vs_old_ct_no_pn",
                "figure_path_png": str(png),
                "figure_path_pdf": str(pdf),
                "title": f"{group}: predicted sigma vs old C(T), no p/n split",
                "n_prediction_points": len(p),
                "n_old_ct_points": len(o),
                "description": "Predicted sigma points by carrier type and one SS2026 old C(T) line aggregated without p/n split.",
            }
        )
        figure_id += 1
    figure_index = pd.DataFrame(figure_rows)

    log("writing CSV outputs...")
    pred.to_csv(output_path(args.output, "focus_pred_sigma_vs_old_ct_no_pn_prediction_rows", args.output_suffix), index=False)
    old_normalized.to_csv(output_path(args.output, "focus_pred_sigma_vs_old_ct_no_pn_old_ct_curves_normalized", args.output_suffix), index=False)
    summary_path = output_path(args.output, "focus_pred_sigma_vs_old_ct_no_pn_summary_by_group", args.output_suffix)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(output_path(args.output, "focus_pred_sigma_vs_old_ct_no_pn_nearest_comparison_table", args.output_suffix), index=False)
    figure_index_path = output_path(args.output, "focus_pred_sigma_vs_old_ct_no_pn_figure_index", args.output_suffix)
    figure_index.to_csv(figure_index_path, index=False)
    mapping.to_csv(output_path(args.output, "focus_pred_sigma_vs_old_ct_no_pn_material_mapping", args.output_suffix), index=False)
    unmatched.to_csv(output_path(args.output, "focus_pred_sigma_vs_old_ct_no_pn_unmatched_old_material_labels", args.output_suffix), index=False)

    log("writing report...")
    write_report(args.report, pred_path, args.old_ct_script, selected, target_groups, summary, figure_index, comparison)
    run_sanity_checks(pred, old_line, figure_index, summary, args.report, pred_path, selected, target_groups)

    elapsed = time.time() - start
    ratio = comparison["log10_pred_over_oldCT"].dropna() if "log10_pred_over_oldCT" in comparison.columns else pd.Series(dtype=float)
    log(f"done. elapsed_seconds={elapsed:.2f}")
    print(f"prediction_file: {pred_path}")
    print(f"old_ct_script: {args.old_ct_script}")
    print(f"old_ct_csv: {selected['path']}")
    print(f"old_ct_column: {selected['ct_col']}")
    print("old_ct_aggregation: no p/n split; median by material_group_key_mapped x T_K")
    print(f"target_groups: {', '.join(target_groups)}")
    print(f"material_group_rows: {len(summary)}")
    print(f"figures: {len(figure_index)}")
    print(f"median_log10_pred_over_oldCT: {ratio.median() if not ratio.empty else 'NA'}")
    print(f"output_dir: {args.output}")
    print(f"figure_dir: {args.figures}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
