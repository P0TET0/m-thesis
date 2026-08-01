import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = EXP_DIR / "data" / "processed" / "step6b_broad_family" / "step5b_test_predictions_valid.parquet"
DEFAULT_FIGURES = EXP_DIR / "figures" / "step6c_broad_family_large_text_no_title"
DEFAULT_CONFIG_ID = "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median"
DEFAULT_BASENAME = "step6c_scatter_pred_vs_exp_broad_material_family_default_by_representative_material_group"
AXIS_MIN = 1.0e-6
AXIS_MAX = 1.0e10
LEGEND_LABELS = {
    "broad::SiGe_like": "Si-Ge系",
    "broad::SnTe_like": "Sn-Te系",
    "broad::PbTe_like": "Pb-Te系",
    "broad::BiTe_like": "Bi-Te系",
    "broad::SbTe_like": "Sb-Te系",
    "broad::BiSbTe_tetradymite_like": "Bi-Sb-Te系",
    "broad::GeTe_like": "Ge-Te系",
    "broad::Mg2SiSn_like": "Mg-Si/Sn系",
    "broad::CoSb_skutterudite_like": "Co-Sb系・スクッテルダイト系",
    "broad::selenide": "セレン化物系",
    "broad::telluride": "テルル化物系",
    "broad::sulfide": "硫化物系",
    "broad::oxide": "酸化物系",
    "broad::other_formula_system": "その他の化学式系",
    "unknown_material_group": "分類不能",
}
ENGLISH_LEGEND_LABELS = {
    "broad::SiGe_like": "Si-Ge system",
    "broad::SnTe_like": "Sn-Te system",
    "broad::PbTe_like": "Pb-Te system",
    "broad::BiTe_like": "Bi-Te system",
    "broad::SbTe_like": "Sb-Te system",
    "broad::BiSbTe_tetradymite_like": "Bi-Sb-Te system",
    "broad::GeTe_like": "Ge-Te system",
    "broad::Mg2SiSn_like": "Mg-Si/Sn system",
    "broad::CoSb_skutterudite_like": "Co-Sb / skutterudite system",
    "broad::selenide": "Selenide system",
    "broad::telluride": "Telluride system",
    "broad::sulfide": "Sulfide system",
    "broad::oxide": "Oxide system",
    "broad::other_formula_system": "Other formula systems",
    "unknown_material_group": "Unclassified",
}
TELLURIUM_GROUPS = {
    "broad::telluride",
    "broad::BiTe_like",
    "broad::SbTe_like",
    "broad::BiSbTe_tetradymite_like",
    "broad::PbTe_like",
    "broad::SnTe_like",
    "broad::GeTe_like",
}
OXIDE_SULFIDE_GROUPS = {
    "broad::oxide",
    "broad::sulfide",
}
THREE_CLASS_COLORS = {
    "Other systems": "#8c8c8c",
    "Oxide and sulfide systems": "#d95f02",
    "Tellurium systems": "#1f77b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step6C overall scatter colored by representative material group.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--legend-language", choices=["ja", "en"], default="ja")
    parser.add_argument("--legend-font-size", type=int, default=12)
    parser.add_argument("--legend-placement", choices=["outside", "inside"], default="outside")
    parser.add_argument("--grouping", choices=["representative", "three_class"], default="representative")
    parser.add_argument("--hide-legend", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    png = args.figures / f"{args.basename}.png"
    pdf = args.figures / f"{args.basename}.pdf"
    if not args.overwrite and (png.exists() or pdf.exists()):
        raise FileExistsError(f"output exists; choose another basename or pass --overwrite: {png}, {pdf}")

    cols = [
        "config_id",
        "prediction_status",
        "material_group_key_for_prediction",
        "sigma_S_per_m",
        "sigma_pred_S_per_m",
    ]
    df = pd.read_parquet(args.predictions, columns=cols)
    df = df[df["config_id"].eq(args.config_id) & df["prediction_status"].eq("ok")].copy()
    df["sigma_S_per_m"] = pd.to_numeric(df["sigma_S_per_m"], errors="coerce")
    df["sigma_pred_S_per_m"] = pd.to_numeric(df["sigma_pred_S_per_m"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["material_group_key_for_prediction", "sigma_S_per_m", "sigma_pred_S_per_m"]
    )
    df = df[(df["sigma_S_per_m"] > 0) & (df["sigma_pred_S_per_m"] > 0)].copy()
    if df.empty:
        raise ValueError("no positive prediction rows to plot")

    legend_labels = LEGEND_LABELS if args.legend_language == "ja" else ENGLISH_LEGEND_LABELS
    other_label = "その他" if args.legend_language == "ja" else "Other groups"
    top_groups: list[str] = []
    if args.grouping == "three_class":
        group = df["material_group_key_for_prediction"]
        df["plot_group"] = np.select(
            [group.isin(OXIDE_SULFIDE_GROUPS), group.isin(TELLURIUM_GROUPS)],
            ["Oxide and sulfide systems", "Tellurium systems"],
            default="Other systems",
        )
        plot_groups = ["Other systems", "Oxide and sulfide systems", "Tellurium systems"]
    else:
        top_groups = df["material_group_key_for_prediction"].value_counts().head(args.top_n).index.tolist()
        df["plot_group"] = np.where(
            df["material_group_key_for_prediction"].isin(top_groups),
            df["material_group_key_for_prediction"],
            "Other groups",
        )
        plot_groups = ["Other groups", *top_groups]

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    palette = list(plt.get_cmap("tab10").colors)
    for i, group in enumerate(plot_groups):
        part = df[df["plot_group"].eq(group)]
        if part.empty:
            continue
        if args.grouping == "three_class":
            color = THREE_CLASS_COLORS[group]
            label = f"{group} (n={len(part)})"
            alpha = 0.48 if group != "Other systems" else 0.22
        elif group == "Other groups":
            color = "#9aa0a6"
            label = f"{other_label} (n={len(part)})"
            alpha = 0.16
        else:
            color = palette[(i - 1) % len(palette)]
            label = f"{legend_labels.get(group, group)} (n={len(part)})"
            alpha = 0.52
        ax.scatter(
            part["sigma_S_per_m"],
            part["sigma_pred_S_per_m"],
            s=10,
            alpha=alpha,
            linewidths=0,
            color=color,
            label=label,
        )

    ax.plot([AXIS_MIN, AXIS_MAX], [AXIS_MIN, AXIS_MAX], color="black", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_xlabel("Experimental electrical conductivity (S/m)", fontsize=16)
    ax.set_ylabel("Predicted electrical conductivity (S/m)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, which="both", alpha=0.25)
    if not args.hide_legend:
        if args.legend_placement == "inside":
            ax.legend(loc="lower right", fontsize=args.legend_font_size, frameon=True, framealpha=0.88)
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=args.legend_font_size, frameon=True)
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)
    print(f"rows_plotted={len(df)}")
    print(f"representative_groups={len(top_groups)}")


if __name__ == "__main__":
    main()
