"""Pool the 14 step29 oxide systems and compare their curves with SiGe."""

import json

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from plot_starrydata2_sige_vs_listed_materials_step29 import (
    LISTED_MATERIAL_SYSTEMS,
    PROJECT_ROOT,
    SIGE_SYSTEM,
    plot_pair,
)


def main() -> None:
    output_dir = PROJECT_ROOT / "data/output/starrydata2_sige_vs_combined_oxides"
    output_dir.mkdir(parents=True, exist_ok=True)
    oxide_systems = [
        system for system in LISTED_MATERIAL_SYSTEMS
        if "O" in system.split("-") and system != "Bi-Cu-O-Se"
    ]
    assert len(oxide_systems) == 14
    source = PROJECT_ROOT / "data/output/starrydata2_step29_sige_vs_listed_materials/step27_selected_curve_map.csv"
    curves = pd.read_csv(source, keep_default_na=False)
    # Each step29 material comparison repeats the same SiGe curves.
    curves = curves.drop_duplicates(subset="curve_id").copy()
    for column in ("temperature_K", "sigma_S_per_m"):
        curves[column] = curves[column].map(json.loads)
    selected = curves[curves["element_system"].isin([SIGE_SYSTEM, *oxide_systems])].copy()
    selected["family_label"] = selected["element_system"].map(
        lambda system: "SiGe" if system == SIGE_SYSTEM else "Oxides"
    )
    counts = selected["element_system"].value_counts()
    assert all(counts.get(system, 0) > 0 for system in [SIGE_SYSTEM, *oxide_systems])
    assert not selected["curve_id"].duplicated().any(), "Duplicate source curve IDs"
    for show_median in (True, False):
        plot_pair(
            selected, "Oxides", output_dir,
            log_y=True, dpi=220, y_quantile_low=0.01,
            y_quantile_high=0.995, show_median=show_median,
        )
    counts.rename_axis("element_system").rename("n_curves").to_csv(
        output_dir / "curve_counts_by_system.csv"
    )
    export = selected.copy()
    for column in ("temperature_K", "sigma_S_per_m"):
        export[column] = export[column].map(json.dumps)
    export.to_csv(output_dir / "selected_curve_map.csv", index=False)
    report = {
        "source": str(source),
        "oxide_systems": oxide_systems,
        "excluded_oxyselenide": "Bi-Cu-O-Se",
        "sige_curves": int(selected["family_label"].eq("SiGe").sum()),
        "oxide_curves": int(selected["family_label"].eq("Oxides").sum()),
        "aggregation": "Pool individual curves, not sum conductivity values. Each curve has equal weight.",
        "trend": "Median at 180 equally spaced temperatures; linear interpolation within each curve range, no extrapolation; at least 2 curves required.",
        "y_axis": "Log scale; displayed limits use pooled 1st and 99.5th percentiles, matching step29.",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
