"""Compare existing ZT errors before/after removing temperature_K <= 300 K."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.compute_poster_unified_log10_metrics import compute_log10_ratio_metrics

SOURCE = ROOT / "data/output/df_ml_fml_material_parity_20261025"
OUTPUT = HERE / "temperature_gt_300K"
METHODS = ("DF", "ML", "FML")


def metrics(rows):
    observed, predicted = rows.zt_obs, rows.zt_pred
    error = predicted - observed
    sst = float(((observed - observed.mean()) ** 2).sum())
    log_metrics = compute_log10_ratio_metrics(observed, predicted)
    return {
        "paired_points": len(rows), "samples": rows.sample_key.nunique(),
        "temperature_min_K": rows.temperature_K.min(),
        "temperature_max_K": rows.temperature_K.max(),
        "zt_obs_mean": observed.mean(),
        "rmse": np.sqrt((error**2).mean()), "mae": error.abs().mean(),
        "r2_y_equals_x": 1.0 - float((error**2).sum()) / sst if sst else np.nan,
        **{k: v for k, v in log_metrics.items() if k != "exclusion_reasons"},
    }


def main():
    records, sources = [], []
    saved_summary = pd.read_csv(OUTPUT / "filter_summary.csv")
    for method in METHODS:
        path = SOURCE / f"plot_source_{method}.csv"
        original = pd.read_csv(path)
        selected = pd.read_csv(OUTPUT / path.name)
        pd.testing.assert_frame_equal(
            selected, original[original.temperature_K.gt(300)].reset_index(drop=True),
            check_exact=False, rtol=1e-12, atol=1e-12)
        pairs = original[np.isfinite(original.zt_obs) & np.isfinite(original.zt_pred)]
        groups = {"before": pairs, "after": pairs[pairs.temperature_K.gt(300)],
                  "excluded": pairs[pairs.temperature_K.le(300)]}
        computed = {stage: metrics(rows) for stage, rows in groups.items()}
        before, after, removed = (computed[k] for k in ("before", "after", "excluded"))
        assert before["paired_points"] == after["paired_points"] + removed["paired_points"]
        # Reconstruct full-data errors from the two disjoint temperature subsets.
        for field, power, count in (("rmse", 2, "paired_points"), ("mae", 1, "paired_points"),
                                    ("rmse_log10_pred_over_exp", 2, "n_valid"),
                                    ("mae_log10_pred_over_exp", 1, "n_valid")):
            combined = ((after[count] * after[field]**power + removed[count] * removed[field]**power)
                        / before[count]) ** (1.0 / power)
            np.testing.assert_allclose(before[field], combined, rtol=1e-12)
        saved = saved_summary[(saved_summary.method == method) & saved_summary.quantity.eq("zt")
                              & saved_summary.material_group.eq("all")].iloc[0]
        np.testing.assert_allclose(
            [after["paired_points"], after["rmse"], after["mae"], after["rmse_log10_pred_over_exp"]],
            [saved.paired_points_after, saved.rmse_all_paired_points,
             saved.mae_all_paired_points, saved.rmse_log10_positive_pairs], rtol=1e-12)
        records.extend({"method": method, "stage": stage, **values} for stage, values in computed.items())
        sources.append({"method": method, "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT / "zt_metrics_before_after.csv", index=False, encoding="utf-8-sig")
    changes = []
    for method in METHODS:
        indexed = table[table.method.eq(method)].set_index("stage")
        row = {"method": method}
        for metric in ("rmse", "mae", "rmse_log10_pred_over_exp", "mae_log10_pred_over_exp",
                       "r2_y_equals_x", "factor2_accuracy", "factor10_accuracy"):
            before, after = (float(indexed.loc[s, metric]) for s in ("before", "after"))
            row.update({f"{metric}_before": before, f"{metric}_after": after,
                        f"{metric}_delta": after - before})
            if metric in ("rmse", "mae", "rmse_log10_pred_over_exp", "mae_log10_pred_over_exp"):
                row[f"{metric}_relative_change_percent"] = 100 * (after / before - 1)
            if metric.startswith("factor"):
                row[f"{metric}_change_percentage_points"] = 100 * (after - before)
        changes.append(row)
    pd.DataFrame(changes).to_csv(OUTPUT / "zt_metrics_changes.csv", index=False, encoding="utf-8-sig")
    manifest = {"sources": sources, "retained_condition": "temperature_K > 300",
                "evaluation_unit": "temperature point; no sample reweighting",
                "linear_metrics": "All finite ZT observed/predicted pairs, including points outside fixed figure axes",
                "log_metrics": "Finite, strictly positive observed/predicted pairs only; log10(predicted/observed)",
                "factor2_accuracy": "Fraction with 0.5 <= predicted/observed <= 2 among positive pairs",
                "r2_y_equals_x": "1 - sum((predicted-observed)^2) / sum((observed-mean(observed))^2)",
                "validation": "Exact retained subset, saved after-filter metrics, counts and weighted reconstruction of before-filter errors passed",
                "interpretation": "Evaluation subsets differ before/after and across methods; predictions were not retrained"}
    (OUTPUT / "zt_metrics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table[["method", "stage", "paired_points", "rmse", "mae", "rmse_log10_pred_over_exp",
                 "mae_log10_pred_over_exp", "factor2_accuracy"]].to_string(index=False))
    print("PASS: filtered subsets, previous metrics and temperature-subset decomposition")


if __name__ == "__main__":
    main()
