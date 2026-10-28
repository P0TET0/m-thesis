#!/usr/bin/env python3
"""Validate exp007 Phase 0 outputs and recheck exp006 source integrity."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP007 = ROOT / "experiments" / "exp007"
DEFAULT_OUTPUT = EXP007 / "data" / "processed"
DEFAULT_REPORT = EXP007 / "reports" / "phase0_baseline_comparison.md"
DEFAULT_CHECK_LOG = EXP007 / "reports" / "phase0_check_log.txt"

EXPECTED_VARIANTS = {
    "legacy_old_default",
    "broad_family_100K",
    "broad_family_25K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check-log", type=Path, default=DEFAULT_CHECK_LOG)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    files = {
        "metrics": args.output_dir / "phase0_baseline_metrics.csv",
        "status": args.output_dir / "phase0_prediction_status_summary.csv",
        "validation": args.output_dir / "phase0_reference_validation.csv",
        "common_metrics": args.output_dir / "phase0_common_row_metrics.csv",
        "common_predictions": args.output_dir / "phase0_common_row_predictions.csv",
        "deltas": args.output_dir / "phase0_metric_deltas_vs_100K.csv",
        "decision": args.output_dir / "phase0_baseline_decision.csv",
        "manifest": args.output_dir / "phase0_input_manifest.csv",
    }
    missing = [str(path) for path in [*files.values(), args.report] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 0 outputs: {missing}")

    metrics = pd.read_csv(files["metrics"])
    status = pd.read_csv(files["status"])
    validation = pd.read_csv(files["validation"])
    common_metrics = pd.read_csv(files["common_metrics"])
    common_predictions = pd.read_csv(files["common_predictions"], low_memory=False)
    decision = pd.read_csv(files["decision"])
    manifest = pd.read_csv(files["manifest"])

    checks: dict[str, bool] = {
        "three_variants_present": set(metrics["variant"]) == EXPECTED_VARIANTS,
        "two_weightings_per_variant": len(metrics) == 6
        and set(metrics["metric_weighting"]) == {"row_equal", "sample_equal"},
        "status_has_three_variants": set(status["variant"]) == EXPECTED_VARIANTS,
        "all_reference_checks_pass": bool(validation["passed"].all()),
        "reference_tolerance_met": bool(
            (pd.to_numeric(validation["abs_difference"], errors="coerce") <= 1e-12).all()
        ),
        "common_metrics_complete": len(common_metrics) == 6
        and set(common_metrics["variant"]) == EXPECTED_VARIANTS,
        "common_cohort_nonempty": len(common_predictions) > 0,
        "one_primary_baseline": int(decision["selected_primary"].sum()) == 1,
        "primary_is_100K": decision.loc[
            decision["selected_primary"], "variant"
        ].tolist()
        == ["broad_family_100K"],
        "roles_are_unique": decision["role"].nunique() == 3,
        "manifest_has_six_sources": len(manifest) == 6,
        "manifest_run_integrity_passed": bool(
            manifest["source_unchanged_during_run"].all()
        ),
        "report_records_primary": "Primary baseline: `broad_family_100K`"
        in args.report.read_text(encoding="utf-8"),
    }

    current_hash_matches: list[bool] = []
    for row in manifest.itertuples(index=False):
        source = ROOT / str(row.source_file)
        current_hash_matches.append(
            source.is_file() and sha256(source) == str(row.sha256)
        )
    checks["exp006_sources_still_match_manifest"] = all(current_hash_matches)

    failed = [name for name, passed in checks.items() if not passed]
    lines = [
        "exp007 Phase 0 independent check",
        "=" * 37,
        *[f"{'PASS' if passed else 'FAIL'} {name}" for name, passed in checks.items()],
        "",
        f"common_prediction_rows={len(common_predictions)}",
        f"maximum_reference_abs_difference={validation['abs_difference'].max():.3g}",
        f"Final status: {'PASS' if not failed else 'FAIL'}",
        "",
    ]
    args.check_log.parent.mkdir(parents=True, exist_ok=True)
    args.check_log.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if failed:
        raise RuntimeError(f"Phase 0 checks failed: {failed}")


if __name__ == "__main__":
    main()
