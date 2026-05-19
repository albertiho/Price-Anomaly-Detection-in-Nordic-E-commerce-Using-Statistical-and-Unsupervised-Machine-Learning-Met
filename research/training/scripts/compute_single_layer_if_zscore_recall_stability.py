#!/usr/bin/env python3
"""Compute recall-stability values for standalone IF and Z-score results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


RUN_ID = "all_competitors_all_countries_country_level_single_layer_if_zscore"
THESIS_METRICS_DIR = (
    _PROJECT_ROOT
    / "results"
    / "detector_combinations"
    / RUN_ID
    / "analysis"
    / "thesis_metrics"
)
DEFAULT_METRICS_CSV = THESIS_METRICS_DIR / "layered_detector_metrics.csv"
DEFAULT_OUTPUT_CSV = THESIS_METRICS_DIR / "single_layer_if_zscore_recall_stability_summary.csv"

DETECTOR_ORDER = ["Z-score", "IF"]
REQUIRED_COLUMNS = {
    "detector_combination",
    "test_case_name",
    "scope_id",
    "recall",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute recall-stability statistics for the standalone IF and Z-score "
            "evaluation on the combined test case."
        )
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help="Path to the single-layer layered_detector_metrics.csv file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path where the recall-stability summary CSV should be written.",
    )
    return parser.parse_args()


def load_single_layer_combined_rows(metrics_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(metrics_csv)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{metrics_csv} is missing required columns: {missing}")

    combined = frame[
        (frame["test_case_name"] == "combined")
        & (frame["detector_combination"].isin(DETECTOR_ORDER))
    ].copy()
    if combined.empty:
        raise ValueError(f"No combined IF/Z-score rows found in {metrics_csv}")

    found_detectors = set(combined["detector_combination"].unique())
    missing_detectors = set(DETECTOR_ORDER).difference(found_detectors)
    if missing_detectors:
        missing = ", ".join(sorted(missing_detectors))
        raise ValueError(f"Missing detector rows in {metrics_csv}: {missing}")

    return combined


def build_recall_stability_summary(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    overall = (
        metrics_frame.groupby("detector_combination", as_index=False)
        .agg(
            valid_scope_mh_evaluations=("scope_id", "count"),
            unique_scopes=("scope_id", "nunique"),
            mean_recall=("recall", "mean"),
            std_recall=("recall", "std"),
            min_recall=("recall", "min"),
            max_recall=("recall", "max"),
        )
        .reset_index(drop=True)
    )
    scope_means = (
        metrics_frame.groupby(["detector_combination", "scope_id"], as_index=False)
        .agg(scope_mean_recall=("recall", "mean"))
        .reset_index(drop=True)
    )
    scope_stability = (
        scope_means.groupby("detector_combination", as_index=False)
        .agg(scope_mean_recall_std=("scope_mean_recall", "std"))
        .reset_index(drop=True)
    )
    summary = overall.merge(scope_stability, on="detector_combination", how="left")
    summary["detector_combination"] = pd.Categorical(
        summary["detector_combination"],
        categories=DETECTOR_ORDER,
        ordered=True,
    )
    return summary.sort_values("detector_combination").reset_index(drop=True)


def print_latex_rows(summary: pd.DataFrame) -> None:
    for row in summary.itertuples(index=False):
        print(
            f"{row.detector_combination} & "
            f"{row.mean_recall:.4f} & "
            f"{row.std_recall:.3f} & "
            f"{row.min_recall:.4f} & "
            f"{row.max_recall:.2f} & "
            f"{row.scope_mean_recall_std:.3f} \\\\"
        )


def main() -> None:
    args = parse_args()
    combined = load_single_layer_combined_rows(args.metrics_csv)
    summary = build_recall_stability_summary(combined)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    print(f"Wrote {args.output_csv}")
    print_latex_rows(summary)


if __name__ == "__main__":
    main()
