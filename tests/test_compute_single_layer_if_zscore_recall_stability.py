from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from research.training.scripts.compute_single_layer_if_zscore_recall_stability import (
    build_recall_stability_summary,
    load_single_layer_combined_rows,
)


def test_load_single_layer_combined_rows_keeps_only_if_zscore_combined_rows(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {
                "detector_combination": "IF",
                "test_case_name": "combined",
                "scope_id": "scope-1",
                "recall": 0.8,
            },
            {
                "detector_combination": "Z-score",
                "test_case_name": "combined",
                "scope_id": "scope-1",
                "recall": 0.6,
            },
            {
                "detector_combination": "IF",
                "test_case_name": "new_prices",
                "scope_id": "scope-1",
                "recall": 0.9,
            },
            {
                "detector_combination": "Sanity -> IF",
                "test_case_name": "combined",
                "scope_id": "scope-1",
                "recall": 0.7,
            },
        ]
    ).to_csv(metrics_csv, index=False)

    loaded = load_single_layer_combined_rows(metrics_csv)

    assert loaded["detector_combination"].tolist() == ["IF", "Z-score"]
    assert set(loaded["test_case_name"]) == {"combined"}


def test_build_recall_stability_summary_matches_layered_table_definition() -> None:
    metrics = pd.DataFrame(
        [
            {"detector_combination": "Z-score", "scope_id": "scope-1", "recall": 0.5},
            {"detector_combination": "Z-score", "scope_id": "scope-1", "recall": 0.7},
            {"detector_combination": "Z-score", "scope_id": "scope-2", "recall": 0.9},
            {"detector_combination": "Z-score", "scope_id": "scope-2", "recall": 1.0},
            {"detector_combination": "IF", "scope_id": "scope-1", "recall": 0.8},
            {"detector_combination": "IF", "scope_id": "scope-1", "recall": 1.0},
            {"detector_combination": "IF", "scope_id": "scope-2", "recall": 0.6},
            {"detector_combination": "IF", "scope_id": "scope-2", "recall": 0.8},
        ]
    )

    summary = build_recall_stability_summary(metrics)
    by_detector = summary.set_index("detector_combination")

    assert summary["detector_combination"].astype(str).tolist() == ["Z-score", "IF"]
    assert by_detector.loc["Z-score", "valid_scope_mh_evaluations"] == 4
    assert by_detector.loc["Z-score", "unique_scopes"] == 2
    assert by_detector.loc["Z-score", "mean_recall"] == 0.775
    assert by_detector.loc["Z-score", "min_recall"] == 0.5
    assert by_detector.loc["Z-score", "max_recall"] == 1.0
    assert by_detector.loc["IF", "mean_recall"] == 0.8
    assert round(float(by_detector.loc["IF", "scope_mean_recall_std"]), 6) == 0.141421
