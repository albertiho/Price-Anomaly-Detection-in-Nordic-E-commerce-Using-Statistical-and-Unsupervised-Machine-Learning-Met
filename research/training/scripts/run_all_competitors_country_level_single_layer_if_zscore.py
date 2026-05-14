#!/usr/bin/env python3
"""Evaluate standalone IF and Z-score on the final competitor-scope surface.

This is a thin variant of ``run_all_competitors_country_level_layered_finalists``.
It keeps the same countries, sampled minimum-history settings, competitor-level
splits, synthetic injection protocol, and retained country-level Isolation
Forest configurations, but evaluates only the two single-layer detector
baselines:

- ``IF``
- ``Z-score``

The separate ``RUN_ID`` prevents these supplemental baseline results from
mixing with the thesis-final layered finalist run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from research.training.scripts import run_all_competitors_country_level_layered_finalists
from research.training.scripts.analyze_if_zscore_layered_combinations import (
    IF_ONLY_NAME,
    ZSCORE_ONLY_NAME,
)

RUN_ID = "all_competitors_all_countries_country_level_single_layer_if_zscore"
SINGLE_LAYER_COMBINATIONS = [
    IF_ONLY_NAME,
    ZSCORE_ONLY_NAME,
]


def configure_runner() -> None:
    """Configure the shared final-surface runner for single-layer baselines."""
    run_all_competitors_country_level_layered_finalists.RUN_ID = RUN_ID
    run_all_competitors_country_level_layered_finalists.FINAL_COMBINATIONS = list(
        SINGLE_LAYER_COMBINATIONS
    )


def run() -> Path:
    """Run the single-layer IF/Z-score evaluation variant."""
    configure_runner()
    return run_all_competitors_country_level_layered_finalists.run()


def main() -> None:
    """Execute the single-layer IF/Z-score evaluation variant."""
    run()


if __name__ == "__main__":
    main()
