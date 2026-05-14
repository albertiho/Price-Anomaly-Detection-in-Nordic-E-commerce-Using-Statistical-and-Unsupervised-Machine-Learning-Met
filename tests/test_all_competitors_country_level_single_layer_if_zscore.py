from __future__ import annotations

import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from research.training.scripts import run_all_competitors_country_level_layered_finalists
from research.training.scripts.run_all_competitors_country_level_single_layer_if_zscore import (
    IF_ONLY_NAME,
    RUN_ID,
    SINGLE_LAYER_COMBINATIONS,
    ZSCORE_ONLY_NAME,
    configure_runner,
)


def test_single_layer_combinations_are_only_if_and_zscore() -> None:
    assert SINGLE_LAYER_COMBINATIONS == [
        IF_ONLY_NAME,
        ZSCORE_ONLY_NAME,
    ]


def test_single_layer_run_id_is_separate_from_layered_finalists() -> None:
    assert RUN_ID != run_all_competitors_country_level_layered_finalists.RUN_ID
    assert RUN_ID == "all_competitors_all_countries_country_level_single_layer_if_zscore"


def test_configure_runner_sets_single_layer_variant(monkeypatch) -> None:
    monkeypatch.setattr(run_all_competitors_country_level_layered_finalists, "RUN_ID", "sentinel")
    monkeypatch.setattr(
        run_all_competitors_country_level_layered_finalists,
        "FINAL_COMBINATIONS",
        ["sentinel"],
    )

    configure_runner()

    assert run_all_competitors_country_level_layered_finalists.RUN_ID == RUN_ID
    assert run_all_competitors_country_level_layered_finalists.FINAL_COMBINATIONS == [
        IF_ONLY_NAME,
        ZSCORE_ONLY_NAME,
    ]
