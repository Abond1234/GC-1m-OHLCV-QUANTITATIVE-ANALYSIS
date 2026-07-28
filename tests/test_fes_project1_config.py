from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - unittest discovery without pytest
    import unittest

    raise unittest.SkipTest("pytest is not installed; run this module with pytest") from None

from src.statistical_research.fes_project1_config import (
    DETERMINISTIC_SEED,
    O1_FEATURES,
    SPECIFICATION_PATH,
    SPECIFICATION_SHA256,
    assert_frozen_specification,
    build_frozen_config,
    build_trial_ledger_skeleton,
    config_sha256,
    trial_ledger_sha256,
    write_contract_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_specification_digest_is_frozen() -> None:
    if not (PROJECT_ROOT / SPECIFICATION_PATH).exists():
        # The frozen specification markdown has not been committed to the
        # repository yet (it exists only on the research owner's machine), so
        # no checkout except theirs can hash it. Skip rather than fail every
        # other contributor's suite; committing the document re-arms this test.
        pytest.skip("frozen specification markdown is not present in this checkout")
    assert assert_frozen_specification(PROJECT_ROOT) == SPECIFICATION_SHA256


def test_config_is_deterministic_and_stops_at_section_1() -> None:
    first = build_frozen_config()
    second = build_frozen_config()

    assert first == second
    assert first["randomness"]["primary_seed"] == DETERMINISTIC_SEED
    assert first["notebook"]["section_boundary"] == 1
    assert first["notebook"]["section_2_started"] is False
    assert len(first["features"]) == 10
    assert len(first["interactions"]) == 5
    assert config_sha256(first) == (
        "efb7ee9a79a81b96023fd20ae747a99129809ba4307a314d482b0370b648401f"
    )


def test_trial_ledger_predeclares_all_frozen_families() -> None:
    ledger = build_trial_ledger_skeleton()
    trial_ids = [row["trial_id"] for row in ledger]
    family_counts: dict[str, int] = {}
    for row in ledger:
        family = row["multiplicity_family"]
        family_counts[family] = family_counts.get(family, 0) + 1

    assert len(ledger) == 150
    assert len(trial_ids) == len(set(trial_ids))
    assert family_counts["CONFIRMATORY_20"] == 20
    assert family_counts["EXPLORATORY_40"] == 40
    assert family_counts["INTERACTION_8"] == 8
    assert all(row["status"] == "PLANNED" for row in ledger)
    assert all(row["frozen_before_outcomes"] is True for row in ledger)
    assert trial_ledger_sha256(ledger) == (
        "60eadc61477430b3fb6e4ba7719db983270c4d8ea5a49c747d1827d1009cbb5a"
    )


def test_frozen_o1_content_has_exact_15_unique_features() -> None:
    assert len(O1_FEATURES) == 15
    assert len(set(O1_FEATURES)) == 15
    assert O1_FEATURES[0] == "atr_20"
    assert O1_FEATURES[-1] == "vwap_elasticity_30_exp"


def test_contract_writer_is_idempotent_and_refuses_conflicts(tmp_path: Path) -> None:
    first = write_contract_artifacts(tmp_path)
    second = write_contract_artifacts(tmp_path)
    assert first == second

    config_path = tmp_path / first["config_path"]
    config_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_contract_artifacts(tmp_path)
