from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.statistical_research.mlat_artifacts import (
    load_saved_table,
    prepare_csv_table,
    save_csv,
    verify_saved_table,
)


class MlatArtifactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temporary_directory.name)
        self.addCleanup(self._temporary_directory.cleanup)

    def test_csv_round_trip_distinguishes_empty_strings_nulls_and_mixed_objects(
        self,
    ) -> None:
        source = pd.DataFrame(
            {
                "name": pd.Series(["", "NA", "plain"], dtype="string"),
                "value": [1.5, np.nan, 3.25],
                "mixed": [
                    ["a", 1],
                    7,
                    {"z": 2, "a": 1},
                ],
            }
        )
        expected = prepare_csv_table(source)
        path = self.tmp_path / "round_trip.csv"

        entry = save_csv(expected, path)
        checks = verify_saved_table(expected, path, manifest_entry=entry)
        reloaded = load_saved_table(path, expected=expected)

        assert checks["passed"].all()
        pd.testing.assert_frame_equal(expected, reloaded, check_exact=True)
        assert reloaded.loc[0, "name"] == ""
        assert reloaded.loc[1, "name"] == "NA"
        assert pd.isna(reloaded.loc[1, "value"])
        assert reloaded.loc[0, "mixed"] == json.dumps(["a", 1])
        assert reloaded.loc[1, "mixed"] == "7"
        assert reloaded.loc[2, "mixed"] == json.dumps({"a": 1, "z": 2}, sort_keys=True)

    def test_save_csv_rejects_reserved_null_token(self) -> None:
        table = pd.DataFrame({"value": ["__MLAT_CSV_NULL_V1__"]})

        with self.assertRaisesRegex(ValueError, "reserved null token"):
            save_csv(table, self.tmp_path / "collision.csv")


if __name__ == "__main__":
    unittest.main()
