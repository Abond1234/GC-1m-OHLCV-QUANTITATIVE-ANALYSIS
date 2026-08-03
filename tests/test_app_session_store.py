"""Versioned session persistence (Qt-free)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.app.datalayer.session_store import (
    SCHEMA_VERSION,
    build_payload,
    exit_config_from_dict,
    exit_config_to_dict,
    read_session,
    write_session,
)
from src.app.sim.exit_config import ExitConfig


def _payload(**overrides):
    base = dict(
        theme_mode="dark",
        view={"start_date": "2024-12-24", "end_date": "2024-12-31", "timeframe": "15m"},
        account={"starting_balance": 100000.0, "sizing_mode": "risk_percent"},
        evaluation={"name": "Prop 100k", "starting_balance": 100000.0, "profit_target_pct": 8.0},
        placed=[
            {
                "id": 1,
                "entry_position": 12345,
                "direction": 1,
                "color": "#e0b74e",
                "cfg": exit_config_to_dict(ExitConfig(stop_value=2.5)),
            }
        ],
        active_id=1,
        next_id=2,
        drawings=[{"kind": "hline", "y": 2650.0, "gspan": [12000, 13000]}],
    )
    base.update(overrides)
    return build_payload(**base)


class SessionStoreTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            write_session(path, _payload())
            loaded = read_session(path)
            self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
            self.assertEqual(loaded["view"]["timeframe"], "15m")
            self.assertEqual(loaded["placed"][0]["entry_position"], 12345)
            cfg = exit_config_from_dict(loaded["placed"][0]["cfg"])
            self.assertEqual(cfg.stop_value, 2.5)

    def test_results_are_never_serialized(self):
        payload = _payload()
        text = str(payload)
        for banned in ("gross_r", "exit_price", "stop_track", "result"):
            self.assertNotIn(banned, text)

    def test_unknown_version_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            bad = _payload()
            bad["schema_version"] = 99
            write_session(path, bad)
            with self.assertRaises(ValueError):
                read_session(path)

    def test_missing_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            bad = _payload()
            del bad["placed"]
            write_session(path, bad)
            with self.assertRaises(ValueError):
                read_session(path)

    def test_exit_config_dict_ignores_unknown_keys(self):
        payload = exit_config_to_dict(ExitConfig())
        payload["future_field"] = 123
        cfg = exit_config_from_dict(payload)
        self.assertEqual(cfg.stop_mode, ExitConfig().stop_mode)


if __name__ == "__main__":
    unittest.main()
