"""Save/load the working session as versioned JSON.

Persists what the user built - placed trades (by entry position and exit
config), drawings (global coordinates), view state (date anchor, range,
timeframe), account and evaluation settings - but never simulation OUTPUTS:
on load every trade is re-simulated through the verified engine
(``analysis.placed_trade.place_trade``), so a session file can shape what is
recomputed but can never inject results.

Files default to the git-ignored ``reports/sessions/`` directory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

from ..sim.exit_config import ExitConfig

# v2: the view block carries a literal start_date/end_date window (was an
# anchor_date + rolling range_days count).
SCHEMA_VERSION = 2


def default_sessions_dir(project_root: Path) -> Path:
    return Path(project_root) / "reports" / "sessions"


def exit_config_to_dict(cfg: ExitConfig) -> dict:
    return asdict(cfg)


def exit_config_from_dict(payload: dict) -> ExitConfig:
    allowed = {f.name for f in fields(ExitConfig)}
    return ExitConfig(**{k: v for k, v in payload.items() if k in allowed})


def build_payload(
    *,
    theme_mode: str,
    view: dict,
    account: dict,
    evaluation_preset: str,
    placed: list[dict],
    active_id: int | None,
    next_id: int,
    drawings: list[dict],
) -> dict:
    """Assemble the schema; ``placed`` entries carry cfg dicts, never results."""

    return {
        "schema_version": SCHEMA_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "theme": theme_mode,
        "view": dict(view),
        "account": dict(account),
        "evaluation_preset": evaluation_preset,
        "placed": list(placed),
        "active_id": active_id,
        "next_id": int(next_id),
        "drawings": list(drawings),
    }


def write_session(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_session(path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported session schema_version {version!r}; this build reads {SCHEMA_VERSION}"
        )
    for key in ("view", "account", "placed", "drawings"):
        if key not in payload:
            raise ValueError(f"session file missing required key {key!r}")
    return payload
