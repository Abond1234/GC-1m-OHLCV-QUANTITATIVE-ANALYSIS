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
# v3: the evaluation is a full prop-firm policy dict (was a preset name).
# v4: adds the active indicator-study list (older files load with none active).
SCHEMA_VERSION = 4
_READABLE_VERSIONS = (3, SCHEMA_VERSION)


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
    evaluation: dict,
    placed: list[dict],
    active_id: int | None,
    next_id: int,
    drawings: list[dict],
    indicators: list[dict] | None = None,
) -> dict:
    """Assemble the schema; ``placed`` entries carry cfg dicts, never results."""

    return {
        "schema_version": SCHEMA_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "theme": theme_mode,
        "view": dict(view),
        "account": dict(account),
        "evaluation": dict(evaluation),
        "placed": list(placed),
        "active_id": active_id,
        "next_id": int(next_id),
        "drawings": list(drawings),
        "indicators": list(indicators or []),
    }


def write_session(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_session(path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version not in _READABLE_VERSIONS:
        raise ValueError(
            f"unsupported session schema_version {version!r}; this build reads "
            f"{' and '.join(str(v) for v in _READABLE_VERSIONS)}"
        )
    for key in ("view", "account", "placed", "drawings"):
        if key not in payload:
            raise ValueError(f"session file missing required key {key!r}")
    return payload
