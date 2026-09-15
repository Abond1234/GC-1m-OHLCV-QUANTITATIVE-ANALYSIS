"""Run the frozen POI-first feature-combination study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.statistical_research.feature_combination_runner import (  # noqa: E402
    run_feature_combination_research,
)


def main() -> None:
    result = run_feature_combination_research(ROOT)
    print(json.dumps(result.headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
