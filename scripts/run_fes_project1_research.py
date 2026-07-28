"""Non-interactive section runner for the frozen Project 1 FES study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.statistical_research.fes_project1_evaluation import (
    LockedValidationAuthorization,
    assert_section5_freeze,
    run_locked_section6_batch,
    save_section6_outputs,
)
from src.statistical_research.fes_project1_nested import (
    build_section5_development,
    save_section5_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one explicitly authorized frozen FES notebook section."
    )
    parser.add_argument("--section", type=int, required=True, choices=(5, 6))
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--section5-manifest-sha256")
    parser.add_argument("--policy-sha256")
    return parser


def _assert_section5_hashes(
    manifest_sha256: str | None,
    policy_sha256: str | None,
) -> LockedValidationAuthorization:
    if not manifest_sha256 or not policy_sha256:
        raise ValueError(
            "Section 6 requires both --section5-manifest-sha256 and "
            "--policy-sha256."
        )
    return assert_section5_freeze(
        PROJECT_ROOT,
        expected_manifest_sha256=manifest_sha256,
        expected_policy_sha256=policy_sha256,
    )


def main() -> int:
    args = _parser().parse_args()
    if not args.non_interactive:
        raise ValueError("The frozen runner requires --non-interactive.")
    if args.section == 5:
        result = build_section5_development(
            PROJECT_ROOT,
            use_cache=not args.no_cache,
            progress=True,
        )
        outputs = save_section5_outputs(result, PROJECT_ROOT)
        print(json.dumps(outputs, indent=2, sort_keys=True, default=str))
        return 0
    authorization = _assert_section5_hashes(
        args.section5_manifest_sha256,
        args.policy_sha256,
    )
    result = run_locked_section6_batch(PROJECT_ROOT, authorization)
    outputs = save_section6_outputs(result, PROJECT_ROOT)
    print(json.dumps(outputs, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
