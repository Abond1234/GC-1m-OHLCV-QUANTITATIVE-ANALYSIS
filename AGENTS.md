# Agent instructions

The canonical guidance for working in this repository lives in **[`CLAUDE.md`](CLAUDE.md)**. Read it before making any change.

It is kept as a single file rather than duplicated here so the two cannot drift apart. In short, this is a research codebase whose value is the trustworthiness of its conclusions:

- Research questions, thresholds, and horizons are frozen in a contract **before** anything is computed; never resolve an ambiguity after seeing results.
- The final-test partition (2025 onward) is read once, after verdicts are fixed, and is never explored descriptively.
- Never commit market data or generated artifacts; `.gitignore` is the data-governance boundary.
- Research numerics run on the deterministic CPU path so that a CPU-only laptop and a CUDA workstation agree exactly.
- Verify with `python -m unittest discover -s tests` and `python -m ruff check .` before claiming a change works.
- No AI attribution trailers in commits or PRs, and no emojis anywhere.

The full reasoning behind each of these, plus environment setup, notebook workflow, and collaboration rules, is in `CLAUDE.md`.
