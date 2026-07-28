"""Flexible-exit trade simulation (no Qt dependency).

Reuses the verified per-bar exit ordering from the research engine and adds
path-dependent exits (trailing stop, breakeven move, custom time exit). Under the
frozen contract it reproduces ``strategy_lab.simulate_positions`` to the tick.
"""
