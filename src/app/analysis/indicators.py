"""Chart study engine: OHLCV indicators computed on the displayed bars.

Pure numpy, no Qt. Every function returns arrays of the input length with NaN
over the warmup span, so lines simply start once defined. These are display
aids for the simulator's chart - they are computed at the displayed timeframe
(like any charting platform) and play no role in research artifacts, entries,
or fills, which stay on true 1-minute bars in the verified engine.

The registry drives the Indicators workspace: search list, default parameters
and their bounds, which pane an output renders in (price overlay, oscillator
pane, or the volume pane), and whether a study is sourced from precomputed
data (the VWAP variants) instead of computed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_float(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sma(values, n: int) -> np.ndarray:
    """Simple moving average; NaN until ``n`` values exist."""

    x = _as_float(values)
    out = np.full(len(x), np.nan)
    if n <= 0 or len(x) < n:
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1 :] = (csum[n:] - csum[:-n]) / n
    return out


def ema(values, n: int) -> np.ndarray:
    """Exponential moving average, seeded with the first ``n``-bar SMA."""

    x = _as_float(values)
    out = np.full(len(x), np.nan)
    if n <= 0 or len(x) < n:
        return out
    alpha = 2.0 / (n + 1.0)
    out[n - 1] = float(np.mean(x[:n]))
    for i in range(n, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    """Rolling population standard deviation (matches the app's z-score maths)."""

    mean = sma(x, n)
    mean_sq = sma(x * x, n)
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    return np.sqrt(var)


def bollinger(values, n: int, k: float) -> dict[str, np.ndarray]:
    """Rolling mean with ``k`` population standard deviations either side."""

    x = _as_float(values)
    mid = sma(x, n)
    dev = _rolling_std(x, n) * float(k)
    return {"mid": mid, "upper": mid + dev, "lower": mid - dev}


def true_range(high, low, close) -> np.ndarray:
    """True range with the first bar falling back to high - low."""

    h, l, c = _as_float(high), _as_float(low), _as_float(close)  # noqa: E741
    tr = h - l
    if len(c) > 1:
        prev = c[:-1]
        tr = np.concatenate(
            [
                tr[:1],
                np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - prev), np.abs(l[1:] - prev)]),
            ]
        )
    return tr


def atr(high, low, close, n: int) -> np.ndarray:
    """Average true range as a simple ``n``-bar mean of true range.

    A simple mean (not Wilder smoothing) deliberately matches the repository's
    ``rolling_atr_20m`` convention, so the study reads consistently with the
    exit engine's ATR-sized stops.
    """

    return sma(true_range(high, low, close), n)


def rsi(values, n: int) -> np.ndarray:
    """Wilder's relative strength index (0-100)."""

    x = _as_float(values)
    out = np.full(len(x), np.nan)
    if n <= 0 or len(x) < n + 1:
        return out
    change = np.diff(x)
    gain = np.clip(change, 0.0, None)
    loss = np.clip(-change, 0.0, None)
    avg_gain = float(np.mean(gain[:n]))
    avg_loss = float(np.mean(loss[:n]))
    for i in range(n, len(x)):
        if i > n:
            avg_gain = (avg_gain * (n - 1) + gain[i - 1]) / n
            avg_loss = (avg_loss * (n - 1) + loss[i - 1]) / n
        if avg_loss == 0.0:
            out[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def rolling_volatility(values, n: int) -> np.ndarray:
    """Rolling population std of one-bar simple returns, in percent."""

    x = _as_float(values)
    out = np.full(len(x), np.nan)
    if len(x) < 2:
        return out
    prev = x[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev != 0.0, (x[1:] - prev) / prev, np.nan) * 100.0
    out[1:] = _rolling_std(rets, n)
    return out


def obv(close, volume) -> np.ndarray:
    """On-balance volume: cumulative volume signed by the bar-to-bar move."""

    c, v = _as_float(close), _as_float(volume)
    if len(c) == 0:
        return np.array([])
    signs = np.concatenate([[0.0], np.sign(np.diff(c))])
    return np.cumsum(signs * v)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    default: float
    lo: float
    hi: float
    integer: bool = True
    step: float = 1.0


@dataclass(frozen=True)
class IndicatorDef:
    """One study the workspace can activate.

    ``pane`` places the outputs: "price" overlays the candles, "osc" renders in
    the oscillator pane, "volume" in the volume pane. ``source`` names a
    precomputed series in the data bundle (the VWAP variants) instead of a
    computation here. ``guides`` are fixed horizontal reference levels for the
    oscillator pane (e.g. RSI 30/70).
    """

    key: str
    label: str
    pane: str
    params: tuple[ParamSpec, ...] = ()
    outputs: tuple[str, ...] = ("value",)
    source: str | None = None
    band: tuple[str, str] | None = None  # outputs shaded as a translucent band
    guides: tuple[float, ...] = ()


def _n(default: int, lo: int = 2, hi: int = 400) -> tuple[ParamSpec, ...]:
    return (ParamSpec("n", "length", default, lo, hi),)


INDICATORS: dict[str, IndicatorDef] = {
    d.key: d
    for d in (
        IndicatorDef("sma", "SMA", "price", _n(20)),
        IndicatorDef("ema", "EMA", "price", _n(20)),
        IndicatorDef(
            "bollinger",
            "Bollinger Bands",
            "price",
            (*_n(20), ParamSpec("k", "std devs", 2.0, 0.5, 5.0, integer=False, step=0.5)),
            outputs=("mid", "upper", "lower"),
            band=("upper", "lower"),
        ),
        IndicatorDef("atr", "ATR", "osc", _n(14)),
        IndicatorDef("rsi", "RSI", "osc", _n(14), guides=(30.0, 70.0)),
        IndicatorDef("rvol", "Rolling volatility", "osc", _n(30)),
        IndicatorDef("obv", "OBV", "osc"),
        IndicatorDef("vma", "Volume MA", "volume", _n(20)),
        IndicatorDef("vwap20", "VWAP 20", "price", source="vwap20"),
        IndicatorDef("vwap_day", "VWAP day", "price", source="vwap_day"),
        IndicatorDef("vwap_session", "VWAP session", "price", source="vwap_session"),
    )
}


@dataclass
class IndicatorInstance:
    """An activated study: definition key, its parameter values, and style."""

    id: int
    key: str
    params: dict[str, float] = field(default_factory=dict)
    color: str = "#c9a227"
    visible: bool = True

    @property
    def definition(self) -> IndicatorDef:
        return INDICATORS[self.key]

    def display_label(self) -> str:
        args = " ".join(
            f"{self.params[p.name]:g}" for p in self.definition.params if p.name in self.params
        )
        return f"{self.definition.label} {args}".strip()


def compute(key: str, params: dict[str, float], ohlc: dict) -> dict[str, np.ndarray]:
    """Evaluate a computed study on the displayed window's OHLCV arrays."""

    n = int(params.get("n", 0))
    close = ohlc["close"]
    if key == "sma":
        return {"value": sma(close, n)}
    if key == "ema":
        return {"value": ema(close, n)}
    if key == "bollinger":
        return bollinger(close, n, float(params.get("k", 2.0)))
    if key == "atr":
        return {"value": atr(ohlc["high"], ohlc["low"], close, n)}
    if key == "rsi":
        return {"value": rsi(close, n)}
    if key == "rvol":
        return {"value": rolling_volatility(close, n)}
    if key == "obv":
        return {"value": obv(close, ohlc["volume"])}
    if key == "vma":
        return {"value": sma(ohlc["volume"], n)}
    raise KeyError(f"unknown computed indicator: {key}")
