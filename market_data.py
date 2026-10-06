"""
market_data.py - Línea base cuantitativa para preguntas de mercados (Market Pulse).

Dada una serie (Yahoo Finance o FRED) y una fecha objetivo, calcula una distribución de
probabilidad del valor en esa fecha a partir de la volatilidad histórica, con colas gruesas
(t de Student, 5 grados de libertad). Es la "respuesta de manual" de un analista cuantitativo
y los modelos de IA la usan como punto de partida; además entra en la mezcla final como un
pronosticador más.
"""
from __future__ import annotations

import csv
import io
import logging
import math
from datetime import date, datetime, timezone
from statistics import NormalDist

import requests

logger = logging.getLogger("market_data")
UA = {"User-Agent": "Mozilla/5.0 (forecast-bot)"}
PCTS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]


def yahoo_history(symbol: str) -> list[tuple[date, float]]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=2y&interval=1d"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    out = [(datetime.fromtimestamp(t, tz=timezone.utc).date(), float(c)) for t, c in zip(ts, closes) if c is not None]
    return out


def fred_history(series: str) -> list[tuple[date, float]]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    rows = list(csv.reader(io.StringIO(r.text)))
    out = []
    for row in rows[1:]:
        try:
            out.append((date.fromisoformat(row[0]), float(row[1])))
        except (ValueError, IndexError):
            continue
    return out


def _t_quantile(q: float, df: int = 5) -> float:
    try:
        from scipy.stats import t  # type: ignore

        return float(t.ppf(q, df)) * math.sqrt((df - 2) / df)  # escalado a varianza 1
    except Exception:
        return NormalDist().inv_cdf(q)


def business_days(a: date, b: date) -> int:
    if b <= a:
        return 0
    days = (b - a).days
    weeks, rem = divmod(days, 7)
    n = weeks * 5
    wd = a.weekday()
    for i in range(1, rem + 1):
        if (wd + i) % 7 < 5:
            n += 1
    return n


def baseline(source: str, symbol: str, target: date, today: date | None = None,
             additive: bool | None = None) -> dict | None:
    """Devuelve {'current', 'as_of', 'vol_daily', 'h', 'percentiles': [(p, valor)], 'text'} o None."""
    today = today or datetime.now(timezone.utc).date()
    hist = yahoo_history(symbol) if source == "yahoo" else fred_history(symbol)
    hist = [(d, v) for d, v in hist if d <= today]
    if len(hist) < 60:
        return None
    if additive is None:
        additive = source == "fred" or min(v for _, v in hist[-250:]) <= 0
    vals = [v for _, v in hist]
    if additive:
        rets = [b - a for a, b in zip(vals[:-1], vals[1:])]
    else:
        rets = [math.log(b / a) for a, b in zip(vals[:-1], vals[1:]) if a > 0 and b > 0]

    def sd(x: list[float]) -> float:
        m = sum(x) / len(x)
        return math.sqrt(sum((y - m) ** 2 for y in x) / max(len(x) - 1, 1))

    s60, s250 = sd(rets[-60:]), sd(rets[-250:])
    vol = math.sqrt(0.5 * s60 ** 2 + 0.5 * s250 ** 2)
    h = max(business_days(hist[-1][0], target), 1)
    cur = vals[-1]
    out = []
    for p in PCTS:
        z = _t_quantile(p) * vol * math.sqrt(h)
        out.append((p, cur + z if additive else cur * math.exp(z)))
    txt = (f"Quantitative baseline from {source}:{symbol}: last value {cur:.4g} on {hist[-1][0]}; "
           f"daily volatility {vol:.4g} ({'absolute' if additive else 'log'}); {h} trading days to {target}. "
           f"Fat-tailed random-walk percentiles: "
           + ", ".join(f"P{int(p * 100)}={v:.4g}" for p, v in out if p in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95) or p in (0.2, 0.4, 0.6, 0.8)))
    return {"current": cur, "as_of": str(hist[-1][0]), "vol_daily": vol, "h": h, "percentiles": out,
            "additive": additive, "text": txt}
