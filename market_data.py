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


# --------------------------------------------------------------------------- fuentes extra para la investigación
def urls_in(*texts: str | None, limit: int = 3) -> list[str]:
    """URLs que aparecen en los criterios de resolución (la fuente que decide la pregunta)."""
    import re
    seen: list[str] = []
    for t in texts:
        for u in re.findall(r"https?://[^\s)\]>\"'<]+", t or ""):
            u = u.rstrip(".,;:")
            if u not in seen and "metaculus.com" not in u:
                seen.append(u)
    return seen[:limit]


def page_text(url: str, limit: int = 6000) -> str:
    """Texto plano de una página (sin JavaScript). Vacío si falla."""
    import html as _html
    import re
    try:
        r = requests.get(url, headers=UA, timeout=25)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        logger.info(f"No se pudo leer {url}: {e}")
        return ""
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype and "json" not in ctype:
        return ""
    t = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", r.text)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = _html.unescape(re.sub(r"\s+", " ", t)).strip()
    return t[:limit]


def prediction_markets(query: str, n: int = 4) -> list[dict]:
    """Mercados abiertos en Manifold y Polymarket que encajan con la búsqueda (solo lectura, APIs públicas)."""
    import json as _json
    out: list[dict] = []
    try:
        r = requests.get("https://api.manifold.markets/v0/search-markets",
                         params={"term": query, "limit": n, "filter": "open", "sort": "liquidity"}, headers=UA, timeout=20)
        r.raise_for_status()
        for m in r.json()[:n]:
            if m.get("probability") is None:
                continue
            out.append({"site": "Manifold", "q": m.get("question"), "p": float(m["probability"]),
                        "volume": m.get("volume"), "url": m.get("url"), "close": m.get("closeTime")})
    except Exception as e:  # noqa: BLE001
        logger.info(f"Manifold falló: {e}")
    try:
        r = requests.get("https://gamma-api.polymarket.com/public-search",
                         params={"q": query, "limit_per_type": n, "events_status": "active"}, headers=UA, timeout=20)
        r.raise_for_status()
        for ev in (r.json().get("events") or [])[:n]:
            for m in (ev.get("markets") or [])[:3]:
                try:
                    prices = _json.loads(m.get("outcomePrices") or "[]")
                    outs = _json.loads(m.get("outcomes") or "[]")
                except Exception:  # noqa: BLE001
                    continue
                if not prices or m.get("closed"):
                    continue
                out.append({"site": "Polymarket", "q": m.get("question") or ev.get("title"),
                            "p": float(prices[0]), "outcome": outs[0] if outs else "Yes",
                            "volume": m.get("volume"), "url": f"https://polymarket.com/event/{ev.get('slug', '')}"})
    except Exception as e:  # noqa: BLE001
        logger.info(f"Polymarket falló: {e}")
    return out


def markets_text(markets: list[dict]) -> str:
    if not markets:
        return ""
    lines = ["Prediction-market prices found (check carefully that each market really matches the question's exact criteria and dates; "
             "liquid markets are strong evidence, thin markets are weak):"]
    for m in markets:
        vol = m.get("volume")
        vol_s = f", volume ~{float(vol):,.0f}" if isinstance(vol, (int, float, str)) and str(vol).replace('.', '', 1).isdigit() else ""
        lines.append(f"- {m['site']}: \"{m.get('q')}\" → {m.get('outcome', 'YES')} {m['p']:.0%}{vol_s} ({m.get('url')})")
    return "\n".join(lines)
