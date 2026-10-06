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
import os
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
        logger.info(f"No se pudo leer {url}: {e}; pruebo con Jina Reader")
        try:
            return jina_read(url, limit)
        except Exception:  # noqa: BLE001
            return ""
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype and "json" not in ctype:
        return ""
    t = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", r.text)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = _html.unescape(re.sub(r"\s+", " ", t)).strip()
    if len(t) < 800:   # probablemente una página hecha con JavaScript: probar con Jina Reader
        try:
            jt = jina_read(url, limit)
            if len(jt) > len(t):
                return jt[:limit]
        except Exception as e:  # noqa: BLE001
            logger.info(f"Jina no pudo leer {url}: {e}")
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


# --------------------------------------------------------------------------- buscadores gratuitos
# Sin clave: Google News (RSS), GDELT (noticias de todo el mundo), Wikipedia, Jina Reader (lee páginas con JavaScript).
# Con clave gratuita (si está en los Secrets de GitHub): Tavily (TAVILY_API_KEY), Brave Search (BRAVE_API_KEY), Exa (EXA_API_KEY).
STOP = {"will", "the", "a", "an", "of", "in", "on", "by", "before", "after", "be", "to", "for", "what", "how", "many", "much",
        "which", "who", "is", "are", "and", "or", "than", "more", "less", "at", "least", "with", "end", "between", "does",
        "do", "did", "any", "there", "this", "that", "its", "it", "as", "from", "into", "during", "until", "next", "most"}


def keywords(text: str, n: int = 6) -> str:
    import re
    words = re.findall(r"[A-Za-z0-9][\w'.$%-]*", text or "")
    return " ".join([w for w in words if w.lower() not in STOP][:n])


def _clean(s: str, limit: int = 400) -> str:
    import html as _html
    import re
    return _html.unescape(re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", s or ""))).strip()[:limit]


def google_news(query: str, n: int = 8) -> list[str]:
    import xml.etree.ElementTree as ET
    r = requests.get("https://news.google.com/rss/search", params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                     headers=UA, timeout=25)
    r.raise_for_status()
    out = []
    for it in ET.fromstring(r.content).iter("item"):
        t, d = it.findtext("title") or "", it.findtext("pubDate") or ""
        src = it.find("source")
        out.append(f"- {d[:16]} · {_clean(t, 200)}" + (f" ({src.text})" if src is not None and src.text else ""))
        if len(out) >= n:
            break
    return out


def gdelt(query: str, n: int = 8) -> list[str]:
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc",
                     params={"query": query, "mode": "ArtList", "format": "json", "maxrecords": n, "sort": "DateDesc",
                             "timespan": "3w"}, headers=UA, timeout=25)
    r.raise_for_status()
    try:
        arts = r.json().get("articles", [])
    except ValueError:
        return []
    return [f"- {a.get('seendate', '')[:8]} · {_clean(a.get('title'), 200)} ({a.get('domain', '')})" for a in arts[:n]]


def wikipedia(query: str, n: int = 2) -> list[str]:
    r = requests.get("https://en.wikipedia.org/w/api.php",
                     params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": n},
                     headers=UA, timeout=25)
    r.raise_for_status()
    out = []
    for h in r.json().get("query", {}).get("search", [])[:n]:
        title = h.get("title", "")
        try:
            s = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_"), headers=UA, timeout=20)
            ext = s.json().get("extract", "") if s.ok else ""
        except Exception:  # noqa: BLE001
            ext = ""
        out.append(f"- {title}: {_clean(ext or h.get('snippet'), 700)}")
    return out


def tavily(query: str, n: int = 6) -> list[str]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    r = requests.post("https://api.tavily.com/search", headers={"Authorization": f"Bearer {key}", **UA},
                      json={"api_key": key, "query": query, "search_depth": "advanced", "include_answer": True,
                            "max_results": n, "topic": "general"}, timeout=40)
    r.raise_for_status()
    j = r.json()
    out = [f"Answer: {_clean(j.get('answer'), 600)}"] if j.get("answer") else []
    out += [f"- {_clean(x.get('title'), 150)} ({x.get('url')}): {_clean(x.get('content'), 450)}" for x in j.get("results", [])[:n]]
    return out


def brave(query: str, n: int = 6) -> list[str]:
    key = os.getenv("BRAVE_API_KEY")
    if not key:
        return []
    r = requests.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": n, "freshness": "pm"},
                     headers={"X-Subscription-Token": key, "Accept": "application/json"}, timeout=30)
    r.raise_for_status()
    res = (r.json().get("web") or {}).get("results", [])
    return [f"- {_clean(x.get('title'), 150)} ({x.get('url')}) {x.get('age', '')}: {_clean(x.get('description'), 350)}" for x in res[:n]]


def exa(query: str, n: int = 5) -> list[str]:
    key = os.getenv("EXA_API_KEY")
    if not key:
        return []
    r = requests.post("https://api.exa.ai/search", headers={"x-api-key": key, "Content-Type": "application/json"},
                      json={"query": query, "numResults": n, "type": "auto", "contents": {"highlights": {"numSentences": 3}}},
                      timeout=40)
    r.raise_for_status()
    out = []
    for x in r.json().get("results", [])[:n]:
        hl = " … ".join(x.get("highlights") or [])
        out.append(f"- {_clean(x.get('title'), 150)} ({x.get('url')}) {str(x.get('publishedDate') or '')[:10]}: {_clean(hl, 450)}")
    return out


def jina_read(url: str, limit: int = 6000) -> str:
    """Lee una página aunque use JavaScript (servicio gratuito r.jina.ai)."""
    r = requests.get("https://r.jina.ai/" + url, headers={**UA, "Accept": "text/plain"}, timeout=40)
    r.raise_for_status()
    return r.text[:limit]


def free_search(question_text: str, extra_query: str | None = None) -> str:
    """Junta todo lo que encuentran los buscadores gratuitos. Cada fuente que falle se salta sin más."""
    q = extra_query or keywords(question_text)
    if not q:
        return ""
    blocks = []
    for name, fn in (("Tavily", tavily), ("Brave Search", brave), ("Exa", exa), ("Google News", google_news),
                     ("GDELT (global news)", gdelt), ("Wikipedia", wikipedia)):
        try:
            rows = fn(q)
        except Exception as e:  # noqa: BLE001
            logger.info(f"{name} falló: {e}")
            continue
        if rows:
            blocks.append(f"[{name}]\n" + "\n".join(rows))
    if not blocks:
        return ""
    return f"Free web search results for: {q}\n(Check dates; headlines are evidence, not proof.)\n\n" + "\n\n".join(blocks)


def related_metaculus(question_text: str, own_post_id: int | None = None, n: int = 6) -> str:
    """Preguntas abiertas parecidas en Metaculus con la predicción de su comunidad (información pública)."""
    token = os.getenv("METACULUS_TOKEN")
    q = keywords(question_text, 5)
    if not token or not q:
        return ""
    r = requests.get("https://www.metaculus.com/api/posts/", params={"search": q, "statuses": "open", "limit": 15},
                     headers={"Authorization": f"Token {token}", "Accept-Language": "en"}, timeout=30)
    r.raise_for_status()
    rows = []
    for post in (r.json().get("results") or []):
        if own_post_id and post.get("id") == own_post_id:
            continue
        qq = post.get("question") or {}
        if qq.get("type") != "binary":
            continue
        agg = (qq.get("aggregations") or {})
        latest = ((agg.get("recency_weighted") or {}).get("latest") or {})
        c = latest.get("centers")
        if not c:
            continue
        rows.append(f"- \"{_clean(post.get('title'), 160)}\" → community {float(c[0]):.0%} "
                    f"({post.get('nr_forecasters') or qq.get('nr_forecasters') or '?'} forecasters) "
                    f"https://www.metaculus.com/questions/{post.get('id')}/")
        if len(rows) >= n:
            break
    if not rows:
        return ""
    return ("Related open questions on Metaculus with their community forecast (different questions: use only as context, "
            "check how they relate to this one):\n" + "\n".join(rows))
