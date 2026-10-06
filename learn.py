"""
learn.py - El bot aprende de sus propios resultados.

1. Lee data/forecast_log.jsonl (cada predicción individual de cada modelo).
2. Pregunta a Metaculus cómo se resolvió cada pregunta (cache en data/resolutions.json).
3. Calcula, con las preguntas sí/no ya resueltas:
     - qué modelos aciertan más (log score) -> pesos en config/bot_config.json
     - si el conjunto es demasiado tímido o demasiado atrevido -> recalibración Platt (a, b)
   Con pocos datos los cambios se frenan (encogimiento hacia "no tocar nada").
4. Escribe un informe en data/learn_report.md.

Uso: python learn.py            (necesita METACULUS_TOKEN)
     python learn.py --offline  (solo con la cache, sin llamar a Metaculus)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "data" / "forecast_log.jsonl"
RES_PATH = ROOT / "data" / "resolutions.json"
CONFIG_PATH = ROOT / "config" / "bot_config.json"
REPORT_PATH = ROOT / "data" / "learn_report.md"

MIN_N_PLATT = 30        # preguntas sí/no resueltas antes de recalibrar
MIN_N_MODEL = 20        # predicciones resueltas de un modelo antes de cambiar su peso
PRIOR_STRENGTH = 40.0   # cuánto "pesa" la opción de no tocar nada (en preguntas equivalentes)

logger = logging.getLogger("learn")


def clip(p: float, eps: float = 0.01) -> float:
    return min(max(p, eps), 1 - eps)


def logit(p: float) -> float:
    p = clip(p, 1e-4)
    return math.log(p / (1 - p))


def logloss(p: float, y: int) -> float:
    p = clip(p)
    return -math.log(p if y else 1 - p)


def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    out = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def fetch_resolutions(post_ids: list[int], cache: dict[str, dict], offline: bool) -> dict[str, dict]:
    if offline:
        return cache
    from forecasting_tools import MetaculusClient  # import tardío: --offline no lo necesita

    client = MetaculusClient()
    for pid in post_ids:
        key = str(pid)
        if key in cache and cache[key].get("final"):
            continue
        try:
            q = client.get_question_by_post_id(pid)
        except Exception as e:
            logger.warning(f"No se pudo leer la pregunta {pid}: {e}")
            continue
        res = getattr(q, "resolution_string", None)
        state = str(getattr(q, "state", "") or "")
        cache[key] = {
            "resolution": res,
            "state": state,
            "final": res is not None,
            "checked": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    return cache


def fit_platt(xs: list[float], ys: list[int]) -> tuple[float, float]:
    """Regresión logística y ~ sigmoid(a + b*x) con un 'prior' que tira hacia a=0, b=1."""
    a, b = 0.0, 1.0
    lam = PRIOR_STRENGTH / max(len(xs), 1)
    n = len(xs)
    for _ in range(50):
        ga = gb = 0.0
        haa = hab = hbb = 0.0
        for x, y in zip(xs, ys):
            p = 1 / (1 + math.exp(-(a + b * x)))
            ga += (p - y)
            gb += (p - y) * x
            w = p * (1 - p)
            haa += w
            hab += w * x
            hbb += w * x * x
        # penalización L2 hacia (0, 1), escalada al número de datos
        ga += lam * n * (a - 0.0) * 0.1
        gb += lam * n * (b - 1.0)
        haa += lam * n * 0.1
        hbb += lam * n
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a -= da
        b -= db
        if abs(da) < 1e-8 and abs(db) < 1e-8:
            break
    return max(-0.8, min(0.8, a)), max(0.6, min(1.8, b))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    entries = read_log()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cache = json.loads(RES_PATH.read_text(encoding="utf-8")) if RES_PATH.exists() else {}

    # última predicción de cada pregunta (la que cuenta si se repitió)
    latest: dict[int, dict] = {}
    for e in entries:
        if e.get("post_id") is None:
            continue
        latest[e["post_id"]] = e
    cache = fetch_resolutions(sorted(latest), cache, args.offline)
    RES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RES_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")

    rows = []  # (entry, y)
    for pid, e in latest.items():
        if e.get("type") != "BinaryQuestion" or e.get("raw") is None:
            continue
        r = (cache.get(str(pid)) or {}).get("resolution")
        if r in ("yes", "no"):
            rows.append((e, 1 if r == "yes" else 0))

    n = len(rows)
    lines = [f"# Informe de aprendizaje ({datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M')} UTC)", ""]
    lines.append(f"- Pronósticos registrados: {len(latest)} preguntas · sí/no ya resueltas: **{n}**")

    changed = False
    if n:
        ll_raw = sum(logloss(e["raw"], y) for e, y in rows) / n
        ll_fin = sum(logloss(e["final"], y) for e, y in rows) / n
        br_fin = sum(brier(e["final"], y) for e, y in rows) / n
        base = sum(y for _, y in rows) / n
        lines.append(f"- Pérdida logarítmica media: conjunto bruto {ll_raw:.3f} · enviado {ll_fin:.3f} (menos es mejor; 0,693 = decir siempre 50 %)")
        lines.append(f"- Brier medio enviado: {br_fin:.3f} · proporción de SÍ: {base:.0%}")

        # ---- pesos por modelo
        per_model: dict[str, list[float]] = defaultdict(list)
        for e, y in rows:
            for m in e.get("models", []):
                if isinstance(m.get("p"), (int, float)):
                    per_model[m["model"]].append(logloss(float(m["p"]), y))
        lines += ["", "## Modelos", "", "| Modelo | Resueltas | Pérdida log | Peso nuevo |", "|---|---|---|---|"]
        med = sorted(sum(v) / len(v) for v in per_model.values())[len(per_model) // 2] if per_model else 0.0
        weights = dict(cfg.get("weights", {}))
        for model, losses in sorted(per_model.items()):
            avg = sum(losses) / len(losses)
            w_old = float(weights.get(model, 1.0))
            if len(losses) >= MIN_N_MODEL:
                raw_w = math.exp(-8.0 * (avg - med))
                shrink = len(losses) / (len(losses) + 50.0)
                w_new = round(max(0.3, min(2.0, 1.0 + (raw_w - 1.0) * shrink)), 3)
            else:
                w_new = w_old
            if abs(w_new - w_old) > 1e-6:
                changed = True
            weights[model] = w_new
            lines.append(f"| {model} | {len(losses)} | {avg:.3f} | {w_new:.2f} |")
        cfg["weights"] = weights

        # ---- recalibración Platt sobre el conjunto bruto
        if n >= MIN_N_PLATT:
            a, b = fit_platt([logit(e["raw"]) for e, _ in rows], [y for _, y in rows])
            old = cfg.get("platt", {"a": 0.0, "b": 1.0})
            cfg["platt"] = {"a": round(a, 4), "b": round(b, 4), "n": n}
            changed = changed or abs(a - old.get("a", 0)) > 1e-4 or abs(b - old.get("b", 1)) > 1e-4
            sense = "más atrevido (las predicciones eran demasiado tímidas)" if b > 1.02 else \
                    "más prudente (las predicciones eran demasiado extremas)" if b < 0.98 else "sin cambio de confianza"
            lines += ["", f"## Calibración", "", f"- Nuevo ajuste: a = {a:+.3f}, b = {b:.3f} → {sense}."]
            if abs(a) > 0.05:
                lines.append(f"- Sesgo: tendía a {'quedarse corto con el SÍ' if a > 0 else 'pasarse con el SÍ'}.")
            # cuánto habría mejorado con el ajuste nuevo (en la misma muestra: orientativo)
            ll_new = sum(logloss(1 / (1 + math.exp(-(a + b * logit(e["raw"])))), y) for e, y in rows) / n
            lines.append(f"- Con el ajuste, la pérdida en estas preguntas habría sido {ll_new:.3f} (antes {ll_raw:.3f}).")
        else:
            lines += ["", f"## Calibración", "", f"- Aún no se toca: hacen falta {MIN_N_PLATT} preguntas sí/no resueltas (hay {n})."]

        # ---- peores fallos, para revisar
        worst = sorted(rows, key=lambda r: -logloss(r[0]["final"], r[1]))[:5]
        lines += ["", "## Peores fallos", ""]
        for e, y in worst:
            lines.append(f"- {e.get('title', '')[:110]} → dijimos {e['final']:.0%}, salió {'SÍ' if y else 'NO'} ({e.get('url')})")
    else:
        lines.append("- Aún no hay preguntas resueltas: no se cambia nada.")

    if changed:
        cfg["version"] = int(cfg.get("version", 1)) + 1
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        lines.append(f"\nConfiguración actualizada a la versión {cfg['version']}.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
