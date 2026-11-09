"""Gegenprobe zu RQ4 für Ragas und TruLens Answer Relevance ohne den festen Urteilssatz:

Ergänzt scripts/14 (DeepEval).
Dieselben Bausteine wie im Hauptlauf: Ragas AnswerRelevancy mit demselben Judge und Embedding-Modell (_build_components), TruLens relevance_with_cot_reasons mit derselben Skala (_build_provider, score_range aus der config).
Anders ist nur die Antwort, die Begründung allein.

Kostet Geld, der Verbrauch steht in results/usage.jsonl.
Schreibt nach jedem Tripel, ein Neustart setzt fort.

    python scripts/15_ragas_trulens_ar_ohne_urteilssatz.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_eval import credentials
from rag_eval.config import load_config
from rag_eval.data.io import read_jsonl
from rag_eval.data.loaders.depvalidation import _VERDICT
from rag_eval.data.schema import GeneratedSample
from rag_eval.evaluation.usage import TRACKER, bisher_ausgegeben, protokollieren

USAGE_LOG = "results/usage.jsonl"
QUELLE = "data/processed/depvalidation_generated.jsonl"


def ohne_urteilssatz(s: GeneratedSample) -> str:
    satz = _VERDICT[bool(s.meta["predicted"])]
    assert s.generated_answer.startswith(satz + " "), s.sample_id
    return s.generated_answer[len(satz) + 1 :].strip()


def _fertig(pfad: Path) -> set[str]:
    return {json.loads(l)["sample_id"] for l in open(pfad)} if pfad.exists() else set()


def main() -> None:
    cfg = load_config()
    judge = cfg["judge"]["model"].replace("/", "-")
    ziel = {
        fw: Path(cfg["paths"]["results"])
        / "scores"
        / f"depvalidation_ohne_urteilssatz_{fw}_ar_{judge}.jsonl"
        for fw in ("ragas", "trulens")
    }
    credentials.aktiviere("judge")
    vorher = bisher_ausgegeben(USAGE_LOG)
    limit = (cfg.get("budget") or {}).get("gesamt_eur")
    TRACKER.limit_eur = float(limit) if limit else None
    TRACKER.vorlauf_eur = vorher
    print(f"Bisher verbraucht: {vorher:.4f} €, Grenze {TRACKER.limit_eur} €")
    alle = read_jsonl(QUELLE, GeneratedSample)

    # Ragas
    from rag_eval.evaluation.ragas_adapter import (
        _build_components,
        _install_langchain_vertexai_shim,
    )

    _install_langchain_vertexai_shim()
    from ragas.metrics.collections import AnswerRelevancy

    llm, embeddings = _build_components(cfg)
    metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
    offen = [s for s in alle if s.sample_id not in _fertig(ziel["ragas"])]
    print(f"Ragas: {len(offen)} offen")

    async def ragas_lauf():
        with open(ziel["ragas"], "a", encoding="utf-8") as f:
            for i, s in enumerate(offen, 1):
                try:
                    r = await metric.ascore(
                        user_input=s.query, response=ohne_urteilssatz(s)
                    )
                    wert = float(r.value)
                    if wert != wert:
                        raise ValueError("NaN")
                except Exception as e:
                    print(f"  ⚠ ragas {s.sample_id}: {e}")
                    continue
                f.write(json.dumps({"sample_id": s.sample_id, "score": wert}) + "\n")
                f.flush()
                if i % 50 == 0:
                    print(
                        f"  ragas {i}/{len(offen)} · {TRACKER.kosten_eur()[0]:.4f} €",
                        flush=True,
                    )

    asyncio.run(ragas_lauf())

    # TruLens
    from rag_eval.evaluation.trulens_adapter import (
        _DEFAULT_SCORE_RANGE,
        _build_provider,
        _reason_text,
    )

    fw_cfg = cfg.get("frameworks", {}).get("trulens", {}) or {}
    provider = _build_provider(
        cfg["judge"]["model"],
        rpm=fw_cfg.get("rpm", 60),
        timeout=float(cfg["judge"].get("request_timeout", 120.0)),
    )
    lo, hi = tuple(fw_cfg.get("score_range") or _DEFAULT_SCORE_RANGE)
    common = {
        "min_score_val": lo,
        "max_score_val": hi,
        "temperature": cfg["judge"].get("temperature", 0.0),
    }
    offen = [s for s in alle if s.sample_id not in _fertig(ziel["trulens"])]
    print(f"TruLens: {len(offen)} offen (Skala {lo}–{hi})")
    with open(ziel["trulens"], "a", encoding="utf-8") as f:
        for i, s in enumerate(offen, 1):
            try:
                score, reasons = provider.relevance_with_cot_reasons(
                    prompt=s.query, response=ohne_urteilssatz(s), **common
                )
            except Exception as e:
                print(f"  ⚠ trulens {s.sample_id}: {e}")
                continue
            f.write(
                json.dumps(
                    {
                        "sample_id": s.sample_id,
                        "score": float(score),
                        "reasoning": _reason_text(reasons),
                    }
                )
                + "\n"
            )
            f.flush()
            if i % 50 == 0:
                print(
                    f"  trulens {i}/{len(offen)} · {TRACKER.kosten_eur()[0]:.4f} €",
                    flush=True,
                )

    print(TRACKER.bericht())
    kosten = protokollieren(
        USAGE_LOG,
        {
            "datasets": ["depvalidation_ohne_urteilssatz"],
            "frameworks": ["ragas:answer_relevancy", "trulens:answer_relevance"],
            "sample_size": None,
            "out_dir": str(ziel["ragas"].parent),
        },
    )
    print(f"Kosten: {kosten:.4f} €, gesamt {vorher + kosten:.4f} €")


if __name__ == "__main__":
    main()
