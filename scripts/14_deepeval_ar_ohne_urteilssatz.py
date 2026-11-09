"""Gegenprobe zu RQ4: DeepEval Answer Relevance ohne den festen Urteilssatz:

Für RQ4 setzt der Loader jede Antwort aus einem festen Satz ("Yes, the two configuration options depend on each other." oder die Verneinung) und der Begründung des Generators zusammen.
DeepEval Answer Relevance gibt den bejahenden Antworten höhere Scores.
Bei verneinenden Antworten nennt der Judge Aussagen über die Unabhängigkeit irrelevant, weil sie "der Frage widersprechen", und der feste Satz ist so eine Aussage.
Die Gegenprobe bewertet dieselben 200 Tripel mit derselben Metrik, demselben Judge und derselben Frage, aber nur mit der Begründung als Antwort.

Kostet Geld, der Verbrauch steht in results/usage.jsonl.
Schreibt nach jedem Tripel, ein Neustart setzt fort.

    python scripts/14_deepeval_ar_ohne_urteilssatz.py
"""

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


def main() -> None:
    cfg = load_config()
    judge_name = cfg["judge"]["model"].replace("/", "-")
    ziel = (
        Path(cfg["paths"]["results"])
        / "scores"
        / f"depvalidation_ohne_urteilssatz_deepeval_ar_{judge_name}.jsonl"
    )
    credentials.aktiviere("judge")
    vorher = bisher_ausgegeben(USAGE_LOG)
    limit = (cfg.get("budget") or {}).get("gesamt_eur")
    TRACKER.limit_eur = float(limit) if limit else None
    TRACKER.vorlauf_eur = vorher
    print(f"Bisher verbraucht: {vorher:.4f} €, Grenze {TRACKER.limit_eur} €")

    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    from rag_eval.data.schema import Construct
    from rag_eval.evaluation.deepeval_adapter import _measure_with_retry, _ScadsJudge

    judge = _ScadsJudge(
        cfg["judge"]["model"],
        cfg["judge"].get("temperature", 0.0),
        timeout=float(cfg["judge"].get("request_timeout", 120.0)),
    )
    fertig = set()
    if ziel.exists():
        fertig = {json.loads(l)["sample_id"] for l in open(ziel)}
    samples = [
        s for s in read_jsonl(QUELLE, GeneratedSample) if s.sample_id not in fertig
    ]
    print(f"{len(fertig)} schon bewertet, {len(samples)} offen → {ziel}")
    with open(ziel, "a", encoding="utf-8") as f:
        for i, s in enumerate(samples, 1):
            metric = AnswerRelevancyMetric(
                model=judge, async_mode=False, include_reason=True
            )
            tc = LLMTestCase(
                input=s.query,
                actual_output=ohne_urteilssatz(s),
                retrieval_context=s.retrieved_contexts,
            )
            try:
                _measure_with_retry(metric, tc, Construct.ANSWER_RELEVANCE, s.sample_id)
            except Exception as e:
                print(f"  ⚠ {s.sample_id}: {e}")
                continue
            f.write(
                json.dumps(
                    {
                        "sample_id": s.sample_id,
                        "score": float(metric.score),
                        "reasoning": getattr(metric, "reason", None),
                        "predicted": s.meta["predicted"],
                        "answer_correct": s.meta["answer_correct"],
                        "generator": s.meta["generator"],
                    }
                )
                + "\n"
            )
            f.flush()
            if i % 20 == 0:
                eur, _ = TRACKER.kosten_eur()
                print(f"  {i}/{len(samples)} · {eur:.4f} €", flush=True)
    print(TRACKER.bericht())
    kosten = protokollieren(
        USAGE_LOG,
        {
            "datasets": ["depvalidation_ohne_urteilssatz"],
            "frameworks": ["deepeval:answer_relevancy"],
            "sample_size": None,
            "out_dir": str(ziel.parent),
        },
    )
    print(f"Kosten: {kosten:.4f} €, gesamt {vorher + kosten:.4f} €")


if __name__ == "__main__":
    main()
