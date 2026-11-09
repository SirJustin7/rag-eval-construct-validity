"""Generation: Frage und Chunks in den Prompt einsetzen und den Generator aufrufen:"""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from .. import credentials

# Sekunden je Anfrage, danach wird mit Backoff wiederholt. Ohne den Wert gilt der OpenAI-Standard von 600 s, ein hängender Aufruf blockiert dann bei vier Versuchen bis zu 40 Minuten.
DEFAULT_REQUEST_TIMEOUT = 120.0

# Ein Client je Timeout-Wert.
_clients: dict[float, OpenAI] = {}


def _get_client(timeout: float) -> OpenAI:
    if timeout not in _clients:
        credentials.aktiviere("generator")
        # max_retries=0, weil generate_answer selbst wiederholt.
        _clients[timeout] = OpenAI(timeout=timeout, max_retries=0)
    return _clients[timeout]


def build_prompt(query: str, contexts: list[str]) -> str:
    """Setzt Frage und Chunks in das Prompt-Template ein."""
    context_block = "\n\n---\n\n".join(contexts)
    return f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"


def generate_answer(query: str, contexts: list[str], cfg: dict[str, Any]) -> str:
    """Erzeugt die Antwort des Generators.

    cfg ist der rag-Block der Konfiguration.
    """
    gen = cfg["generator"]
    prompt = build_prompt(query, contexts)
    client = _get_client(float(gen.get("request_timeout", DEFAULT_REQUEST_TIMEOUT)))

    # Bis zu vier Versuche mit exponentiellem Backoff, weil der Endpunkt zeitweise mit Fehlern oder Überlast antwortet.
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=gen["model"],
                messages=[
                    {"role": "system", "content": gen["system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
                temperature=gen["temperature"],
                max_tokens=gen["max_tokens"],
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as e:
            last_error = e
            time.sleep(2**attempt)  # 1, 2, 4, 8 s

    raise RuntimeError(
        f"Generierung nach 4 Versuchen fehlgeschlagen (Modell {gen['model']}): {last_error}"
    )


def pruefe_modell_verfuegbar(cfg: dict[str, Any]) -> None:
    """Prüft vor dem Lauf, ob das Generator-Modell am Endpunkt angeboten wird.

    Fehlt es, scheitert sonst jedes Sample erst nach allen Wiederholungen mit einer wenig aussagekräftigen Meldung.
    Hier wird stattdessen sofort abgebrochen und die verfügbare Modellliste ausgegeben.
    """
    gen = cfg["generator"]
    client = _get_client(float(gen.get("request_timeout", DEFAULT_REQUEST_TIMEOUT)))
    gewuenscht = gen["model"]
    try:
        verfuegbar = sorted(m.id for m in client.models.list().data)
    except Exception as e:
        # Liste nicht abrufbar, der Lauf wird trotzdem versucht.
        print(f"[Vorabprüfung] Modellliste nicht abrufbar ({e}); fahre fort.")
        return
    if gewuenscht in verfuegbar:
        print(f"[Vorabprüfung] Generator '{gewuenscht}' ist verfügbar.")
        return
    liste = "\n  ".join(verfuegbar)
    raise RuntimeError(
        f"Generator-Modell '{gewuenscht}' wird vom Endpunkt NICHT angeboten.\n"
        f"Abbruch vor dem Lauf — sonst scheitert jedes einzelne Sample.\n\n"
        f"Verfügbar sind:\n  {liste}\n\n"
        f"→ rag.generator.model in config/config.yaml anpassen."
    )
