"""Getrennte Zugangsdaten für Generator und Judge:

Generator und Judge laufen über verschiedene Endpunkte.
Jede Rolle hat eigene Variablen, mit Rückfall auf OPENAI_*:

    GENERATOR_BASE_URL / GENERATOR_API_KEY
    JUDGE_BASE_URL     / JUDGE_API_KEY
    OPENAI_BASE_URL    / OPENAI_API_KEY      (Rückfall für beide)

Die Framework-Adapter erzeugen ihre Clients über die OpenAI-SDK, die OPENAI_* aus der Umgebung liest. aktiviere() setzt diese Variablen deshalb zu Beginn eines Laufs auf die Werte der jeweiligen Rolle.
Generierung (scripts/02) und Bewertung (scripts/03) laufen in getrennten Prozessen.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

ROLLEN = ("generator", "judge")


def zugang(rolle: str) -> tuple[str | None, str | None]:
    """(api_key, base_url) für "generator" oder "judge"."""
    if rolle not in ROLLEN:
        raise ValueError(f"Unbekannte Rolle {rolle!r}; erlaubt: {ROLLEN}")
    load_dotenv()
    prefix = rolle.upper()
    key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get("OPENAI_API_KEY")
    url = os.environ.get(f"{prefix}_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    return key, url


def aktiviere(rolle: str) -> str | None:
    """Setzt OPENAI_* auf die Zugangsdaten der Rolle und gibt die base_url zurück."""
    key, url = zugang(rolle)
    if key:
        os.environ["OPENAI_API_KEY"] = key
    if url:
        os.environ["OPENAI_BASE_URL"] = url
    return url


def getrennt_konfiguriert() -> bool:
    """True, wenn Generator und Judge auf verschiedene Endpunkte zeigen.

    Sonst bewertet der Judge womöglich Antworten seines eigenen Modells. Der Aufrufer warnt dann nur, für Probeläufe ist ein gemeinsamer Endpunkt in Ordnung.
    """
    _, gen_url = zugang("generator")
    _, judge_url = zugang("judge")
    return bool(gen_url and judge_url and gen_url != judge_url)
