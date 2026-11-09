"""Erkennung von Abstinenzen, also Antworten, in denen das System nicht antwortet:

Der System-Prompt verlangt "If the context does not contain the answer, say so explicitly."
Ein großer Teil der Antworten tut das. Ob diese Antworten in die Korrelationen eingehen, verändert jede Kennzahl.
Die Regel wurde deshalb vor den Läufen festgelegt.

Eine Antwort, die eine Lücke benennt, ist nicht immer eine Verweigerung.
"No. Kyle Patrick Dickherber is known as the lead singer of The Click Five, but the context does not state ..." beantwortet die Frage und nennt zusätzlich eine Lücke.
Deshalb gibt es neben Antwort und Abstinenz die Teilantwort.

Eine Abstinenz liegt vor, wenn jeder Satz und Teilsatz der Antwort eine Lückenformel enthält.
Kommt ein Abschnitt ohne Lückenformel hinzu, sagt die Antwort inhaltlich etwas und ist eine Teilantwort.

Die Regel prüft nur den Antworttext, nie die Scores.
"""

from __future__ import annotations

import re
from enum import Enum

# Wendungen, mit denen der Generator eine Lücke im Kontext benennt. Gesammelt aus den erzeugten Antworten, weil der Generator freier formuliert als im Prompt vorgegeben.
LUECKEN_MUSTER = re.compile(
    r"do(?:es)? not contain"
    r"|do(?:es)? not (?:state|specify|mention|include|provide|say|indicate|detail)"
    r"|do(?:es)? not appear to (?:contain|state|mention)"
    r"|no (?:direct |relevant |specific )?information"
    r"|no mention"
    r"|not mentioned"
    r"|not specified"
    r"|is not (?:available|provided|present)"
    r"|cannot be (?:answered|determined)"
    r"|can(?:no|')t be (?:answered|determined)"
    r"|unable to (?:answer|determine)"
    r"|(?:not|isn't) enough information"
    r"|insufficient",
    re.IGNORECASE,
)

# Satzgrenze: Satzzeichen, Leerzeichen, dann ein Großbuchstabe oder ein Anführungszeichen.
_SATZ = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'*])")

# Zusätzlich an Teilsatzgrenzen trennen.
# Sonst gälte "... does not state X; it only mentions Y" als Abstinenz, obwohl der zweite Teil etwas aussagt.
# Dadurch zählen eher weniger Antworten als Abstinenz, nicht mehr.
_TEILSATZ = re.compile(r";\s+|,\s+(?:but|although|however|though)\b", re.IGNORECASE)


class Antworttyp(str, Enum):
    LEER = "leer"
    ABSTINENZ = "abstinenz"
    TEILANTWORT = "teilantwort"
    ANTWORT = "antwort"


def _saetze(text: str) -> list[str]:
    """Zerlegt die Antwort in Sätze und Teilsätze."""
    teile: list[str] = []
    for satz in _SATZ.split(text.strip()):
        teile.extend(t for t in _TEILSATZ.split(satz) if t.strip())
    return [t for t in teile if t.strip()]


def klassifiziere(antwort: str) -> Antworttyp:
    """Ordnet eine Antwort einem der vier Typen zu.

    Leere Antworten sind keine Abstinenz. Sie entstehen, wenn die Generierung fehlschlägt, und sind kein Verhalten des Systems.
    """
    if not antwort or not antwort.strip():
        return Antworttyp.LEER

    saetze = _saetze(antwort)
    mit_luecke = [bool(LUECKEN_MUSTER.search(s)) for s in saetze]

    if not any(mit_luecke):
        return Antworttyp.ANTWORT
    if all(mit_luecke):
        return Antworttyp.ABSTINENZ
    return Antworttyp.TEILANTWORT


def ist_abstinenz(antwort: str) -> bool:
    """Nur vollständige Abstinenzen, Teilantworten zählen nicht."""
    return klassifiziere(antwort) is Antworttyp.ABSTINENZ


def verteilung(antworten: list[str]) -> dict[str, int]:
    """Häufigkeit je Antworttyp."""
    zaehler = {t.value: 0 for t in Antworttyp}
    for a in antworten:
        zaehler[klassifiziere(a).value] += 1
    return zaehler
