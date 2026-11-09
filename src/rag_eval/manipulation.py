"""Manipulierte Antworten, um zu prüfen, ob die Metriken einen bekannten Fehler erkennen:

Korrelationen können nicht unterscheiden, ob alle Antworten wirklich treu sind oder ob die Metrik es nicht erkennt.
Bei einem Deckeneffekt, also wenn die meisten Faithfulness-Werte auf 1,0 liegen, sieht beides gleich aus.
Ein absichtlich eingebauter Fehler trennt die beiden Fälle.

Verändert wird nur die Antwort, der Kontext bleibt in allen Bedingungen gleich. Faithfulness soll reagieren, Context Relevance und Answer Relevance nicht.
Eine Metrik, die sich bewegt, obwohl ein anderes Konstrukt verändert wurde, trennt die Konstrukte nicht.

Die meisten Antworten bestehen aus einem einzigen Satz (71 % bei CRAG). Statt einen Teil der Antwort zu ersetzen, werden deshalb k ungedeckte Aussagen eingefügt (k = 1, 2, 3).
Bei einer Antwort aus einem Satz sind das 50 %, 67 % und 75 % ihrer Sätze.

Einfügen und Ersetzen wirken verschieden. Beim Ersetzen verschwindet der Teil, der die Frage beantwortet, und Answer Relevance sollte fallen.
Beim Einfügen bleibt er stehen. Für die inject-Bedingungen wird Answer Relevance deshalb als unverändert bis leicht niedriger erwartet.

Alle Auswahlen sind über seed festgelegt.
"""

from __future__ import annotations

import random
import re
from enum import Enum

from .analysis.abstinenz import Antworttyp, klassifiziere
from .data.schema import GeneratedSample

# Satzgrenze wie in analysis.abstinenz.
_SATZ = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'*])")


class Bedingung(str, Enum):
    """Die Manipulationsbedingungen."""

    NORMAL = "normal"  # unveränderte Antwort (schon bewertet)
    INJECT_1 = "inject_1"  # 1 ungedeckte Aussage eingefügt
    INJECT_2 = "inject_2"
    INJECT_3 = "inject_3"
    REPLACE_FULL = "replace_full"  # Antwort ganz durch eine fremde ersetzt
    SWAP_FACT = "swap_fact"  # eine Zahl der Antwort gegen eine falsche getauscht


# Erwartetes Verhalten je Bedingung, vor den Läufen festgelegt.
ERWARTUNG: dict[str, dict[str, str]] = {
    # Beim Einfügen bleibt die Originalantwort ganz erhalten und beantwortet die Frage weiter. Eine stabile Answer Relevance ist hier also erwartet.
    Bedingung.INJECT_1.value: {
        "context_relevance": "unverändert — der Kontext wird nicht angefasst",
        "faithfulness": "niedriger",
        "answer_relevance": "unverändert bis leicht niedriger — die Originalantwort "
        "bleibt erhalten, es kommt nur Ungedecktes hinzu",
    },
}
ERWARTUNG[Bedingung.INJECT_2.value] = {
    **ERWARTUNG[Bedingung.INJECT_1.value],
    "faithfulness": "niedriger als inject_1",
}
ERWARTUNG[Bedingung.INJECT_3.value] = {
    **ERWARTUNG[Bedingung.INJECT_1.value],
    "faithfulness": "niedriger als inject_2",
}
# Das stärkste Ende der Reihe. Der Teil, der die Frage beantwortet, verschwindet, Answer Relevance sollte also fallen. Einfügen trifft nur Faithfulness, Ersetzen trifft beide.
ERWARTUNG[Bedingung.REPLACE_FULL.value] = {
    "context_relevance": "unverändert — der Kontext wird nicht angefasst",
    "faithfulness": "am niedrigsten",
    "answer_relevance": "deutlich niedriger — die Antwort beantwortet die Frage nicht mehr",
}
# Die inject-Bedingungen machen die Antwort ungedeckt, themenfremd und länger. Hier bleibt sie beim Thema und gleich lang, nur eine Zahl ist falsch.
# Weil die richtige Zahl im Kontext steht, widerspricht die falsche ihm.
# DeepEval zählt alles außer einem Widerspruch als treu und sollte deshalb hier, anders als beim Einfügen, auch fallen.
ERWARTUNG[Bedingung.SWAP_FACT.value] = {
    "context_relevance": "unverändert — der Kontext wird nicht angefasst",
    "faithfulness": "niedriger, bei allen drei Frameworks — der getauschte Wert "
    "widerspricht dem Kontext, in dem der richtige steht",
    "answer_relevance": "unverändert — die Antwort bleibt beim Thema und "
    "beantwortet die Frage weiterhin",
}


def saetze(text: str) -> list[str]:
    """Zerlegt die Antwort in Sätze, leere Teile fallen weg."""
    return [s.strip() for s in _SATZ.split((text or "").strip()) if s.strip()]


def _norm(text: str) -> str:
    return " ".join((text or "").split())


# Ungedeckte Aussagen einfügen und Antwort ersetzen


def inject_claims(
    samples: list[GeneratedSample], k: int, seed: int = 42
) -> list[GeneratedSample]:
    """Fügt k ungedeckte Aussagen aus Antworten auf andere Fragen ein.

    Der Kontext bleibt gleich, die eingefügten Aussagen sind deshalb ungedeckt. Faithfulness sollte fallen, Context Relevance sich nicht bewegen.
    Jede Aussage kommt aus einer anderen Antwort.
    Samples ohne Text werden übersprungen.
    """
    if k < 1:
        raise ValueError("k muss mindestens 1 sein")
    rng = random.Random(seed + k)

    # Ziele: alle Antworten mit Text, auch Abstinenzen. Sie liegen bei Faithfulness meist auf 1,0, eine eingefügte Aussage macht sie ungedeckt.
    verwertbar = [s for s in samples if saetze(s.generated_answer)]

    # Spender: nur Antworten, die etwas behaupten. Eine Abstinenz ("The provided context does not contain ...") behauptet nichts und wäre als eingefügter Satz selbst treu.
    spenderpool = [
        s
        for s in verwertbar
        if klassifiziere(s.generated_answer)
        in (Antworttyp.ANTWORT, Antworttyp.TEILANTWORT)
    ]
    if len(spenderpool) <= k:
        return []

    out = []
    for s in verwertbar:
        spender = _waehle_spender(s, spenderpool, rng, anzahl=k)
        if len(spender) < k:
            continue
        # Von jedem Spender der letzte Satz.
        zusatz = [saetze(d.generated_answer)[-1] for d in spender]
        text = " ".join([s.generated_answer.strip(), *zusatz])
        out.append(
            GeneratedSample(
                sample_id=s.sample_id,
                dataset=s.dataset,
                query=s.query,
                retrieved_contexts=list(s.retrieved_contexts),
                generated_answer=text,
                reference_answer=s.reference_answer,
                meta={
                    **s.meta,
                    "manipulation": f"inject_{k}",
                    "manipulation_seite": "generierung",
                    "manipulation_quelle": [d.sample_id for d in spender],
                    "eingefuegte_saetze": zusatz,
                    "saetze_original": len(saetze(s.generated_answer)),
                },
            )
        )
    return out


def replace_answer(
    samples: list[GeneratedSample], seed: int = 42
) -> list[GeneratedSample]:
    """Ersetzt die Antwort ganz durch die Antwort auf eine andere Frage.

    Einen Teil einer Antwort aus einem Satz kann man nicht ersetzen, die ganze Antwort dagegen immer.
    Anders als beim Einfügen verschwindet der Teil, der die Frage beantwortet, und Answer Relevance sollte fallen.
    Zusammen zeigen beide Bedingungen, ob eine Metrik auf Ungedecktes reagiert oder nur darauf, dass die Frage unbeantwortet bleibt.

    Der Kontext bleibt gleich, Context Relevance sollte sich nicht bewegen.
    """
    rng = random.Random(seed + 100)
    verwertbar = [s for s in samples if saetze(s.generated_answer)]
    # Spender wie beim Einfügen: nur Antworten, die etwas behaupten.
    spenderpool = [
        s
        for s in verwertbar
        if klassifiziere(s.generated_answer)
        in (Antworttyp.ANTWORT, Antworttyp.TEILANTWORT)
    ]
    if len(spenderpool) <= 1:
        return []

    out = []
    for s in verwertbar:
        spender = _waehle_spender(s, spenderpool, rng, anzahl=1)
        if not spender:
            continue
        d = spender[0]
        out.append(
            GeneratedSample(
                sample_id=s.sample_id,
                dataset=s.dataset,
                query=s.query,
                retrieved_contexts=list(s.retrieved_contexts),
                generated_answer=d.generated_answer.strip(),
                reference_answer=s.reference_answer,
                meta={
                    **s.meta,
                    "manipulation": Bedingung.REPLACE_FULL.value,
                    "manipulation_seite": "generierung",
                    "manipulation_quelle": [d.sample_id],
                    "ersetzter_anteil": 1.0,
                    "saetze_original": len(saetze(s.generated_answer)),
                },
            )
        )
    return out


def _waehle_spender(
    s: GeneratedSample,
    pool: list[GeneratedSample],
    rng: random.Random,
    anzahl: int = 1,
    versuche: int = 60,
) -> list[GeneratedSample]:
    """Wählt anzahl verschiedene Samples, deren Kontext sich vom eigenen unterscheidet.

    Bei geteiltem Kontext könnte die eingefügte Aussage doch gedeckt sein.
    """
    eigen = {_norm(c) for c in s.retrieved_contexts}
    gewaehlt: list[GeneratedSample] = []
    gesehen: set[str] = {s.sample_id}
    for _ in range(versuche):
        if len(gewaehlt) >= anzahl:
            break
        kand = pool[rng.randrange(len(pool))]
        if kand.sample_id in gesehen:
            continue
        if eigen & {_norm(c) for c in kand.retrieved_contexts}:
            continue
        gesehen.add(kand.sample_id)
        gewaehlt.append(kand)
    return gewaehlt


# Eine Zahl tauschen
# Eine Ziffer direkt an einem Buchstaben ("A4B", "v0.2") zählt nicht.
_ZAHL = re.compile(
    r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.,])\d+(?:\.\d+)?(?![\w])"
)


def _kommt_vor(wert: str, text: str) -> bool:
    """Steht wert für sich in text, ohne Ziffer oder Buchstabe direkt daneben?"""
    return (
        re.search(rf"(?<![\w]){re.escape(wert)}(?![\w])", text, re.IGNORECASE)
        is not None
    )


def _zahl_wert(s: str) -> float:
    return float(s.replace(",", ""))


def _zahl_format(wert: float, vorlage: str) -> str:
    nachkomma = len(vorlage.split(".")[1]) if "." in vorlage else 0
    if nachkomma:
        return f"{wert:,.{nachkomma}f}" if "," in vorlage else f"{wert:.{nachkomma}f}"
    return f"{int(round(wert)):,}" if "," in vorlage else str(int(round(wert)))


def _ist_prozent(z: str, antwort: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(z)}\s*(?:%|percent)", antwort) is not None


_MONAT = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)


def _ist_tag(z: str, antwort: str) -> bool:
    """Steht die Zahl als Tag neben einem Monatsnamen ("21 November", "March 13")?

    Nur ganze Zahlen bis 31, sonst gälte "February 2024" als Tag.
    """
    if not z.isdigit() or int(z) > 31:
        return False
    zz = re.escape(z)
    return (
        re.search(
            rf"(?<![\w]){zz}\s+(?:{_MONAT})\b|\b(?:{_MONAT})\s+{zz}(?![\w])", antwort
        )
        is not None
    )


def _falsche_zahl(
    z: str, verboten: str, rng: random.Random, hoechstens: float | None = None
) -> str | None:
    """Eine falsche, aber plausible Zahl gleicher Form, die in verboten nicht vorkommt.

    Jahre werden um 1 bis 7 verschoben, alle anderen Zahlen um 20 bis 50 % verändert.
    So bleibt die Größenordnung erhalten. Prozentwerte bleiben höchstens bei 100 (hoechstens), sonst "126 % der Einwohner".
    """
    w = _zahl_wert(z)
    ist_jahr = "," not in z and "." not in z and len(z) == 4 and 1000 <= w <= 2100
    if ist_jahr:
        # Nicht über 2024 hinaus, weil die CRAG-Fragen von 2024 sind und ein späteres Jahr offensichtlich falsch wäre.
        kandidaten = [
            str(int(w) + d)
            for d in range(-7, 8)
            if d and int(w) + d <= max(2024, int(w))
        ]
    else:
        kandidaten = [
            _zahl_format(w * f, z) for f in (0.5, 0.6, 0.7, 0.8, 1.2, 1.3, 1.4, 1.5)
        ]
    rng.shuffle(kandidaten)
    for k in kandidaten:
        if (
            k == z
            or _kommt_vor(k, verboten)
            or _kommt_vor(k.replace(",", ""), verboten)
        ):
            continue
        if hoechstens is not None and _zahl_wert(k) > hoechstens:
            continue
        return k
    return None


def swap_fact(samples: list[GeneratedSample], seed: int = 42) -> list[GeneratedSample]:
    """Tauscht genau eine Zahl der Antwort gegen eine falsche.

    Beim Einfügen wird die Antwort ungedeckt, themenfremd und länger zugleich.
    Hier bleibt sie beim Thema und gleich lang, nur eine Zahl wird falsch. So lässt sich trennen, ob eine Metrik auf fehlende Deckung reagiert oder auf ein fremdes Thema.

    Regeln:
      * Nur Antworten, die etwas behaupten (Antwort, Teilantwort). Eine Abstinenz enthält keinen Wert, den man verfälschen könnte.
      * Getauscht wird eine Zahl ab 10, die im Kontext steht und nicht in der Frage. Steht sie im Kontext, widerspricht der falsche Wert ihm.
        Stünde sie in der Frage, würde der Tausch die Frage verändern. Kleinere Zahlen sind oft Listenpunkte oder Verhältnisse.
      * Jahre werden um 1 bis 7 verschoben, aber nicht über 2024 hinaus, andere Zahlen um 20 bis 50 %. Prozentwerte bleiben höchstens bei 100, Tage neben einem Monatsnamen höchstens bei 31.
      * Der neue Wert kommt weder im Kontext noch in der Frage noch sonst in der Antwort vor, sonst wäre er vielleicht gedeckt.
        Jedes Vorkommen des alten Werts wird ersetzt, damit die Antwort in sich stimmig bleibt.

    Namen werden nicht getauscht. Ohne zu wissen, ob ein Name eine Person, eine Band oder einen Ort bezeichnet, würde ein Tausch die Antwort oft vom Thema wegführen,
    z.B. "Led Zeppelin" -> "North America".
    """
    rng = random.Random(seed + 200)
    ziele = [
        s
        for s in samples
        if saetze(s.generated_answer)
        and klassifiziere(s.generated_answer)
        in (Antworttyp.ANTWORT, Antworttyp.TEILANTWORT)
    ]
    out = []
    for s in ziele:
        antwort = s.generated_answer
        ctx = " ".join(s.retrieved_contexts)
        verboten = f"{ctx}\n{s.query}\n{antwort}"
        zahlen = [
            z
            for z in dict.fromkeys(_ZAHL.findall(antwort))
            if _zahl_wert(z) >= 10 and _kommt_vor(z, ctx) and not _kommt_vor(z, s.query)
        ]
        rng.shuffle(zahlen)
        alt = neu = None
        for z in zahlen:
            # Obergrenze für Prozentwerte und Tage, sonst wird z.B. aus "21 November" ein "32 November".
            grenze = (
                100
                if _ist_prozent(z, antwort)
                else (31 if _ist_tag(z, antwort) else None)
            )
            k = _falsche_zahl(z, verboten, rng, grenze)
            if k:
                alt, neu = z, k
                break
        if alt is None:
            continue
        text = re.sub(rf"(?<![\w]){re.escape(alt)}(?![\w])", neu, antwort)
        if text == antwort or _kommt_vor(alt, text):
            continue
        out.append(
            GeneratedSample(
                sample_id=s.sample_id,
                dataset=s.dataset,
                query=s.query,
                retrieved_contexts=list(s.retrieved_contexts),
                generated_answer=text,
                reference_answer=s.reference_answer,
                meta={
                    **s.meta,
                    "manipulation": Bedingung.SWAP_FACT.value,
                    "manipulation_seite": "generierung",
                    "wert_original": alt,
                    "wert_neu": neu,
                    "saetze_original": len(saetze(antwort)),
                },
            )
        )
    return out
