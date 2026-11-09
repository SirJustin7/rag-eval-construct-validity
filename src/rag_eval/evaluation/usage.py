"""Token- und Kostenzählung für die Judge-Aufrufe:

Der Judge gpt-4o-mini kostet Geld, das Budget war begrenzt. Ein Lauf soll sich deshalb vorher hochrechnen und hinterher abrechnen lassen.

track() hängt sich an client.chat.completions.create und liest response.usage mit. Der Client wird gepatcht, bevor instructor ihn übernimmt, so werden auch die strukturierten Aufrufe gezählt.
An den Frameworks selbst ändert sich nichts.

Gezählt wird nur, was über diese Clients läuft. Die lokalen Embeddings tauchen nicht auf, kosten aber auch nichts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# OpenAI-Listenpreise in USD je 1 Mio. Tokens (Input, Output). Modelle ohne Eintrag werden gezählt, aber ohne Kosten.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
}
USD_TO_EUR = 0.92


def _price_for(model: str) -> tuple[float, float] | None:
    """Preis für ein Modell, auch mit Anbieter-Präfix oder Datum im Namen."""
    if model in PRICES:
        return PRICES[model]
    kurz = model.split("/")[-1]  # "openai/gpt-4o" -> "gpt-4o"
    if kurz in PRICES:
        return PRICES[kurz]
    # längster passender Anfang, damit "gpt-4o-mini-2024-07-18" passt
    treffer = [k for k in PRICES if kurz.startswith(k)]
    return PRICES[max(treffer, key=len)] if treffer else None


@dataclass
class ModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    fehler: int = 0  # Aufrufe ohne verwertbares usage-Feld

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BudgetUeberschritten(BaseException):
    """Budgetgrenze erreicht, der Lauf bricht sofort ab.

    Von BaseException statt Exception abgeleitet, weil die Adapter mit except Exception abfangen und wiederholen.
    Eine normale Ausnahme würde dort geschluckt und der Aufruf erneut bezahlt.
    """


@dataclass
class UsageTracker:
    """Sammelt Tokenverbrauch je Modell und rechnet ihn in Euro um."""

    je_modell: dict[str, ModelUsage] = field(default_factory=dict)

    # Obergrenze in EUR. None = keine Grenze.
    # scripts/03 setzt sie aus der Konfiguration.
    limit_eur: float | None = None

    # Vor diesem Lauf schon ausgegeben (aus results/usage.jsonl).
    vorlauf_eur: float = 0.0

    # Angefragter Modellname -> Version, die tatsächlich geantwortet hat, z.B. "gpt-4o-mini" -> "gpt-4o-mini-2024-07-18".
    aufgeloeste_modelle: dict[str, set[str]] = field(default_factory=dict)

    def record(self, model: str, usage: Any, aufgeloest: str | None = None) -> None:
        m = self.je_modell.setdefault(model, ModelUsage())
        m.calls += 1
        if aufgeloest:
            self.aufgeloeste_modelle.setdefault(model, set()).add(aufgeloest)
        if usage is None:
            m.fehler += 1
            return
        m.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        m.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self._pruefe_budget()

    def _pruefe_budget(self) -> None:
        if self.limit_eur is None:
            return
        aktuell, _ = self.kosten_eur()
        gesamt = self.vorlauf_eur + aktuell
        if gesamt >= self.limit_eur:
            raise BudgetUeberschritten(
                f"Budgetgrenze erreicht: {gesamt:.4f} € von {self.limit_eur:.2f} € "
                f"({self.vorlauf_eur:.4f} € aus früheren Läufen + {aktuell:.4f} € "
                f"in diesem). Abbruch. Bereits vollständig bewertete Samples sind "
                f"gespeichert; ein erneuter Start setzt dort fort."
            )

    def kosten_eur(self) -> tuple[float, list[str]]:
        """Summe in EUR und Liste der Modelle ohne Preis."""
        summe, unbekannt = 0.0, []
        for model, u in self.je_modell.items():
            p = _price_for(model)
            if p is None:
                unbekannt.append(model)
                continue
            summe += (u.prompt_tokens * p[0] + u.completion_tokens * p[1]) / 1e6
        return summe * USD_TO_EUR, unbekannt

    def reset(self) -> None:
        self.je_modell.clear()

    def bericht(self) -> str:
        if not self.je_modell:
            return "Keine API-Aufrufe erfasst."
        zeilen = [f"{'Modell':<28}{'Aufrufe':>9}{'Input':>11}{'Output':>10}{'EUR':>9}"]
        for model, u in sorted(self.je_modell.items()):
            p = _price_for(model)
            c = (
                f"{(u.prompt_tokens * p[0] + u.completion_tokens * p[1]) / 1e6 * USD_TO_EUR:>9.4f}"
                if p
                else f"{'?':>9}"
            )
            zeilen.append(
                f"{model:<28}{u.calls:>9}{u.prompt_tokens:>11}{u.completion_tokens:>10}{c}"
            )
            if u.fehler:
                zeilen.append(f"{'  ↳ ohne usage-Feld':<28}{u.fehler:>9}")
        eur, unbekannt = self.kosten_eur()
        zeilen.append(f"{'SUMME':<28}{'':>30}{eur:>9.4f}")
        if unbekannt:
            zeilen.append(f"⚠️  Kein Preis hinterlegt für: {', '.join(unbekannt)}")
        return "\n".join(zeilen)

    def als_dict(self) -> dict[str, Any]:
        eur, unbekannt = self.kosten_eur()
        return {
            "kosten_eur": round(eur, 6),
            "modelle_ohne_preis": unbekannt,
            "je_modell": {
                m: {
                    "calls": u.calls,
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "ohne_usage_feld": u.fehler,
                }
                for m, u in self.je_modell.items()
            },
        }


# Ein Zähler für den ganzen Prozess. Die Adapter melden hierhin, das Skript liest aus.
TRACKER = UsageTracker()


def track(client, tracker: UsageTracker | None = None):
    """Patcht client.chat.completions.create (synchron) und gibt den Client zurück.

    Muss vor instructor.from_openai(client) aufgerufen werden, sonst werden die strukturierten Aufrufe nicht gezählt.
    Ein zweiter Aufruf auf demselben Client ändert nichts (Marker _usage_tracked).
    """
    t = tracker or TRACKER
    comp = client.chat.completions
    if getattr(comp, "_usage_tracked", False):
        return client
    orig = comp.create

    def create(*args, **kwargs):
        resp = orig(*args, **kwargs)
        t.record(
            kwargs.get("model") or "unbekannt",
            getattr(resp, "usage", None),
            getattr(resp, "model", None),
        )
        return resp

    comp.create = create
    comp._usage_tracked = True
    return client


def track_async(client, tracker: UsageTracker | None = None):
    """Wie track(), aber für AsyncOpenAI (ragas)."""
    t = tracker or TRACKER
    comp = client.chat.completions
    if getattr(comp, "_usage_tracked", False):
        return client
    orig = comp.create

    async def create(*args, **kwargs):
        resp = await orig(*args, **kwargs)
        t.record(
            kwargs.get("model") or "unbekannt",
            getattr(resp, "usage", None),
            getattr(resp, "model", None),
        )
        return resp

    comp.create = create
    comp._usage_tracked = True
    return client


def protokollieren(
    pfad: str | Path, meta: dict[str, Any], tracker: UsageTracker | None = None
) -> float:
    """Hängt den Verbrauch des Laufs an pfad an und gibt die Kosten zurück.

    Jede Zeile der Datei ist ein Lauf, die Summe ist der bisherige Verbrauch.
    """
    t = tracker or TRACKER
    p = Path(pfad)
    p.parent.mkdir(parents=True, exist_ok=True)
    satz = {
        "zeitpunkt": datetime.now().isoformat(timespec="seconds"),
        **meta,
        **t.als_dict(),
    }
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(satz, ensure_ascii=False) + "\n")
    return satz["kosten_eur"]


def bisher_ausgegeben(pfad: str | Path) -> float:
    """Summe aller Läufe in EUR, 0.0 wenn es die Datei nicht gibt."""
    p = Path(pfad)
    if not p.exists():
        return 0.0
    summe = 0.0
    for zeile in p.read_text(encoding="utf-8").splitlines():
        if zeile.strip():
            try:
                summe += float(json.loads(zeile).get("kosten_eur", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning("Unlesbare Zeile in %s übersprungen", p)
    return summe


# Hochrechnung: Kosten eines geplanten Laufs aus gemessenen Läufen schätzen.
# Grundlage sind die Zeilen in results/usage.jsonl, jede mit den Tokens eines echten Laufs.


@dataclass
class ProSample:
    """Gemessener Verbrauch eines Frameworks je Sample."""

    framework: str
    calls: float
    prompt_tokens: float
    completion_tokens: float
    gemessen_mit: str  # Judge der Messung
    basis_samples: int  # Zahl der Samples der Messung

    def kosten_eur(self, model: str) -> float | None:
        """Kosten je Sample, wenn dieselben Tokens über model liefen."""
        p = _price_for(model)
        if p is None:
            return None
        usd = (self.prompt_tokens * p[0] + self.completion_tokens * p[1]) / 1e6
        return usd * USD_TO_EUR


def messungen_je_framework(pfad: str | Path) -> dict[str, ProSample]:
    """Liest usage.jsonl und rechnet je Framework auf ein Sample um.

    Gibt es mehrere Messungen eines Frameworks, zählt die letzte.
    Zeilen mit mehreren Frameworks werden übersprungen, weil sich der Verbrauch dann nicht zuordnen lässt.
    """
    p = Path(pfad)
    if not p.exists():
        return {}

    ergebnis: dict[str, ProSample] = {}
    for zeile in p.read_text(encoding="utf-8").splitlines():
        if not zeile.strip():
            continue
        try:
            satz = json.loads(zeile)
        except json.JSONDecodeError:
            logger.warning("Unlesbare Zeile in %s übersprungen", p)
            continue

        fws = satz.get("frameworks") or []
        n = satz.get("sample_size") or 0
        modelle = satz.get("je_modell") or {}
        if len(fws) != 1 or not n or not modelle:
            continue

        prompt = sum(m.get("prompt_tokens", 0) for m in modelle.values())
        completion = sum(m.get("completion_tokens", 0) for m in modelle.values())
        calls = sum(m.get("calls", 0) for m in modelle.values())
        ergebnis[fws[0]] = ProSample(
            framework=fws[0],
            calls=calls / n,
            prompt_tokens=prompt / n,
            completion_tokens=completion / n,
            gemessen_mit=", ".join(sorted(modelle)),
            basis_samples=n,
        )
    return ergebnis


def top_k_faktor(top_k: int, basis: int, kontextanteil: float) -> float:
    """Faktor für die Tokenmenge bei einem anderen top_k.

    Eine Schätzung, keine Messung.
    Context Relevance wächst mit top_k, weil jeder Chunk bewertet wird, Faithfulness und Answer Relevance kaum.
    kontextanteil ist der Anteil der Tokens, der mit top_k wächst, der Rest bleibt gleich.
    """
    if basis <= 0:
        raise ValueError("basis (top_k der Messung) muss > 0 sein")
    return (1.0 - kontextanteil) + kontextanteil * (top_k / basis)


def hochrechnen(
    messungen: dict[str, ProSample],
    model: str,
    n_samples: int,
    n_datasets: int,
    top_k_werte: list[int],
    top_k_basis: int,
    kontextanteil: float = 0.5,
) -> dict[str, Any]:
    """Kosten eines geplanten Laufs in EUR, aufgeschlüsselt.

    Rückgabe mit gesamt, je_framework, je_top_k und ohne_preis. Frameworks ohne Preis fehlen in der Summe.
    """
    je_fw: dict[str, float] = {}
    je_top_k: dict[int, float] = {}
    ohne_preis: list[str] = []

    faktoren = {k: top_k_faktor(k, top_k_basis, kontextanteil) for k in top_k_werte}

    for fw, pro in sorted(messungen.items()):
        basis = pro.kosten_eur(model)
        if basis is None:
            ohne_preis.append(fw)
            continue
        je_fw[fw] = basis * n_samples * n_datasets * sum(faktoren.values())

    for k, f in faktoren.items():
        je_top_k[k] = sum(
            (pro.kosten_eur(model) or 0.0) * n_samples * n_datasets * f
            for pro in messungen.values()
        )

    return {
        "gesamt": sum(je_fw.values()),
        "je_framework": je_fw,
        "je_top_k": je_top_k,
        "faktoren": faktoren,
        "ohne_preis": ohne_preis,
        "gemessen_mit": sorted({p.gemessen_mit for p in messungen.values()}),
    }


def hat_preis(model: str) -> bool:
    """Ist für dieses Modell ein Preis hinterlegt?

    Kostenlose Modelle haben keinen, eine Hochrechnung ergäbe dann 0 €.
    """
    return _price_for(model) is not None
