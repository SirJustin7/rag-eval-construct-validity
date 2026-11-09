"""Schritt 3: Bewertung durch die Frameworks:

Jedes aktivierte Framework bewertet dieselben Tripel, die Scores werden als MetricResult gespeichert (Grundlage für RQ2 bis RQ4).

Aufruf: python scripts/03_run_frameworks.py [--dataset ...] [--framework ...]

Ohne --out-dir schreibt der Lauf nach results/scores/, wo die Dateien der Auswertung liegen.
Der Dateiname enthält Datensatz, Framework, Zuschnitt, Tiefe und Judge (naming.scores_datei), mit und ohne --out-dir gleich.
Für Probeläufe --sample-size und --out-dir setzen:

    python scripts/03_run_frameworks.py --dataset ragbench --framework ragas \
        --sample-size 3 --out-dir results/scores/demo

Checkpointing: Der Lauf schreibt nach je --chunk-size Samples in die Zieldatei und überspringt beim Neustart, was dort schon vollständig bewertet ist.
Mit --restart beginnt er von vorn.

Jeder Judge-Aufruf wird gezählt, der Verbrauch steht am Ende in results/usage.jsonl.
"""

import argparse
from pathlib import Path

from rag_eval import credentials, naming
from rag_eval.config import load_config
from rag_eval.data.io import entdoppele, read_jsonl, write_jsonl
from rag_eval.data.loaders import LOADERS
from rag_eval.data.schema import GeneratedSample, MetricResult
from rag_eval.evaluation import EVALUATORS
from rag_eval.evaluation.usage import (
    TRACKER,
    BudgetUeberschritten,
    bisher_ausgegeben,
    protokollieren,
)

# Nach wie vielen Samples zwischengespeichert wird.
DEFAULT_CHUNK = 10
USAGE_LOG = "results/usage.jsonl"


def _bereits_bewertet(pfad: Path, konstrukte: int) -> set[str]:
    """sample_ids, die in pfad schon vollständig bewertet sind.

    Unvollständige Samples zählen nicht als erledigt.
    Sie laufen erneut, und ihre Teilergebnisse werden vorher entfernt, sonst gäbe es Dubletten.
    """
    if not pfad.exists():
        return set()
    gezaehlt: dict[str, int] = {}
    for r in read_jsonl(str(pfad), MetricResult):
        gezaehlt[r.sample_id] = gezaehlt.get(r.sample_id, 0) + 1
    return {sid for sid, n in gezaehlt.items() if n >= konstrukte}


def _anhaengen(ergebnisse: list[MetricResult], pfad: Path) -> None:
    """Schreibt an eine bestehende JSONL-Datei an (write_jsonl überschreibt)."""
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, "a", encoding="utf-8") as f:
        f.writelines(r.model_dump_json() + "\n" for r in ergebnisse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(LOADERS), default=None)
    parser.add_argument("--framework", choices=list(EVALUATORS), default=None)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Nur die ersten N Samples bewerten (z.B. 3 für einen Schnelltest).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Zielordner für die Score-Dateien. Standard: <results>/scores. Für "
        "Probeläufe einen Unterordner angeben, damit der Hauptlauf nicht "
        "überschrieben wird.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        help=f"Nach je N Samples zwischenspeichern (Standard {DEFAULT_CHUNK}).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge-Modell abweichend von config.yaml. Fuer die "
        "Judge-Wiederholung (andere Modellfamilie).",
    )
    parser.add_argument(
        "--judge-endpoint",
        choices=["judge", "generator"],
        default="judge",
        help="Welche Zugangsdaten der Judge benutzt. 'generator' = SCADS "
        "(kostenlos) fuer Laeufe mit freien Modellen; "
        "Standard 'judge' = Judge-Endpunkt (kostenpflichtig).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Vorhandene Ergebnisse ignorieren und von vorn beginnen.",
    )
    args = parser.parse_args()

    cfg = load_config()
    # Ohne --dataset die aktivierten Datensätze aus der config.
    datasets = (
        [args.dataset]
        if args.dataset
        else [
            name
            for name, ds_cfg in cfg["datasets"].items()
            if (ds_cfg or {}).get("enabled")
        ]
    )
    frameworks = (
        [args.framework]
        if args.framework
        else [
            name for name, fw_cfg in cfg["frameworks"].items() if fw_cfg.get("enabled")
        ]
    )

    # Nicht umgesetzte Adapter vor dem ersten bezahlten Aufruf abfangen.
    fehlend = [fw for fw in frameworks if not EVALUATORS[fw].implemented]
    if fehlend:
        parser.error(
            f"Framework(s) ohne Implementierung: {', '.join(fehlend)}. "
            f"Entweder in config.yaml auf enabled: false setzen oder den Adapter "
            f"fertigstellen. Abbruch VOR dem ersten Judge-Aufruf."
        )
    out_dir = Path(args.out_dir or f"{cfg['paths']['results']}/scores")

    # Zugangsdaten des Judge aktivieren, bevor ein Adapter seinen Client erzeugt, sonst liefe der Judge über den Generator-Endpunkt.
    # Weil der Judge im Dateinamen steht, überschreiben sich Läufe verschiedener Judges nicht.
    if args.judge_model:
        cfg["judge"] = {**cfg["judge"], "model": args.judge_model}
        print(f"[Judge] Modell per --judge-model auf {args.judge_model} gesetzt.")

    judge_url = credentials.aktiviere(args.judge_endpoint)
    if args.judge_endpoint == "generator":
        print(
            "[Judge] laeuft ueber den GENERATOR-Endpunkt (SCADS, kostenlos). "
            "Achtung: Generator und Judge teilen damit den Endpunkt — "
            "verschiedene MODELLE sind Voraussetzung."
        )
    print(f"[Judge] {cfg['judge']['model']} über {judge_url}")
    if not credentials.getrennt_konfiguriert():
        print(
            "⚠️  Generator und Judge nutzen denselben Endpunkt. Fuer den "
            "Hauptlauf getrennte Zugaenge setzen (GENERATOR_* / JUDGE_*), "
            "sonst droht Selbstbevorzugungs-Bias."
        )

    vorher = bisher_ausgegeben(USAGE_LOG)
    if vorher:
        print(f"Bisher laut {USAGE_LOG} verbraucht: {vorher:.4f} €")

    # Budgetgrenze: Der Tracker bricht den Lauf ab, sobald sie erreicht ist.
    limit = (cfg.get("budget") or {}).get("gesamt_eur")
    TRACKER.limit_eur = float(limit) if limit else None
    TRACKER.vorlauf_eur = vorher
    if TRACKER.limit_eur is not None:
        rest = TRACKER.limit_eur - vorher
        print(f"Budgetgrenze: {TRACKER.limit_eur:.2f} € — verbleibend {rest:.4f} €")
        if rest <= 0:
            parser.error(
                f"Budget bereits ausgeschoepft ({vorher:.4f} € von "
                f"{TRACKER.limit_eur:.2f} €). Abbruch VOR dem ersten Judge-Aufruf."
            )
    else:
        print("⚠️  Keine Budgetgrenze gesetzt (config: budget.gesamt_eur fehlt).")
    print()

    for ds in datasets:
        # Antwortdatei zum Zuschnitt und top_k aus der config.
        scope = naming.scope_aus_config(cfg["rag"])
        top_k = naming.top_k_aus_config(cfg["rag"])
        alle = read_jsonl(
            naming.generated_datei(ds, scope, top_k, cfg["paths"]["processed"], cfg),
            GeneratedSample,
        )
        # Die RAGBench-Antworten enthalten 16 doppelte Fragen. Bewertet wird je Frage der erste Eintrag, wie später in der Auswertung.
        alle = entdoppele(alle, f"[{ds}] generierte Samples")
        if args.sample_size is not None:
            alle = alle[: args.sample_size]
            print(f"[{ds}] Probelauf: nur {len(alle)} von den generierten Samples.")

        for fw in frameworks:
            # Mit --out-dir derselbe Dateiname wie ohne, nur in einem anderen Ordner.
            ziel = Path(
                naming.scores_datei(
                    ds,
                    fw,
                    scope,
                    top_k,
                    cfg["judge"]["model"],
                    cfg["paths"]["results"],
                    cfg,
                )
            )
            if args.out_dir:
                ziel = Path(args.out_dir) / ziel.name
            evaluator = EVALUATORS[fw](cfg)
            n_konstrukte = len(evaluator.metric_mapping)

            if args.restart and ziel.exists():
                ziel.unlink()
                print(f"[{ds}/{fw}] --restart: {ziel.name} gelöscht.")

            fertig = _bereits_bewertet(ziel, n_konstrukte)
            offen = [s for s in alle if s.sample_id not in fertig]
            if fertig:
                print(
                    f"[{ds}/{fw}] {len(fertig)} Samples bereits bewertet, "
                    f"{len(offen)} offen."
                )
            if not offen:
                print(f"[{ds}/{fw}] nichts zu tun.")
                continue

            # Teilergebnisse unvollständiger Samples entfernen
            if ziel.exists():
                sauber = [
                    r
                    for r in read_jsonl(str(ziel), MetricResult)
                    if r.sample_id in fertig
                ]
                write_jsonl(sauber, str(ziel))

            geschrieben = len(fertig) * n_konstrukte
            for i in range(0, len(offen), args.chunk_size):
                block = offen[i : i + args.chunk_size]
                res = evaluator.evaluate(block)
                _anhaengen(res, ziel)
                geschrieben += len(res)
                eur, _ = TRACKER.kosten_eur()
                print(
                    f"[{ds}/{fw}] {min(i + len(block), len(offen))}/{len(offen)} "
                    f"Samples · {geschrieben} Scores · bisher {eur:.4f} € in diesem Lauf"
                )

            print(f"[{ds}/{fw}] fertig → {ziel}")

    _abschluss(datasets, frameworks, args, out_dir, vorher)


def _abschluss(datasets, frameworks, args, out_dir, vorher) -> None:
    """Gibt den Verbrauch aus und schreibt ihn in usage.jsonl."""
    print("\n=== Verbrauch dieses Laufs ===")
    print(TRACKER.bericht())
    if TRACKER.aufgeloeste_modelle:
        # Der Endpunkt löst "gpt-4o-mini" auf eine Version mit Datum auf. Nur hier lässt sich festhalten, welche das war.
        print("\nTatsächlich antwortende Modelle (Version mit Datum):")
        for angefragt, snaps in sorted(TRACKER.aufgeloeste_modelle.items()):
            print(f"  {angefragt} → {', '.join(sorted(snaps))}")
    kosten = protokollieren(
        USAGE_LOG,
        {
            "datasets": datasets,
            "frameworks": frameworks,
            "sample_size": args.sample_size,
            "out_dir": str(out_dir),
        },
    )
    gesamt = vorher + kosten
    print(f"\nProtokolliert in {USAGE_LOG}. Gesamtverbrauch bisher: {gesamt:.4f} €")
    if gesamt > 20:
        print("⚠️  Über 20 € — Budget im Blick behalten (20–30 € gesamt).")


if __name__ == "__main__":
    try:
        main()
    except BudgetUeberschritten as stop:
        print(f"\n🛑 {stop}")
        raise SystemExit(2)
