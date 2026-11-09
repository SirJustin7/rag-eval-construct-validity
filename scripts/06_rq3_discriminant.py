"""RQ3: Diskriminante Validität:

Prüft, ob die Metriken die Konstrukte trennen, die sie messen sollen.
Alle Metriken werden miteinander korreliert und die Matrix nach Campbell und Fiske (1959) in drei Blöcke zerlegt:

    monotrait-heteromethod    gleiches Konstrukt, anderes Framework  -> soll hoch sein
    heterotrait-monomethod    anderes Konstrukt, gleiches Framework  -> soll niedrig sein
    heterotrait-heteromethod  anderes Konstrukt, anderes Framework   -> soll niedrig sein

Diskriminante Validität liegt vor, wenn der konvergente Block über den beiden anderen liegt.
Dazu kommt eine explorative PCA über alle Metriken.

Das Skript erzeugt keine neuen Scores und kostet nichts.
Es liest dieselben Dateien wie scripts/04.
Alle Score-Zeilen müssen vom selben Judge stammen, standardmäßig judge.model aus config.yaml.
Gemischte Judges werden abgelehnt.

Aufrufe:
    # Hauptlauf, Judge aus config.yaml
    python scripts/06_rq3_discriminant.py --datasets crag ragbench --stem discriminant_main

    # Judge-Wiederholung mit einem anderen Judge
    python scripts/06_rq3_discriminant.py --datasets ragbench --top-k 10 \\
        --expect-judge "meta-llama/Llama-3.3-70B-Instruct" --stem discriminant_llama
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from rag_eval.analysis.discriminant import (
    BINAER_BLOCK,
    HETERO_HETERO,
    HETERO_MONO,
    MONO_HETERO,
    JudgeMismatch,
    bewertbare_paare,
    block_summary,
    correlation_square,
    correlation_table,
    dominant_construct,
    metric_matrix,
    negative_convergent,
    ordering_holds,
    pca_report,
    variance_check,
    verify_judge,
    violations,
    write_discriminant_report,
)
from rag_eval.analysis.plots import mtmm_block_distribution, mtmm_heatmap
from rag_eval.analysis.saturation import formatiere
from rag_eval.analysis.stichprobe import VARIANTEN, filtere, lade_datensatzlaeufe
from rag_eval.config import load_config
from rag_eval.naming import ist_importiert, waehle_score_dateien


def collect_scores(
    scores_dir: Path,
    datasets: list[str],
    files: list[str] | None,
    top_k: int | None = None,
    scope: str | None = None,
    judge: str | None = None,
):
    """Sucht die Score-Dateien nach Datensatz oder nimmt die angegebenen.

    Unterordner von results/scores werden nicht gelesen.
    """
    if files:
        paths = [Path(f) for f in files]
    else:
        # Nur eine Tiefe und ein Zuschnitt, sonst bricht waehle_score_dateien ab.
        paths = waehle_score_dateien(scores_dir, datasets, top_k, scope, judge)
    if not paths:
        sys.exit(
            f"Keine Score-Dateien gefunden in {scores_dir} für {datasets}. "
            "Erst scripts/03_run_frameworks.py laufen lassen."
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["crag", "ragbench", "ragbench_own"],
        help="Datensatz-Präfixe der Score-Dateien (Default: RQ2-Datensätze). "
        "ragbench_own = RAGBenchs eigene Tripel mit TRACe (11 Metriken).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explizite Score-Dateien; überschreibt --datasets",
    )
    parser.add_argument(
        "--scores-dir",
        default=None,
        help="Abweichendes Score-Verzeichnis (Default: <results>/scores)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Abweichendes Ausgabeverzeichnis (Default: <results>/rq4)",
    )
    parser.add_argument(
        "--stem", default="discriminant", help="Anfang der Dateinamen der Ausgaben"
    )
    parser.add_argument(
        "--expect-judge",
        default=None,
        help="Erwartetes Judge-Modell; Default ist judge.model aus "
        "config.yaml. Abweichende oder gemischte Läufe werden "
        "abgelehnt — dieser Schalter benennt nur ein ANDERES "
        "einzelnes Modell (Judge-Wiederholung).",
    )
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Zusaetzlich eine gemeinsame Matrix ueber ALLE Datensaetze. "
        "Standard ist getrennt: Gepoolt verschwinden Unterschiede zwischen "
        "den Datensaetzen, und die Saettigungseinstufung faellt milder aus "
        "als sie es je Datensatz ist.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval-Tiefe der auszuwertenden Laeufe (5 oder 10). "
        "Pflicht, sobald mehrere Tiefen bewertet sind.",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help="Index-Zuschnitt (per_domain). Pflicht bei mehreren.",
    )
    parser.add_argument(
        "--abstinenz",
        choices=sorted(VARIANTEN),
        default="beide",
        help="Kapitel 4.2.5: jede Matrix mit UND ohne Abstinenzen. "
        "Leere Antworten fallen in jeder Variante heraus.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Paarweiser statt listenweiser Fallausschluss "
        "(mehr Daten, aber je Zelle ein anderes n)",
    )
    args = parser.parse_args()

    cfg = load_config()
    results_root = Path(cfg["paths"]["results"])
    scores_dir = Path(args.scores_dir) if args.scores_dir else results_root / "scores"
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "rq3"

    # Nur Dateien des erwarteten Judges. Die Judge-Wiederholungen liegen im selben Ordner, verify_judge würde die Matrix sonst ablehnen.
    paths = collect_scores(
        scores_dir,
        args.datasets,
        args.files,
        args.top_k,
        args.scope,
        judge=args.expect_judge or cfg["judge"]["model"],
    )
    print(f"Gelesen: {len(paths)} Datei(en)")
    for p in paths:
        print(f"  · {p}")

    # Je Datensatz: doppelte Scores entfernen wie in scripts/04, Antworttypen aus der Antwortdatei, bei ragbench_own die TRACe-Labels dazu.
    datensaetze = lade_datensatzlaeufe(
        paths, args.datasets, cfg["paths"]["processed"], cfg
    )

    for variante in VARIANTEN[args.abstinenz]:
        # Getrennt je Datensatz. Gepoolt fällt die Sättigung milder aus und Unterschiede zwischen den Datensätzen verschwinden.
        laeufe = []
        for d in datensaetze:
            teil, bericht = filtere(d.results, d.typen, variante)
            laeufe.append((d.name, teil, bericht, d.binaere_metriken))
        if args.pooled:
            retrieval = [
                lauf
                for lauf, d in zip(laeufe, datensaetze)
                if not ist_importiert(d.dataset, cfg)
            ]
            if retrieval:
                name = "gepoolt" + (f"_k{args.top_k}" if args.top_k else "")
                laeufe.append(
                    (
                        name,
                        [r for lauf in retrieval for r in lauf[1]],
                        None,
                        frozenset(),
                    )
                )

        for name, teil, bericht, binaer in laeufe:
            titel = f"{name} / {variante}"
            if not teil:
                print(f"\n[{titel}] keine Scores gefunden — uebersprungen.")
                continue
            print(f"\n{'=' * 62}\n=== {titel.upper()} — {len(teil)} Scores\n{'=' * 62}")
            if bericht:
                print(f"Stichprobe {bericht.zeile()}")
            _auswerten(
                teil,
                cfg,
                out_dir,
                f"{args.stem}_{name}_{variante}",
                args,
                binaer,
                bericht,
            )


def _auswerten(
    results, cfg, out_dir, stem, args, binaer=frozenset(), bericht=None
) -> None:
    matrix, info = metric_matrix(results, complete_cases=not args.pairwise)

    # Judge prüfen, bevor Zahlen ausgegeben werden.
    expected = args.expect_judge or cfg["judge"]["model"]
    try:
        verify_judge(info, expected)
    except JudgeMismatch as err:
        sys.exit(f"\nAbbruch — Judge-Prüfung fehlgeschlagen:\n{err}")

    print(f"\nMatrix: {info.n_samples} Samples × {info.n_metrics} Metriken")
    print(
        f"Judge-Modell: {info.judge_models[0]}"
        + (
            "  (explizit erwartet)"
            if args.expect_judge
            else "  (aus config judge.model)"
        )
    )
    print(f"Datensätze:   {', '.join(info.datasets)}")
    if info.dropped_samples:
        print(
            f"⚠  {info.dropped_samples} Sample(s) ausgeschlossen (nicht von allen "
            f"Metriken bewertet — z.B. Timeout im Lauf)"
        )

    # Sättigung zuerst
    varianz = variance_check(matrix)
    gesaettigt = varianz[varianz["gesaettigt"]]
    print("\n=== Sättigung je Metrik (vorab festgelegte Regel) ===")
    print(formatiere(varianz.rename(columns={"metric": "metrik"})))
    if not gesaettigt.empty:
        print("\n⚠  Gesättigte Metriken (Deckeneffekt) — Korrelationen dieser Metriken")
        print("   sind in JEDEM Block instabil und dürfen nicht als Beleg gelten.")
        print("   Sie bleiben in der Matrix sichtbar, gehen aber nicht in die")
        print("   Blockmediane und in keine Validitätsaussage ein:")
        for row in gesaettigt.itertuples():
            print(
                f"   · {row.metric}: {row.distinkte_werte} distinkte Werte, "
                f"{row.anteil_am_modus:.0%} auf {row.modus:g}, "
                f"u={row.ungebundener_paaranteil:.3f}"
            )

    # MTMM-Matrix. Ausgeschlossene Metriken bleiben in Matrix und Paartabelle sichtbar, gehen aber in keinen Blockmedian und keinen Verstoß ein.
    # Binäre Metriken (TRACe adherence) werden rang-biserial korreliert und getrennt berichtet.
    ausgeschlossen = set(gesaettigt["metric"])
    binaer = {m for m in matrix.columns if m in binaer}
    table = correlation_table(matrix, binaer)
    square = correlation_square(table, list(matrix.columns))
    summary = block_summary(table, ausgeschlossen)

    print("\n=== Blöcke (|ρ|, Spearman) ===")
    if ausgeschlossen:
        print(
            f"   ohne Paare mit ausgeschlossenen Metriken: {', '.join(sorted(ausgeschlossen))}"
        )
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    gesondert = table[table["block"] == BINAER_BLOCK]
    if not gesondert.empty:
        print(
            "\n=== Binäre Metrik, gesondert (rang-biserial nach Cureton, Mann-Whitney-p) ==="
        )
        print(
            gesondert[["metric_a", "metric_b", "n", "r_rb", "r_rb_p"]].to_string(
                index=False, float_format=lambda v: f"{v:.3g}"
            )
        )

    # Negative konvergente Korrelationen zählen im Median über |ρ| wie starke Konvergenz, sind aber das Gegenteil.
    neg_conv = negative_convergent(table)
    if not neg_conv.empty:
        print(
            f"\n⚠  {len(neg_conv)} konvergente(s) Paar(e) mit NEGATIVEM ρ — diese gehen"
        )
        print("   als Betrag in den Blockmedian ein und lassen ihn zu gut aussehen:")
        for row in neg_conv.itertuples():
            print(f"   · {row.metric_a} ↔ {row.metric_b}: ρ = {row.spearman_rho:.3f}")

    if ordering_holds(summary):
        print("\n✓ Erwartete Ordnung erfüllt: konvergenter Block über beiden anderen.")
        if not neg_conv.empty:
            print(
                "   ABER: siehe negative konvergente Korrelationen oben — die Ordnung"
            )
            print("   allein trägt die Aussage dann nicht.")
    else:
        med = summary.set_index("block")["median"]
        print(
            f"\n✗ Erwartete Ordnung NICHT erfüllt "
            f"(konvergent {med.get(MONO_HETERO, float('nan')):.3f} vs. "
            f"{med.get(HETERO_MONO, float('nan')):.3f} / "
            f"{med.get(HETERO_HETERO, float('nan')):.3f})."
        )

    # Einzelne Verstöße
    viol = violations(table, ausgeschlossen)
    if viol.empty:
        print("\nKeine Verstöße nach dem Zeilen-/Spaltenkriterium.")
    else:
        betroffen = viol["validitaets_paar"].nunique()
        print(
            f"\n=== Verstöße: {len(viol)} (betreffen {betroffen} konvergente Paare) ==="
        )
        print(viol.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if len(viol) > 10:
            print(f"   … {len(viol) - 10} weitere in der Arbeitsmappe")

    # PCA (explorativ), ohne die binäre Metrik
    pca_variance, pca_loadings, pca_warnings = pca_report(
        matrix.drop(columns=sorted(binaer))
    )
    print("\n=== PCA (Spearman, Parallelanalyse 999 Permutationen, 95. Perzentil) ===")
    for w in pca_warnings:
        print(f"⚠  {w}")
    print(
        pca_variance.head(4).to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )
    print(f"   behaltene Komponenten: {int(pca_variance['behalten'].sum())}")
    for comp in pca_variance.loc[pca_variance["behalten"], "komponente"]:
        print(f"   {comp} dominiert von: {dominant_construct(pca_loadings, comp)}")

    # Export
    csv_path, xlsx_path = write_discriminant_report(
        out_dir,
        info,
        square,
        table,
        summary,
        viol,
        neg_conv,
        varianz,
        pca_variance,
        pca_loadings,
        stem=stem,
        stichprobe=None
        if bericht is None
        else pd.DataFrame(
            [
                ("variante", bericht.variante),
                ("antworten", bericht.n_antworten),
                ("leer_entfernt", bericht.n_leer),
                ("abstinenzen", bericht.n_abstinenz),
                ("abstinenzen_entfernt", bericht.n_abstinenz_entfernt),
                (
                    "ausgeschlossene_metriken",
                    ", ".join(sorted(ausgeschlossen)) or "keine",
                ),
            ],
            columns=["feld", "wert"],
        ),
    )
    heat = mtmm_heatmap(square, table, out_dir / f"{stem}_mtmm_heatmap.png")
    # Dieselben Paare wie in den Blockmedianen.
    dist = mtmm_block_distribution(
        table[bewertbare_paare(table, ausgeschlossen)], out_dir / f"{stem}_blocks.png"
    )

    print(f"\nGeschrieben:\n  {csv_path}\n  {xlsx_path}\n  {heat}\n  {dist}")


if __name__ == "__main__":
    main()
