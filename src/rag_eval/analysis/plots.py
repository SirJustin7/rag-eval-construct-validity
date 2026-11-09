"""Abbildungen zu den Ergebnissen:

- convergence_profile:                          paarweise ρ je Konstrukt (RQ2)
- mtmm_heatmap und mtmm_block_distribution:     MTMM-Matrix und Blöcke (RQ3)
- criterion_distribution:                       Scores je Urteil richtig oder falsch (RQ4)

"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # ohne Bildschirm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Dieselben Blockfarben wie in der MTMM-Abbildung der Arbeit.
BLOCK_COLORS = {
    "monotrait-heteromethod": "#2e7d5b",
    "heterotrait-monomethod": "#c26a27",
    "heterotrait-heteromethod": "#7d9ec0",
}
# Englisch, weil die Abbildungen in einer englischen Arbeit stehen.
BLOCK_SHORT = {
    "monotrait-heteromethod": "convergent",
    "heterotrait-monomethod": "same framework",
    "heterotrait-heteromethod": "neither",
}

_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 120,
}


def correlation_heatmap(corr_table: pd.DataFrame, construct: str, out: Path) -> None:
    """Nicht umgesetzt, convergence_profile zeigt dieselben Werte."""
    raise NotImplementedError


def criterion_scatter(merged: pd.DataFrame, out: Path) -> None:
    """Nicht umgesetzt, das Kriterium in RQ4 ist binär."""
    raise NotImplementedError


def criterion_distribution(
    scores: pd.DataFrame, out: Path, koeffizienten: pd.DataFrame | None = None
) -> Path:
    """Scoreverteilung je Urteil richtig oder falsch (RQ4).

    Ein Feld je Metrik mit zwei Verteilungen, richtige und falsche Urteile.
    Eine Metrik, die als Proxy taugt, sollte die beiden trennen.

    Boxplots statt Streudiagramm, weil das Kriterium nur zwei Werte hat. Die einzelnen Werte stehen als Punkte daneben, weil viele Scores gleich sind und ein Boxplot allein das verdeckt.

    scores:            Spalten sample_id, metric, score, richtig (bool).
    koeffizienten:     optional die Tabelle aus criterion.kriteriumstabelle, dann steht r in jedem Feld (Zeilen mit gruppe == "alle").
    """
    metriken = [
        f"{fw}.{k}"
        for k in CONSTRUCT_ORDER
        for fw in ("deepeval", "ragas", "trulens")
        if f"{fw}.{k}" in set(scores["metric"])
    ]
    r_je_metrik = {}
    if koeffizienten is not None and "gruppe" in koeffizienten:
        alle = koeffizienten[koeffizienten["gruppe"] == "alle"]
        r_je_metrik = dict(zip(alle["metric"], alle["r_rb"]))

    spalten = 3
    zeilen = (len(metriken) + spalten - 1) // spalten
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(
            zeilen, spalten, figsize=(6.9, 2.1 * zeilen), sharey=True
        )
        achsen = axes.ravel() if hasattr(axes, "ravel") else [axes]
        rng = np.random.default_rng(42)  # fester Seed für den Jitter

        for ax, metrik in zip(achsen, metriken):
            teil = scores[scores["metric"] == metrik]
            gruppen = [
                teil.loc[~teil["richtig"], "score"].dropna(),
                teil.loc[teil["richtig"], "score"].dropna(),
            ]
            konstrukt = metrik.split(".", 1)[1]
            farbe = CONSTRUCT_COLORS.get(konstrukt, "#555555")

            bp = ax.boxplot(
                gruppen,
                widths=0.5,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "black"},
            )
            for kasten, deckung in zip(bp["boxes"], (0.25, 0.6)):
                kasten.set_facecolor(farbe)
                kasten.set_alpha(deckung)
                kasten.set_edgecolor(farbe)

            for i, werte in enumerate(gruppen, start=1):
                x = rng.normal(i, 0.045, size=len(werte))
                ax.plot(x, werte, ".", color=farbe, alpha=0.28, markersize=2.4)

            ax.set_xticks([1, 2])
            ax.set_xticklabels(
                [
                    f"incorrect\n(n = {len(gruppen[0])})",
                    f"correct\n(n = {len(gruppen[1])})",
                ],
                fontsize=8,
            )
            titel = f"{FRAMEWORK_LABEL.get(metrik.split('.')[0], metrik)}"
            r = r_je_metrik.get(metrik)
            if r is not None and not pd.isna(r):
                titel += f"   r = {r:+.2f}"
            ax.set_title(titel, fontsize=8.5, pad=4)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="y", color="#dddddd", linewidth=0.6)
            ax.set_axisbelow(True)

        for ax in achsen[len(metriken) :]:
            ax.axis("off")
        # Konstruktname am Anfang jeder Zeile
        for i in range(zeilen):
            konstrukt = metriken[i * spalten].split(".", 1)[1]
            achsen[i * spalten].set_ylabel(
                CONSTRUCT_LABEL.get(konstrukt, konstrukt), fontsize=9
            )
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


def _short(metric: str) -> str:
    """'ragas/context_relevance' -> 'ragas\\nCR' (Achsenbeschriftung)."""
    framework, construct = metric.split("/", 1)
    kurz = {"context_relevance": "CR", "faithfulness": "F", "answer_relevance": "AR"}
    return f"{framework}\n{kurz.get(construct, construct)}"


def mtmm_heatmap(square: pd.DataFrame, table: pd.DataFrame, out: Path) -> Path:
    """MTMM-Matrix mit Zahlenwerten, Zellrahmen in der Blockfarbe.

    Die Füllung zeigt |ρ|, der Rahmen den Block. So ist zu sehen, ob die dunklen Zellen die konvergenten sind.
    """
    metrics = list(square.index)
    n = len(metrics)
    block_lookup = {
        (row.metric_a, row.metric_b): row.block for row in table.itertuples()
    }
    block_lookup.update({(b, a): v for (a, b), v in block_lookup.items()})

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7.4, 6.6))
        cmap = plt.get_cmap("Greys")

        for i, row_metric in enumerate(metrics):
            for j, col_metric in enumerate(metrics):
                if j > i:
                    continue
                y = n - 1 - i
                if i == j:
                    ax.add_patch(
                        Rectangle(
                            (j, y),
                            1,
                            1,
                            facecolor="#e8e8e8",
                            edgecolor="white",
                            linewidth=1.0,
                        )
                    )
                    continue
                rho = square.loc[row_metric, col_metric]
                block = block_lookup.get((row_metric, col_metric))
                shade = 0.0 if pd.isna(rho) else min(abs(rho), 1.0)
                ax.add_patch(
                    Rectangle(
                        (j, y),
                        1,
                        1,
                        facecolor=cmap(0.10 + 0.62 * shade),
                        edgecolor=BLOCK_COLORS.get(block, "#cccccc"),
                        linewidth=2.4,
                        zorder=2,
                    )
                )
                if pd.notna(rho):
                    ax.text(
                        j + 0.5,
                        y + 0.5,
                        f"{rho:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        zorder=3,
                        color="white" if shade > 0.55 else "#222222",
                    )

        ax.set_xticks([k + 0.5 for k in range(n)])
        ax.set_xticklabels([_short(m) for m in metrics], fontsize=7.5)
        ax.set_yticks([n - 1 - k + 0.5 for k in range(n)])
        ax.set_yticklabels([_short(m) for m in metrics], fontsize=7.5)
        ax.tick_params(length=0)
        ax.set_xlim(-0.1, n + 0.1)
        ax.set_ylim(-0.1, n + 0.1)
        ax.set_aspect("equal")
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=11,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=2.2,
                label=f"{BLOCK_SHORT[name]} ({name})",
            )
            for name, color in BLOCK_COLORS.items()
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.34, 0.99),
            frameon=False,
            fontsize=8,
            labelspacing=0.5,
        )
        ax.text(
            0.34,
            0.66,
            "Fill: |ρ| (darker = stronger)\nBorder: block type",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
            va="top",
        )

        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return out


def mtmm_block_distribution(table: pd.DataFrame, out: Path) -> Path:
    """Verteilung von |ρ| je Block (RQ3).

    Zeigt, ob der konvergente Block über den beiden anderen liegt.
    Einzelne Punkte statt Balken, weil bei wenigen Werten je Block der Median allein die Streuung und die Ausreißer verdeckt.
    """
    from ..analysis.discriminant import BLOCK_ORDER

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        rng = np.random.default_rng(42)

        for idx, block in enumerate(BLOCK_ORDER):
            vals = table.loc[table["block"] == block, "abs_rho"].dropna()
            if vals.empty:
                continue
            jitter = rng.uniform(-0.16, 0.16, size=len(vals))
            ax.scatter(
                vals,
                [idx] * len(vals) + jitter,
                s=42,
                alpha=0.75,
                color=BLOCK_COLORS[block],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            med = vals.median()
            ax.plot(
                [med, med],
                [idx - 0.30, idx + 0.30],
                color="#222222",
                linewidth=2.0,
                zorder=4,
            )
            ax.text(
                med,
                idx + 0.40,
                f"Median {med:.2f}",
                ha="center",
                fontsize=8,
                color="#222222",
            )

        ax.set_yticks(range(len(BLOCK_ORDER)))
        ax.set_yticklabels([f"{BLOCK_SHORT[b]}\n{b}" for b in BLOCK_ORDER], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("|ρ| (Spearman)")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="x", color="#e0e0e0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return out


# Konstruktfarben, andere als BLOCK_COLORS, weil Farbe hier das Konstrukt zeigt und nicht den MTMM-Block
CONSTRUCT_COLORS = {
    "answer_relevance": "#2a78d6",
    "context_relevance": "#eb6834",
    "faithfulness": "#1baf7a",
}
# Von oben nach unten abnehmende Konvergenz.
CONSTRUCT_ORDER = ["answer_relevance", "context_relevance", "faithfulness"]
CONSTRUCT_LABEL = {
    "answer_relevance": "Answer Relevance",
    "context_relevance": "Context Relevance",
    "faithfulness": "Faithfulness",
}
FRAMEWORK_LABEL = {"deepeval": "DeepEval", "ragas": "Ragas", "trulens": "TruLens"}
# Datensatz -> (Markerform, gefüllt?). Farbe zeigt das Konstrukt, die Form den Datensatz.
DATASET_MARKER = {"CRAG": ("o", True), "RAGBench": ("s", False)}


def convergence_profile(tabellen: dict[str, pd.DataFrame], out: Path) -> Path:
    """Konvergenzprofil (RQ2): alle paarweisen ρ, nach Konstrukt gruppiert.

    Keine Heatmap, weil es je Konstrukt nur drei Paare gibt und eine Matrix die Unterschiede zwischen den Konstrukten schlechter zeigt.
    Beide Datensätze stehen im selben Feld, damit sie direkt vergleichbar sind.

    tabellen: {"CRAG": df, "RAGBench": df} mit den Spalten aus results/rq2/convergence_*.csv.
    """
    paare = [("deepeval", "ragas"), ("deepeval", "trulens"), ("ragas", "trulens")]
    paar_label = [f"{FRAMEWORK_LABEL[a]} ↔ {FRAMEWORK_LABEL[b]}" for a, b in paare]

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(
            len(CONSTRUCT_ORDER),
            1,
            figsize=(6.9, 5.6),
            sharex=True,
            gridspec_kw={"hspace": 0.46},
        )

        for ax, konstrukt in zip(axes, CONSTRUCT_ORDER):
            farbe = CONSTRUCT_COLORS[konstrukt]
            alle_rho = []

            for i, (a, b) in enumerate(paare):
                y = len(paare) - 1 - i  # erstes Paar oben
                punkte = []
                for ds, df in tabellen.items():
                    treffer = df[
                        (df["construct"] == konstrukt)
                        & (df["framework_a"] == a)
                        & (df["framework_b"] == b)
                    ]
                    if treffer.empty:
                        continue
                    rho = float(treffer["spearman_rho"].iloc[0])
                    punkte.append((ds, rho))
                    alle_rho.append(rho)

                # Linie zwischen den Werten der beiden Datensätze
                if len(punkte) == 2:
                    ax.plot(
                        [punkte[0][1], punkte[1][1]],
                        [y, y],
                        color=farbe,
                        linewidth=2.0,
                        alpha=0.30,
                        solid_capstyle="round",
                        zorder=2,
                    )
                for ds, rho in punkte:
                    form, gefuellt = DATASET_MARKER[ds]
                    ax.scatter(
                        rho,
                        y,
                        marker=form,
                        s=64,
                        zorder=3,
                        color=farbe if gefuellt else "#ffffff",
                        edgecolor=farbe,
                        linewidth=1.7,
                    )

            _konstrukt_panel(ax, konstrukt, paar_label, alle_rho, farbe)

        axes[-1].set_xlabel("Spearman ρ between two frameworks", labelpad=8)
        _profil_legende(fig, axes[0])
        fig.subplots_adjust(left=0.235, right=0.985, top=0.90, bottom=0.10)
        out.parent.mkdir(parents=True, exist_ok=True)
        for pfad in (out, out.with_suffix(".pdf")):
            fig.savefig(pfad, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return out


def _konstrukt_panel(ax, konstrukt, paar_label, alle_rho, farbe) -> None:
    """Achsen, Raster und Median für das Feld eines Konstrukts."""
    ax.set_yticks(range(len(paar_label)))
    ax.set_yticklabels(list(reversed(paar_label)), fontsize=8.5)
    ax.set_ylim(-0.55, 2.95)
    ax.set_xlim(-0.2, 1.0)
    ax.set_xticks(np.arange(-0.2, 1.01, 0.2))

    ax.grid(axis="x", color="#ececec", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0.0, color="#9a9a9a", linewidth=0.9, zorder=1)  # Nullkorrelation
    for seite in ("top", "right", "left", "bottom"):
        ax.spines[seite].set_visible(False)
    ax.tick_params(axis="both", length=0)

    # Titel in schwarz, die Farbe steht im Kästchen davor.
    ax.scatter(
        [0.0],
        [1.055],
        marker="s",
        s=42,
        color=farbe,
        clip_on=False,
        transform=ax.transAxes,
        zorder=5,
    )
    ax.text(
        0.022,
        1.055,
        CONSTRUCT_LABEL[konstrukt],
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#222222",
    )

    # Nur der Median als Zahl, die Einzelwerte stehen in der Tabelle.
    if alle_rho:
        med = float(np.median(alle_rho))
        ax.plot([med, med], [-0.42, 2.42], color="#555555", linewidth=1.2, zorder=4)
        # Rechts oben, an der Linie überdeckt die Zahl bei kleinem Median den Titel.
        ax.text(
            0.995,
            0.93,
            f"Median ρ = {med:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555555",
        )


def _profil_legende(fig, ax) -> None:
    """Legende für die Datensätze (Form = Datensatz, Farbe = Konstrukt)."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor="#555555",
            markeredgecolor="#555555",
            label="CRAG",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=7,
            markerfacecolor="#ffffff",
            markeredgecolor="#555555",
            markeredgewidth=1.7,
            label="RAGBench",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.14),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.4,
        columnspacing=1.4,
    )
