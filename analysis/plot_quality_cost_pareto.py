#!/usr/bin/env python3
"""Generate Figure 08: quality-cost Pareto trade-off across benchmarks.

The y-axis scores are the primary metrics reported in Table 1 of
els-cas-templates/cas-dc-template.tex. The x-axis costs are diagnostic
avg_exec_cost values from the corresponding analysis CSV files.
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATA = """target,method,group,cost,score,metric
MMLU,ParagraphRAG,Fixed retrieval,0.250000,86.06,Accuracy
MMLU,DocumentRAG,Fixed retrieval,0.450000,74.74,Accuracy
MMLU,UniversalRAG-DistilBERT,Universal/adaptive/self,0.007771,88.40,Accuracy
MMLU,UniversalRAG-T5-large,Universal/adaptive/self,0.017429,88.23,Accuracy
MMLU,Adaptive-RAG,Universal/adaptive/self,0.229571,85.94,Accuracy
MMLU,Self-RAG,Universal/adaptive/self,0.082429,87.20,Accuracy
MMLU,Hard-DistilBERT,Hard routing,0.000000,88.63,Accuracy
MMLU,Hard-T5-large,Hard routing,0.017429,87.94,Accuracy
MMLU,COVER-DistilBERT,COVER,0.253886,88.91,Accuracy
MMLU,COVER-T5-large,COVER,0.250686,88.86,Accuracy
SQuAD,ParagraphRAG,Fixed retrieval,0.250000,19.60,F1
SQuAD,DocumentRAG,Fixed retrieval,0.250000,19.60,F1
SQuAD,UniversalRAG-DistilBERT,Universal/adaptive/self,0.268971,19.44,F1
SQuAD,UniversalRAG-T5-large,Universal/adaptive/self,0.268400,19.46,F1
SQuAD,Adaptive-RAG,Universal/adaptive/self,0.249714,19.47,F1
SQuAD,Self-RAG,Universal/adaptive/self,0.244714,19.66,F1
SQuAD,Hard-DistilBERT,Hard routing,0.250000,19.53,F1
SQuAD,Hard-T5-large,Hard routing,0.249429,19.40,F1
SQuAD,COVER-DistilBERT,COVER,0.498857,24.15,F1
SQuAD,COVER-T5-large,COVER,0.499286,19.68,F1
Natural Questions,ParagraphRAG,Fixed retrieval,0.250000,47.09,F1
Natural Questions,DocumentRAG,Fixed retrieval,0.250000,47.09,F1
Natural Questions,UniversalRAG-DistilBERT,Universal/adaptive/self,0.289543,47.21,F1
Natural Questions,UniversalRAG-T5-large,Universal/adaptive/self,0.246514,47.43,F1
Natural Questions,Adaptive-RAG,Universal/adaptive/self,0.249000,47.06,F1
Natural Questions,Self-RAG,Universal/adaptive/self,0.236429,47.56,F1
Natural Questions,Hard-DistilBERT,Hard routing,0.247571,47.16,F1
Natural Questions,Hard-T5-large,Hard routing,0.245714,47.28,F1
Natural Questions,COVER-DistilBERT,COVER,0.299429,55.72,F1
Natural Questions,COVER-T5-large,COVER,0.252857,57.64,F1
HotpotQA,ParagraphRAG,Fixed retrieval,0.250000,19.00,F1
HotpotQA,DocumentRAG,Fixed retrieval,0.450000,43.03,F1
HotpotQA,UniversalRAG-DistilBERT,Universal/adaptive/self,0.427429,41.29,F1
HotpotQA,UniversalRAG-T5-large,Universal/adaptive/self,0.437943,41.73,F1
HotpotQA,Adaptive-RAG,Universal/adaptive/self,0.405257,46.03,F1
HotpotQA,Self-RAG,Universal/adaptive/self,0.435343,43.24,F1
HotpotQA,Hard-DistilBERT,Hard routing,0.428057,41.62,F1
HotpotQA,Hard-T5-large,Hard routing,0.437943,42.13,F1
HotpotQA,COVER-DistilBERT,COVER,0.699571,45.80,F1
HotpotQA,COVER-T5-large,COVER,0.700000,46.12,F1
WebQA,ParagraphRAG,Fixed retrieval,0.250000,39.72,ROUGE-L
WebQA,DocumentRAG,Fixed retrieval,0.450000,37.27,ROUGE-L
WebQA,UniversalRAG-DistilBERT,Universal/adaptive/self,0.594286,41.11,ROUGE-L
WebQA,UniversalRAG-T5-large,Universal/adaptive/self,0.593071,41.14,ROUGE-L
WebQA,Adaptive-RAG,Universal/adaptive/self,0.594286,40.91,ROUGE-L
WebQA,Self-RAG,Universal/adaptive/self,0.594286,41.35,ROUGE-L
WebQA,Hard-DistilBERT,Hard routing,0.584393,41.33,ROUGE-L
WebQA,Hard-T5-large,Hard routing,0.593321,41.33,ROUGE-L
WebQA,COVER-DistilBERT,COVER,0.609214,43.17,ROUGE-L
WebQA,COVER-T5-large,COVER,0.602857,43.42,ROUGE-L
"""


PANELS = [
    ("MMLU", "Accuracy (%)", (-0.015, 0.315), (85.0, 90.0)),
    ("SQuAD", "F1 (%)", (0.225, 0.515), (19.1, 24.6)),
    ("Natural Questions", "F1 (%)", (0.232, 0.305), (46.8, 58.2)),
    ("HotpotQA", "F1 (%)", (0.390, 0.715), (40.5, 46.8)),
    ("WebQA", "ROUGE-L (%)", (0.575, 0.615), (40.5, 43.7)),
]


STYLE = {
    "Fixed retrieval": {
        "marker": "o",
        "facecolor": "#b8b8b8",
        "edgecolor": "#4d4d4d",
        "size": 24,
        "alpha": 0.82,
        "zorder": 3,
    },
    "Universal/adaptive/self": {
        "marker": "^",
        "facecolor": "#4daf4a",
        "edgecolor": "#1b5e20",
        "size": 28,
        "alpha": 0.90,
        "zorder": 4,
    },
    "Hard routing": {
        "marker": "s",
        "facecolor": "#2f68d8",
        "edgecolor": "#17459e",
        "size": 26,
        "alpha": 0.90,
        "zorder": 5,
    },
    "COVER": {
        "marker": "*",
        "facecolor": "#ff3b1f",
        "edgecolor": "black",
        "size": 112,
        "alpha": 0.96,
        "zorder": 7,
    },
}


KEY_COVER_LABEL = {
    "MMLU": "COVER-DistilBERT",
    "SQuAD": "COVER-DistilBERT",
    "Natural Questions": "COVER-T5-large",
    "HotpotQA": "COVER-T5-large",
    "WebQA": "COVER-T5-large",
}


LABEL_OFFSETS = {
    ("MMLU", "COVER-DistilBERT"): (0.010, 0.26),
    ("SQuAD", "COVER-DistilBERT"): (0.010, 0.16),
    ("Natural Questions", "COVER-T5-large"): (0.004, 0.26),
    ("HotpotQA", "COVER-T5-large"): (-0.065, 0.26),
    ("WebQA", "COVER-T5-large"): (0.002, 0.20),
}


# Tiny deterministic display offsets separate near-coincident markers. They are
# small relative to the axis range and do not change the source values used for
# Pareto computation or textual reporting.
DISPLAY_OFFSETS = {
    ("MMLU", "Hard-DistilBERT"): (-0.006, 0.08),
    ("MMLU", "UniversalRAG-DistilBERT"): (0.010, -0.03),
    ("MMLU", "Hard-T5-large"): (0.020, -0.08),
    ("MMLU", "UniversalRAG-T5-large"): (0.032, 0.05),
    ("MMLU", "COVER-DistilBERT"): (-0.010, 0.10),
    ("MMLU", "COVER-T5-large"): (0.020, -0.10),
    ("SQuAD", "ParagraphRAG"): (-0.010, 0.03),
    ("SQuAD", "DocumentRAG"): (0.010, -0.03),
    ("SQuAD", "Hard-DistilBERT"): (-0.008, -0.08),
    ("SQuAD", "Hard-T5-large"): (0.008, -0.12),
    ("SQuAD", "Adaptive-RAG"): (-0.014, 0.06),
    ("SQuAD", "Self-RAG"): (0.014, 0.08),
    ("Natural Questions", "ParagraphRAG"): (-0.010, -0.02),
    ("Natural Questions", "DocumentRAG"): (0.010, 0.02),
    ("Natural Questions", "Hard-DistilBERT"): (-0.008, -0.08),
    ("Natural Questions", "Hard-T5-large"): (0.008, 0.08),
    ("Natural Questions", "Adaptive-RAG"): (-0.015, -0.06),
    ("Natural Questions", "Self-RAG"): (0.015, 0.08),
    ("HotpotQA", "UniversalRAG-DistilBERT"): (-0.018, -0.18),
    ("HotpotQA", "Hard-DistilBERT"): (-0.006, 0.14),
    ("HotpotQA", "UniversalRAG-T5-large"): (0.008, -0.12),
    ("HotpotQA", "Hard-T5-large"): (0.020, 0.12),
    ("HotpotQA", "DocumentRAG"): (0.032, 0.05),
    ("HotpotQA", "Self-RAG"): (0.018, 0.26),
    ("HotpotQA", "COVER-DistilBERT"): (-0.012, -0.28),
    ("HotpotQA", "COVER-T5-large"): (0.012, 0.28),
    ("WebQA", "UniversalRAG-DistilBERT"): (-0.018, -0.10),
    ("WebQA", "UniversalRAG-T5-large"): (-0.006, 0.06),
    ("WebQA", "Adaptive-RAG"): (0.006, -0.16),
    ("WebQA", "Self-RAG"): (0.018, 0.14),
    ("WebQA", "Hard-DistilBERT"): (-0.012, 0.08),
    ("WebQA", "Hard-T5-large"): (0.012, -0.08),
    ("WebQA", "COVER-DistilBERT"): (0.014, -0.12),
    ("WebQA", "COVER-T5-large"): (-0.014, 0.12),
}


def display_xy(row: dict[str, object]) -> tuple[float, float]:
    return float(row["cost"]), float(row["score"])

def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reader = csv.DictReader(StringIO(DATA.strip()))
    for row in reader:
        rows.append(
            {
                "target": row["target"],
                "method": row["method"],
                "group": row["group"],
                "cost": float(row["cost"]),
                "score": float(row["score"]),
                "metric": row["metric"],
            }
        )
    return rows


def pareto_frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frontier: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda r: (float(r["cost"]), -float(r["score"]))):
        cost = float(row["cost"])
        score = float(row["score"])
        dominated = False
        for other in rows:
            other_cost = float(other["cost"])
            other_score = float(other["score"])
            if (
                other_cost <= cost + 1e-12
                and other_score >= score - 1e-12
                and (other_cost < cost - 1e-12 or other_score > score + 1e-12)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)

    # If several non-dominated methods share visually identical costs, keep the
    # highest score for the line while all markers remain plotted.
    by_cost: dict[float, dict[str, object]] = {}
    for row in frontier:
        rounded = round(float(row["cost"]), 6)
        if rounded not in by_cost or float(row["score"]) > float(by_cost[rounded]["score"]):
            by_cost[rounded] = row
    return [by_cost[k] for k in sorted(by_cost)]


def short_cover_name(method: str) -> str:
    if method == "COVER-DistilBERT":
        return "COVER-D"
    if method == "COVER-T5-large":
        return "COVER-T5"
    return method


def in_view(row: dict[str, object], xlim: tuple[float, float], ylim: tuple[float, float]) -> bool:
    x, y = display_xy(row)
    return xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]


def draw_panel(
    ax: plt.Axes,
    target: str,
    ylabel: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    rows: list[dict[str, object]],
) -> None:
    panel_rows = [row for row in rows if row["target"] == target]
    visible_rows = [row for row in panel_rows if in_view(row, xlim, ylim)]

    frontier = [row for row in pareto_frontier(panel_rows) if in_view(row, xlim, ylim)]
    if len(frontier) >= 2:
        ax.plot(
            [float(row["cost"]) for row in frontier],
            [float(row["score"]) for row in frontier],
            color="#9a9a9a",
            linewidth=0.85,
            alpha=0.75,
            zorder=1,
            solid_capstyle="round",
        )

    for group, style in STYLE.items():
        group_rows = [row for row in visible_rows if row["group"] == group]
        ax.scatter(
            [float(row["cost"]) for row in group_rows],
            [float(row["score"]) for row in group_rows],
            s=style["size"],
            marker=style["marker"],
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=0.55 if group != "COVER" else 0.85,
            alpha=style["alpha"],
            zorder=style["zorder"],
        )

    for row in visible_rows:
        method = str(row["method"])
        if row["group"] != "COVER" or method != KEY_COVER_LABEL[target]:
            continue
        x, y = display_xy(row)
        dx, dy = LABEL_OFFSETS.get((target, method), (0.006, 0.0))
        ax.text(
            x + dx,
            y + dy,
            short_cover_name(method),
            fontsize=7.8,
            fontweight="bold",
            ha="left" if dx >= 0 else "right",
            va="center",
            clip_on=True,
            zorder=8,
        )

    ax.set_title(target, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=5)
    ax.set_xlabel("Average Execution Cost", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
    ax.grid(True, color="#d5d5d5", linestyle="--", linewidth=0.42, alpha=0.45)
    ax.tick_params(axis="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")


def draw_legend_panel(ax: plt.Axes) -> None:
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker=STYLE["Fixed retrieval"]["marker"],
            linestyle="None",
            markersize=6.5,
            markerfacecolor=STYLE["Fixed retrieval"]["facecolor"],
            markeredgecolor=STYLE["Fixed retrieval"]["edgecolor"],
            label="Fixed retrieval baselines",
        ),
        Line2D(
            [0],
            [0],
            marker=STYLE["Universal/adaptive/self"]["marker"],
            linestyle="None",
            markersize=7.0,
            markerfacecolor=STYLE["Universal/adaptive/self"]["facecolor"],
            markeredgecolor=STYLE["Universal/adaptive/self"]["edgecolor"],
            label="Universal/adaptive/self baselines",
        ),
        Line2D(
            [0],
            [0],
            marker=STYLE["Hard routing"]["marker"],
            linestyle="None",
            markersize=6.5,
            markerfacecolor=STYLE["Hard routing"]["facecolor"],
            markeredgecolor=STYLE["Hard routing"]["edgecolor"],
            label="Hard routing baselines",
        ),
        Line2D(
            [0],
            [0],
            marker=STYLE["COVER"]["marker"],
            linestyle="None",
            markersize=11,
            markerfacecolor=STYLE["COVER"]["facecolor"],
            markeredgecolor=STYLE["COVER"]["edgecolor"],
            label="COVER methods",
        ),
        Line2D([0], [0], color="#9a9a9a", linewidth=0.85, alpha=0.75, label="Visible Pareto frontier"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.90),
        frameon=False,
        fontsize=10.0,
        labelspacing=1.15,
        handlelength=2.2,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_va("center")
    ax.text(
        0.08,
        0.18,
        "Pareto frontier is computed only over plotted methods.\nLower cost and higher quality are preferred.",
        transform=ax.transAxes,
        fontsize=8.7,
        linespacing=1.45,
    )
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.02),
            0.98,
            0.94,
            transform=ax.transAxes,
            fill=False,
            linewidth=0.8,
            edgecolor="#777777",
            clip_on=False,
        )
    )


def main() -> None:
    rows = load_rows()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.3))
    for ax, (target, ylabel, xlim, ylim) in zip(axes.flat, PANELS):
        draw_panel(ax, target, ylabel, xlim, ylim, rows)
    draw_legend_panel(axes[1, 2])

    fig.suptitle(
        "Quality-Cost Trade-off Across In-Domain Benchmarks",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.062, right=0.985, bottom=0.075, top=0.905, wspace=0.31, hspace=0.38)

    out_dir = Path("els-cas-templates")
    for suffix, kwargs in [
        ("png", {"dpi": 600}),
        ("pdf", {}),
    ]:
        fig.savefig(out_dir / f"Figure_08.{suffix}", bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
