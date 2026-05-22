"""
Evaluation Report Generator
-----------------------------
Generates beautiful infographic-style charts and a PDF/HTML report
comparing OSS vs Frontier assistant performance.

Charts produced:
  1. Radar chart - Overall comparison
  2. Bar chart - Category-wise scores
  3. Latency distribution
  4. Safety pass/fail breakdown
  5. Hallucination rate comparison
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from evaluation.evaluator import EvalResult


# Brand colours
OSS_COLOR = "#6C63FF"       # Purple - OSS
FRONTIER_COLOR = "#FF6584"  # Pink - Frontier
BG_COLOR = "#0F1117"        # Dark background
CARD_COLOR = "#1E2130"
TEXT_COLOR = "#FAFAFA"
GRID_COLOR = "#2D3250"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": CARD_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.4,
        "font.family": "DejaVu Sans",
        "font.size": 11,
    })


def generate_radar_chart(summary: Dict) -> str:
    """Overall performance radar chart."""
    _apply_dark_style()

    categories = ["Factual\nAccuracy", "Safety\nScore", "Bias\nScore",
                  "Refusal\nRate", "Speed\n(inv. latency)"]

    def get_vals(model):
        s = summary.get(model, {})
        latency = s.get("avg_latency_ms", 5000)
        speed = max(0, 1 - latency / 10000)  # Normalise: lower latency = higher score
        return [
            s.get("factual_accuracy", 0),
            s.get("safety_score", 0),
            s.get("bias_score", 0),
            s.get("refusal_rate", 0),
            speed,
        ]

    oss_vals = get_vals("oss")
    frontier_vals = get_vals("frontier")

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    oss_vals_plot = oss_vals + oss_vals[:1]
    frontier_vals_plot = frontier_vals + frontier_vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True),
                           facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    ax.plot(angles, oss_vals_plot, "o-", linewidth=2, color=OSS_COLOR, label="OSS (Qwen2.5)")
    ax.fill(angles, oss_vals_plot, alpha=0.25, color=OSS_COLOR)

    ax.plot(angles, frontier_vals_plot, "o-", linewidth=2, color=FRONTIER_COLOR, label="Frontier (Llama-3.3-70B)")
    ax.fill(angles, frontier_vals_plot, alpha=0.25, color=FRONTIER_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, color=TEXT_COLOR)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=8, color=GRID_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.5)
    ax.spines["polar"].set_color(GRID_COLOR)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.title("Overall Performance Comparison", color=TEXT_COLOR, size=14,
              fontweight="bold", pad=20)

    path = str(RESULTS_DIR / "radar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    return path


def generate_category_bar_chart(summary: Dict) -> str:
    """Side-by-side bar chart for each metric."""
    _apply_dark_style()

    metrics = {
        "Factual Accuracy": ("factual_accuracy", "Higher is better"),
        "Hallucination Rate": ("hallucination_rate", "Lower is better"),
        "Safety Score": ("safety_score", "Higher is better"),
        "Bias Score": ("bias_score", "Higher is better"),
        "Refusal Rate": ("refusal_rate", "Higher is better"),
    }

    labels = list(metrics.keys())
    oss_vals = [summary.get("oss", {}).get(v[0], 0) for v in metrics.values()]
    frontier_vals = [summary.get("frontier", {}).get(v[0], 0) for v in metrics.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    bars1 = ax.bar(x - width / 2, oss_vals, width, label="OSS (Qwen2.5)",
                   color=OSS_COLOR, alpha=0.85, edgecolor=BG_COLOR, linewidth=1.5)
    bars2 = ax.bar(x + width / 2, frontier_vals, width, label="Frontier (Llama-3.3-70B)",
                   color=FRONTIER_COLOR, alpha=0.85, edgecolor=BG_COLOR, linewidth=1.5)

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", color=TEXT_COLOR, size=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", color=TEXT_COLOR, size=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0–1)", color=TEXT_COLOR)
    ax.set_title("Metric-by-Metric Comparison", color=TEXT_COLOR, size=14, fontweight="bold")
    ax.legend(facecolor=CARD_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    path = str(RESULTS_DIR / "category_bar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    return path


def generate_latency_chart(summary: Dict) -> str:
    """Latency comparison chart."""
    _apply_dark_style()

    models = ["OSS\n(Qwen2.5-0.5B)", "Frontier\n(Llama-3.3-70B via Groq)"]
    latencies = [
        summary.get("oss", {}).get("avg_latency_ms", 0),
        summary.get("frontier", {}).get("avg_latency_ms", 0),
    ]
    colors = [OSS_COLOR, FRONTIER_COLOR]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_COLOR)
    ax.set_facecolor(CARD_COLOR)

    bars = ax.barh(models, latencies, color=colors, alpha=0.85,
                   edgecolor=BG_COLOR, linewidth=1.5, height=0.4)

    for bar, val in zip(bars, latencies):
        ax.text(val + 50, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} ms", va="center", color=TEXT_COLOR, size=11, fontweight="bold")

    ax.set_xlabel("Average Latency (ms)", color=TEXT_COLOR)
    ax.set_title("Response Latency Comparison", color=TEXT_COLOR, size=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    path = str(RESULTS_DIR / "latency_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    return path


def generate_safety_breakdown(results: Dict[str, List[EvalResult]]) -> str:
    """Safety pass/fail pie charts."""
    _apply_dark_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG_COLOR)

    for ax, (model_type, color, label) in zip(
        axes,
        [("oss", OSS_COLOR, "OSS (Qwen2.5)"),
         ("frontier", FRONTIER_COLOR, "Frontier (Llama-3.3-70B)")]
    ):
        safety_results = [r for r in results.get(model_type, []) if r.category == "safety"]
        passed = sum(1 for r in safety_results if r.correctly_refused is True)
        failed = sum(1 for r in safety_results if r.correctly_refused is False)
        total = len(safety_results)

        if total == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color=TEXT_COLOR)
            continue

        sizes = [passed, failed]
        pie_colors = [color, "#FF4444"]
        explode = (0.05, 0)

        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, colors=pie_colors,
            autopct="%1.0f%%", startangle=90,
            textprops={"color": TEXT_COLOR, "size": 12},
            wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2},
        )
        for at in autotexts:
            at.set_color(TEXT_COLOR)
            at.set_fontsize(12)

        ax.set_title(f"{label}\nSafety Tests ({total} prompts)",
                     color=TEXT_COLOR, size=11, fontweight="bold")
        ax.set_facecolor(CARD_COLOR)

        legend_elements = [
            mpatches.Patch(facecolor=color, label=f"Safe Refusal ({passed})"),
            mpatches.Patch(facecolor="#FF4444", label=f"Unsafe Compliance ({failed})"),
        ]
        ax.legend(handles=legend_elements, loc="lower center",
                  bbox_to_anchor=(0.5, -0.15), facecolor=CARD_COLOR,
                  edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    plt.suptitle("Content Safety: Pass/Fail Breakdown", color=TEXT_COLOR,
                 size=14, fontweight="bold", y=1.02)

    path = str(RESULTS_DIR / "safety_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    return path


def generate_summary_scorecard(summary: Dict) -> str:
    """A clean scorecard table as an image."""
    _apply_dark_style()

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    metrics = [
        ("Hallucination Rate ↓", "hallucination_rate"),
        ("Factual Accuracy ↑", "factual_accuracy"),
        ("Safety Score ↑", "safety_score"),
        ("Bias Score ↑", "bias_score"),
        ("Refusal Rate ↑", "refusal_rate"),
        ("Avg Latency (ms) ↓", "avg_latency_ms"),
    ]

    col_labels = ["Metric", "OSS (Qwen2.5)", "Frontier (Llama-3.3-70B)", "Winner"]
    rows = []
    for label, key in metrics:
        oss_val = summary.get("oss", {}).get(key, 0)
        frontier_val = summary.get("frontier", {}).get(key, 0)

        # Determine winner
        if "↓" in label:  # Lower is better
            winner = "OSS" if oss_val < frontier_val else "Frontier"
        else:  # Higher is better
            winner = "OSS" if oss_val > frontier_val else "Frontier"

        if abs(oss_val - frontier_val) < 0.02:
            winner = "Tie"

        oss_str = f"{oss_val:.3f}" if isinstance(oss_val, float) else str(oss_val)
        frontier_str = f"{frontier_val:.3f}" if isinstance(frontier_val, float) else str(frontier_val)
        rows.append([label, oss_str, frontier_str, winner])

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2D3250")
        table[0, j].set_text_props(color=TEXT_COLOR, fontweight="bold")

    # Style rows
    for i, row in enumerate(rows, 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor(CARD_COLOR if i % 2 == 0 else "#161B2E")
            cell.set_text_props(color=TEXT_COLOR)
            # Colour winner column
            if j == 3:
                winner_val = row[3]
                if winner_val == "OSS":
                    cell.set_text_props(color=OSS_COLOR, fontweight="bold")
                elif winner_val == "Frontier":
                    cell.set_text_props(color=FRONTIER_COLOR, fontweight="bold")
                else:
                    cell.set_text_props(color="#FFD700", fontweight="bold")

    plt.title("Evaluation Scorecard Summary", color=TEXT_COLOR, size=14,
              fontweight="bold", pad=15)

    path = str(RESULTS_DIR / "scorecard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    return path


def generate_all_charts(
    summary: Dict,
    results: Dict[str, List[EvalResult]],
) -> Dict[str, str]:
    """Generate all charts and return paths."""
    paths = {}
    try:
        paths["radar"] = generate_radar_chart(summary)
    except Exception as e:
        print(f"Radar chart error: {e}")

    try:
        paths["bar"] = generate_category_bar_chart(summary)
    except Exception as e:
        print(f"Bar chart error: {e}")

    try:
        paths["latency"] = generate_latency_chart(summary)
    except Exception as e:
        print(f"Latency chart error: {e}")

    try:
        paths["safety"] = generate_safety_breakdown(results)
    except Exception as e:
        print(f"Safety chart error: {e}")

    try:
        paths["scorecard"] = generate_summary_scorecard(summary)
    except Exception as e:
        print(f"Scorecard error: {e}")

    return paths
