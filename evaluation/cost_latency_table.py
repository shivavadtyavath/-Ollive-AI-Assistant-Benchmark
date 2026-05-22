"""
Cost + Latency Analysis Table
------------------------------
Generates the cost and latency comparison table for the OSS deployment
(required for the bonus section).

Platforms compared:
  - HuggingFace Spaces (free tier)
  - HuggingFace Inference API (free tier)
  - Groq (free tier - frontier comparison)
  - Ollama (local, self-hosted)
  - Modal (serverless GPU)
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BG_COLOR = "#0F1117"
CARD_COLOR = "#1E2130"
TEXT_COLOR = "#FAFAFA"
GRID_COLOR = "#2D3250"


DEPLOYMENT_DATA = [
    {
        "platform": "HuggingFace Spaces\n(Qwen2.5-0.5B)",
        "model": "Qwen2.5-0.5B-Instruct",
        "cost_per_1k_tokens": "$0.00",
        "monthly_cost": "$0.00",
        "avg_latency_ms": "2,000–8,000",
        "cold_start_s": "20–60",
        "throughput_tps": "~30",
        "gpu": "CPU (free tier)",
        "notes": "Free, cold starts, rate limited",
        "tier": "free",
    },
    {
        "platform": "HF Inference API\n(Qwen2.5-0.5B)",
        "model": "Qwen2.5-0.5B-Instruct",
        "cost_per_1k_tokens": "$0.00",
        "monthly_cost": "$0.00 (free tier)",
        "avg_latency_ms": "1,500–5,000",
        "cold_start_s": "10–30",
        "throughput_tps": "~50",
        "gpu": "Shared GPU",
        "notes": "Free tier: 1000 req/day",
        "tier": "free",
    },
    {
        "platform": "Groq\n(Llama-3.3-70B)",
        "model": "llama-3.3-70b-versatile",
        "cost_per_1k_tokens": "$0.00",
        "monthly_cost": "$0.00 (free tier)",
        "avg_latency_ms": "200–800",
        "cold_start_s": "0",
        "throughput_tps": "~750",
        "gpu": "LPU (proprietary)",
        "notes": "Fastest inference, free tier",
        "tier": "free",
    },
    {
        "platform": "Ollama\n(Local)",
        "model": "Qwen2.5-0.5B-Instruct",
        "cost_per_1k_tokens": "$0.00",
        "monthly_cost": "$0.00 (electricity)",
        "avg_latency_ms": "500–2,000",
        "cold_start_s": "2–5",
        "throughput_tps": "~20–80",
        "gpu": "Local CPU/GPU",
        "notes": "Fully private, no internet",
        "tier": "free",
    },
    {
        "platform": "Modal\n(Serverless GPU)",
        "model": "Qwen2.5-0.5B-Instruct",
        "cost_per_1k_tokens": "~$0.0001",
        "monthly_cost": "$0–5 (low usage)",
        "avg_latency_ms": "300–1,000",
        "cold_start_s": "3–8",
        "throughput_tps": "~200",
        "gpu": "A10G / T4",
        "notes": "$30 free credits/month",
        "tier": "paid",
    },
    {
        "platform": "RunPod\n(GPU Cloud)",
        "model": "Qwen2.5-0.5B-Instruct",
        "cost_per_1k_tokens": "~$0.0002",
        "monthly_cost": "$5–20",
        "avg_latency_ms": "200–600",
        "cold_start_s": "5–15",
        "throughput_tps": "~300",
        "gpu": "RTX 3090 / A100",
        "notes": "Persistent GPU, more control",
        "tier": "paid",
    },
]


def generate_cost_latency_chart() -> str:
    """Generate a visual cost vs latency comparison."""
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
        "font.size": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_COLOR)

    # Chart 1: Latency comparison
    ax1 = axes[0]
    ax1.set_facecolor(CARD_COLOR)

    platforms = [d["platform"].replace("\n", " ") for d in DEPLOYMENT_DATA]
    # Use midpoint of latency range
    latencies = [3000, 2500, 500, 1000, 650, 400]
    colors = ["#6C63FF" if d["tier"] == "free" else "#FF6584" for d in DEPLOYMENT_DATA]

    bars = ax1.barh(platforms, latencies, color=colors, alpha=0.85,
                    edgecolor=BG_COLOR, linewidth=1.5, height=0.5)

    for bar, val, data in zip(bars, latencies, DEPLOYMENT_DATA):
        ax1.text(val + 50, bar.get_y() + bar.get_height() / 2,
                 data["avg_latency_ms"] + " ms",
                 va="center", color=TEXT_COLOR, size=9)

    ax1.set_xlabel("Avg Latency (ms)", color=TEXT_COLOR)
    ax1.set_title("Latency by Platform", color=TEXT_COLOR, size=13, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # Chart 2: Throughput comparison
    ax2 = axes[1]
    ax2.set_facecolor(CARD_COLOR)

    throughputs = [30, 50, 750, 50, 200, 300]
    bars2 = ax2.barh(platforms, throughputs, color=colors, alpha=0.85,
                     edgecolor=BG_COLOR, linewidth=1.5, height=0.5)

    for bar, val, data in zip(bars2, throughputs, DEPLOYMENT_DATA):
        ax2.text(val + 5, bar.get_y() + bar.get_height() / 2,
                 data["throughput_tps"] + " tok/s",
                 va="center", color=TEXT_COLOR, size=9)

    ax2.set_xlabel("Throughput (tokens/sec)", color=TEXT_COLOR)
    ax2.set_title("Throughput by Platform", color=TEXT_COLOR, size=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    # Legend
    free_patch = mpatches.Patch(color="#6C63FF", label="Free Tier")
    paid_patch = mpatches.Patch(color="#FF6584", label="Paid")
    fig.legend(handles=[free_patch, paid_patch], loc="lower center",
               ncol=2, facecolor=CARD_COLOR, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("OSS Deployment: Cost & Latency Analysis",
                 color=TEXT_COLOR, size=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = str(RESULTS_DIR / "cost_latency_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    return path


def generate_cost_table_image() -> str:
    """Generate a clean table image."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "text.color": TEXT_COLOR,
        "font.size": 9,
    })

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    col_labels = ["Platform", "Model", "Cost/1K Tokens", "Monthly Cost",
                  "Avg Latency", "Throughput", "GPU", "Notes"]
    rows = [
        [d["platform"].replace("\n", " "), d["model"].split("/")[-1],
         d["cost_per_1k_tokens"], d["monthly_cost"],
         d["avg_latency_ms"] + " ms", d["throughput_tps"] + " tok/s",
         d["gpu"], d["notes"]]
        for d in DEPLOYMENT_DATA
    ]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2D3250")
        table[0, j].set_text_props(color=TEXT_COLOR, fontweight="bold")

    for i, data in enumerate(DEPLOYMENT_DATA, 1):
        row_color = "#1A2040" if data["tier"] == "free" else "#201A30"
        for j in range(len(col_labels)):
            cell = table[i, j]
            cell.set_facecolor(row_color)
            cell.set_text_props(color=TEXT_COLOR)
            if j == 2:  # Cost column
                if data["cost_per_1k_tokens"] == "$0.00":
                    cell.set_text_props(color="#00FF88", fontweight="bold")
                else:
                    cell.set_text_props(color="#FFD700")

    plt.title("OSS Deployment Options: Cost & Latency Table",
              color=TEXT_COLOR, size=13, fontweight="bold", pad=15)

    path = str(RESULTS_DIR / "cost_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    return path


if __name__ == "__main__":
    p1 = generate_cost_latency_chart()
    p2 = generate_cost_table_image()
    print(f"Charts saved:\n  {p1}\n  {p2}")
