"""Generate all 8 statistical plots for the Attention Residuals blog post.

Uses matplotlib with a clean academic style: white background, pastel colors,
sans-serif fonts. All figures sized at 8x5 inches (reasonable for blog).
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data" / "attention-residuals"
OUT_DIR = Path(__file__).parent / "output" / "attention-residuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Color palette (pastel)
C_BLUE = "#6B9BD2"
C_ORANGE = "#E8956A"
C_GREEN = "#7BC8A4"
C_PURPLE = "#B39DDB"
C_PINK = "#F48FB1"
C_GRAY = "#BDBDBD"
C_DARK_BLUE = "#3F6FA0"
C_DARK_ORANGE = "#C4703A"


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {name}.png")


# ── 1. Standard Residual Magnitudes ──────────────────────────────────────────
def plot_standard_residual_magnitudes():
    df = pd.read_csv(DATA_DIR / "fig_standard_residual_magnitudes.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    tokens = ["token_The", "token_cat", "token_sat", "token_down"]
    labels = ['"The"', '"cat"', '"sat"', '"down"']
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]
    for col, label, c in zip(tokens, labels, colors):
        ax.plot(df["layer"], df[col], marker="o", markersize=5, linewidth=2,
                label=label, color=c)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Residual Stream Norm")
    ax.set_title("Residual Stream Magnitude Growth (Standard Transformer)")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "fig_standard_residual_magnitudes")


# ── 2. Pre-Norm Dilution Growth ──────────────────────────────────────────────
def plot_prenorm_dilution():
    df = pd.read_csv(DATA_DIR / "fig_prenorm_dilution_growth.csv")
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(df["layer"], df["standard_residual_norm"], marker="o",
             markersize=5, linewidth=2, color=C_ORANGE,
             label="Standard Residual Norm")
    ax1.plot(df["layer"], df["attnres_norm"], marker="s", markersize=5,
             linewidth=2, color=C_BLUE, label="AttnRes Norm")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Norm")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.bar(df["layer"], df["layer_contribution_fraction"], alpha=0.25,
            color=C_GRAY, width=0.6, label="Layer Contribution Fraction")
    ax2.set_ylabel("Layer Contribution Fraction", color=C_GRAY)
    ax2.tick_params(axis="y", labelcolor=C_GRAY)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_GRAY)
    ax2.legend(loc="upper right")

    ax1.set_title("Pre-Norm Dilution: Residual Growth vs. Layer Contribution")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "fig_prenorm_dilution_growth")


# ── 3. AttnRes vs Standard Magnitudes Comparison ────────────────────────────
def plot_attnres_magnitudes():
    df = pd.read_csv(DATA_DIR / "fig_attnres_magnitudes_comparison.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["layer"], df["standard_norm"], marker="o", markersize=6,
            linewidth=2.5, color=C_ORANGE, label="Standard Transformer")
    ax.plot(df["layer"], df["attnres_norm"], marker="s", markersize=6,
            linewidth=2.5, color=C_BLUE, label="Attention Residual")
    ax.axhline(y=1.0, color=C_GRAY, linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Residual Stream Norm")
    ax.set_title("Residual Magnitude: Standard vs. Attention Residual")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "fig_attnres_magnitudes_comparison")


# ── 4. Gradient Norm Comparison ──────────────────────────────────────────────
def plot_gradient_norms():
    df = pd.read_csv(DATA_DIR / "fig_gradient_norm_comparison.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["layer"], df["standard_gradient_norm"], marker="o",
            markersize=6, linewidth=2.5, color=C_ORANGE,
            label="Standard (vanishing)")
    ax.plot(df["layer"], df["attnres_gradient_norm"], marker="s",
            markersize=6, linewidth=2.5, color=C_BLUE,
            label="Attention Residual (stable)")
    ax.set_xlabel("Layer (early \u2192 late)")
    ax.set_ylabel("Gradient Norm (normalized)")
    ax.set_title("Gradient Flow: Standard vs. Attention Residual")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save(fig, "fig_gradient_norm_comparison")


# ── 5. Benchmark Results Bar Chart ──────────────────────────────────────────
def plot_benchmark_bars():
    df = pd.read_csv(DATA_DIR / "fig_benchmark_results_bar.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df))
    w = 0.35
    bars1 = ax.bar(x - w / 2, df["baseline"], w, label="Baseline",
                   color=C_ORANGE, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w / 2, df["attnres"], w, label="AttnRes",
                   color=C_BLUE, edgecolor="white", linewidth=0.5)

    # Delta labels on top of AttnRes bars
    for i, (bar, delta) in enumerate(zip(bars2, df["delta"])):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"+{delta}", ha="center", va="bottom", fontsize=8,
                color=C_DARK_BLUE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(df["benchmark"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Benchmark Performance: Baseline vs. Attention Residual")
    ax.legend()
    ax.set_ylim(0, max(df["attnres"].max(), df["baseline"].max()) + 8)
    save(fig, "fig_benchmark_results_bar")


# ── 6. Scaling Law Curves ───────────────────────────────────────────────────
def plot_scaling_laws():
    df = pd.read_csv(DATA_DIR / "fig_scaling_law_curves.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["compute_flops_1e18"], df["standard_val_loss"], marker="o",
            markersize=6, linewidth=2.5, color=C_ORANGE,
            label="Standard Transformer")
    ax.plot(df["compute_flops_1e18"], df["attnres_val_loss"], marker="s",
            markersize=6, linewidth=2.5, color=C_BLUE,
            label="Attention Residual")
    ax.set_xscale("log")
    ax.set_xlabel("Compute (FLOPs \u00d7 10\u00b9\u2078)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Scaling Laws: Validation Loss vs. Compute Budget")
    ax.legend()

    # Annotate the compute savings
    ax.annotate("~25% less compute\nfor same loss",
                xy=(16, 2.55), xytext=(40, 2.75),
                arrowprops=dict(arrowstyle="->", color=C_DARK_BLUE, lw=1.5),
                fontsize=9, color=C_DARK_BLUE, ha="center")
    save(fig, "fig_scaling_law_curves")


# ── 7. Depth vs Width Optimal Configuration ─────────────────────────────────
def plot_depth_width():
    df = pd.read_csv(DATA_DIR / "fig_depth_width_optimal.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Optimal width at each depth
    ax1.plot(df["depth_layers"], df["standard_optimal_width"], marker="o",
             markersize=6, linewidth=2.5, color=C_ORANGE, label="Standard")
    ax1.plot(df["depth_layers"], df["attnres_optimal_width"], marker="s",
             markersize=6, linewidth=2.5, color=C_BLUE, label="AttnRes")
    ax1.set_xlabel("Depth (layers)")
    ax1.set_ylabel("Optimal Width (hidden dim)")
    ax1.set_title("Optimal Width at Each Depth")
    ax1.legend()

    # Right: Validation loss at optimal config
    ax2.plot(df["depth_layers"], df["standard_loss"], marker="o",
             markersize=6, linewidth=2.5, color=C_ORANGE, label="Standard")
    ax2.plot(df["depth_layers"], df["attnres_loss"], marker="s",
             markersize=6, linewidth=2.5, color=C_BLUE, label="AttnRes")
    ax2.set_xlabel("Depth (layers)")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Loss at Optimal Depth-Width Configuration")
    ax2.legend()

    fig.suptitle("Depth-Width Trade-off: AttnRes Favors Deeper, Narrower Models",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "fig_depth_width_optimal")


# ── 8. Overhead Breakdown ───────────────────────────────────────────────────
def plot_overhead():
    df = pd.read_csv(DATA_DIR / "fig_overhead_breakdown.csv")
    overhead = df[df["category"] == "overhead"]
    gains = df[df["category"] == "gain"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Overhead costs
    bars1 = ax1.barh(overhead["metric"], overhead["value"],
                     color=C_PINK, edgecolor="white", height=0.5)
    ax1.set_xlabel("Percentage (%)")
    ax1.set_title("Cost of Attention Residual")
    for bar, val in zip(bars1, overhead["value"]):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val}%", va="center", fontsize=10, color=C_DARK_ORANGE)
    ax1.set_xlim(0, max(overhead["value"]) * 1.3)

    # Right: Performance gains
    bars2 = ax2.barh(gains["metric"], gains["value"],
                     color=C_GREEN, edgecolor="white", height=0.5)
    ax2.set_xlabel("Percentage / Points (%)")
    ax2.set_title("Benefits of Attention Residual")
    for bar, val in zip(bars2, gains["value"]):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"+{val}%", va="center", fontsize=10, color="#2E7D32")
    ax2.set_xlim(0, max(gains["value"]) * 1.25)

    fig.suptitle("Overhead vs. Gains: Is Attention Residual Worth It?",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "fig_overhead_breakdown")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating plots for Attention Residuals blog post...")
    plot_standard_residual_magnitudes()
    plot_prenorm_dilution()
    plot_attnres_magnitudes()
    plot_gradient_norms()
    plot_benchmark_bars()
    plot_scaling_laws()
    plot_depth_width()
    plot_overhead()
    print("Done! All 8 plots saved.")
