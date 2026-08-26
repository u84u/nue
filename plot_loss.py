#!/usr/bin/env python3
"""Plot training loss history from loss_history.csv.

Usage:
    python plot_loss.py implementations/1M
"""

import sys
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_loss(checkpoint_dir: str):
    loss_file = os.path.join(checkpoint_dir, "loss_history.csv")
    if not os.path.exists(loss_file):
        print(f"Error: {loss_file} not found")
        return

    steps, losses, lrs = [], [], []
    with open(loss_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            lrs.append(float(row["lr"]))

    if not steps:
        print("No data in loss_history.csv")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1], sharex=True)
    fig.suptitle(f"Training Loss — {os.path.basename(checkpoint_dir)}", fontsize=14, fontweight="bold")

    # ── Loss plot ──
    ax1.plot(steps, losses, color="#2563eb", linewidth=0.8, alpha=0.4, label="Raw loss")

    # Rolling average (window = 1% of data, min 10)
    window = max(10, len(losses) // 100)
    if len(losses) >= window:
        import numpy as np
        kernel = np.ones(window) / window
        smoothed = np.convolve(losses, kernel, mode="valid")
        smooth_steps = steps[window - 1:]
        ax1.plot(smooth_steps, smoothed, color="#dc2626", linewidth=2, label=f"Moving avg ({window})")

    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Stats annotation
    min_loss = min(losses)
    min_step = steps[losses.index(min_loss)]
    ax1.annotate(
        f"Best: {min_loss:.4f} @ step {min_step:,}",
        xy=(min_step, min_loss),
        xytext=(min_step + len(steps) * 0.05, min_loss + (max(losses) - min_loss) * 0.15),
        arrowprops=dict(arrowstyle="->", color="#666"),
        fontsize=10, color="#333",
    )

    # ── LR plot ──
    ax2.plot(steps, lrs, color="#16a34a", linewidth=1.5)
    ax2.set_ylabel("Learning Rate", fontsize=12)
    ax2.set_xlabel("Step", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

    # ── Timing axis ──
    ax1r = ax1.twinx()
    ax1r.set_ylim(ax1.get_ylim())
    # Rough: assume constant throughput → map step to time
    total_steps = steps[-1]
    ax1r.set_ylabel(f"Step (×{total_steps:,} total)", fontsize=10, color="#999")
    ax1r.tick_params(axis="y", labelcolor="#999")

    plt.tight_layout()
    out_path = os.path.join(checkpoint_dir, "loss_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # non-interactive for CLI use
    if len(sys.argv) < 2:
        print("Usage: python plot_loss.py <checkpoint_dir>")
        sys.exit(1)
    plot_loss(sys.argv[1])
