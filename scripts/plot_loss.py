#!/usr/bin/env python3
"""Plot training loss history from loss_history.csv.

Usage:
    python scripts/plot_loss.py                              # default path
    python scripts/plot_loss.py implementations/125M/loss_history.csv
    python scripts/plot_loss.py --ascii                     # terminal-friendly ASCII plot
"""

import sys
import os
import csv

DEFAULT_CSV = "implementations/125M/loss_history.csv"


def load_csv(filepath):
    steps, losses, lrs = [], [], []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            lrs.append(float(row["lr"]))
    return steps, losses, lrs


def plot_matplotlib(steps, losses, lrs, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Loss curve
    ax1.plot(steps, losses, linewidth=0.8, alpha=0.3, color="steelblue", label="Raw loss")
    # Smoothing: exponential moving average
    if len(losses) > 10:
        smoothed = []
        alpha = 0.95
        s = losses[0]
        for l in losses:
            s = alpha * s + (1 - alpha) * l
            smoothed.append(s)
        ax1.plot(steps, smoothed, linewidth=1.5, color="darkblue", label=f"Smoothed (α={alpha})")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning rate
    ax2.plot(steps, lrs, linewidth=1.0, color="darkorange")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_ascii(steps, losses, width=80, height=20):
    """Terminal-friendly ASCII loss plot."""
    if not losses:
        print("No loss data to plot.")
        return

    # Downsample to fit width
    n = len(losses)
    if n > width:
        step = n // width
        sampled = [losses[i] for i in range(0, n, step)][:width]
    else:
        sampled = losses

    mn, mx = min(sampled), max(sampled)
    rng = mx - mn if mx != mn else 1.0

    print(f"\nLoss: {mn:.4f} → {mx:.4f}  (last: {losses[-1]:.4f})")
    print(f"Steps: {steps[0]} → {steps[-1]}")
    print()

    for row in range(height, -1, -1):
        threshold = mn + rng * row / height
        label = f"{threshold:8.4f} │"
        chars = []
        for val in sampled:
            if val >= threshold:
                chars.append("█")
            else:
                chars.append(" ")
        print(label + "".join(chars))

    print(" " * 9 + "└" + "─" * len(sampled))
    print(" " * 9 + f" Step {steps[0]}" + " " * (len(sampled) - 20) + f"Step {steps[-1]}")
    print()


def main():
    # Parse args
    csv_path = DEFAULT_CSV
    ascii_mode = "--ascii" in sys.argv

    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            csv_path = arg

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        print("Run training first to generate loss_history.csv")
        sys.exit(1)

    steps, losses, lrs = load_csv(csv_path)
    print(f"Loaded {len(losses)} entries from {csv_path}")

    if ascii_mode or not sys.stdout.isatty():
        plot_ascii(steps, losses)
    else:
        save_path = csv_path.replace(".csv", ".png")
        try:
            plot_matplotlib(steps, losses, lrs, save_path)
        except ImportError:
            print("matplotlib not installed, falling back to ASCII plot")
            plot_ascii(steps, losses)


if __name__ == "__main__":
    main()
