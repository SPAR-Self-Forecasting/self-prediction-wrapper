#!/usr/bin/env python
"""Visualize self-prediction wrapper experiment results.

Reads data from local wandb run directories and produces comparison charts
in the figures/ directory.
"""

import json
import os
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

WANDB_DIR = Path(__file__).parent / "wandb"
OUTPUT_DIR = Path(__file__).parent / "figures"

# Color palette
C_LOCAL = "#4C72B0"
C_API = "#DD8452"
C_API_LP = "#55A868"  # API models with logprobs

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def load_runs(wandb_dir):
    """Load all valid experiment runs, keeping only the latest per model."""
    all_runs = []
    for run_dir in sorted(wandb_dir.glob("run-*")):
        files_dir = run_dir / "files"
        summary_path = files_dir / "wandb-summary.json"
        config_path = files_dir / "config.yaml"
        if not summary_path.exists() or not config_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)
        if "top1_agreement" not in summary:
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        model = config.get("model", {}).get("value", "")
        if not model:
            continue
        n_prompts = config["n_prompts"]["value"]
        is_api = config.get("api", {}).get("value", False)

        # Load examples table
        table_dir = files_dir / "media" / "table"
        examples = []
        if table_dir.exists():
            table_files = list(table_dir.glob("*.table.json"))
            if table_files:
                with open(table_files[0]) as f:
                    table = json.load(f)
                for row in table["data"]:
                    examples.append(dict(zip(table["columns"], row)))

        # Skip runs where all examples are empty (thinking model failures)
        if is_api and examples:
            non_empty = sum(1 for e in examples
                           if e.get("top1_direct", "") or e.get("top1_wrapped", ""))
            if non_empty == 0:
                continue

        short_model = model.split("/")[-1]

        run_data = {
            "model": model,
            "short_model": short_model,
            "n_prompts": n_prompts,
            "is_api": is_api,
            "top1_agreement": summary["top1_agreement"],
            "js_mean": summary.get("js_mean"),
            "js_median": summary.get("js_median"),
            "examples": examples,
            "run_dir": str(run_dir),
        }
        all_runs.append(run_data)

    # Keep latest run per (model, n_prompts) — dirs are sorted by timestamp
    latest = {}
    for r in all_runs:
        key = (r["model"], r["n_prompts"])
        latest[key] = r  # last one wins (sorted ascending)

    runs = list(latest.values())

    # Assign labels and colors
    for r in runs:
        if r["is_api"]:
            r["label"] = r["short_model"]
            r["color"] = C_API_LP if r["js_mean"] is not None else C_API
        else:
            r["label"] = f"{r['short_model']} (n={r['n_prompts']})"
            r["color"] = C_LOCAL

    return runs


def plot_top1_all(runs):
    """Bar chart of top-1 agreement across ALL models (local + API)."""
    # Sort by top1_agreement descending
    sorted_runs = sorted(runs, key=lambda r: r["top1_agreement"], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [r["label"] for r in sorted_runs]
    agreements = [r["top1_agreement"] * 100 for r in sorted_runs]
    colors = [r["color"] for r in sorted_runs]
    x = np.arange(len(sorted_runs))

    ax.bar(x, agreements, 0.6, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(agreements):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)

    ax.set_ylabel("Top-1 Agreement (%)")
    ax.set_title("Self-Prediction Wrapper: Top-1 Agreement Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 115)

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=C_LOCAL, edgecolor="black", label="Local (HF)"),
        Patch(facecolor=C_API, edgecolor="black", label="API (no logprobs)"),
        Patch(facecolor=C_API_LP, edgecolor="black", label="API (with logprobs)"),
    ]
    ax.legend(handles=legend_items, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_top1_all_models.png")
    plt.close(fig)
    print("  Saved fig1_top1_all_models.png")


def plot_local_comparison(runs):
    """Original fig1: metrics comparison for local models only."""
    local = [r for r in runs if not r["is_api"]]
    if len(local) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    labels = [r["label"] for r in local]
    x = np.arange(len(local))

    js_means = [r["js_mean"] for r in local]
    bars1 = ax1.bar(x, js_means, 0.5, color=[r["color"] for r in local],
                    edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("JS Divergence")
    ax1.set_title("JS Divergence (mean)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax1.set_ylim(0, max(js_means) * 1.3)
    for i, v in enumerate(js_means):
        ax1.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)

    agreements = [r["top1_agreement"] * 100 for r in local]
    bars2 = ax2.bar(x, agreements, 0.5, color=[r["color"] for r in local],
                    edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Top-1 Agreement (%)")
    ax2.set_title("Top-1 Agreement")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylim(0, 115)
    for i, v in enumerate(agreements):
        ax2.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

    fig.suptitle("Local Models: Aggregate Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_local_metrics.png")
    plt.close(fig)
    print("  Saved fig2_local_metrics.png")


def plot_js_distribution(runs):
    """Box + strip plot for runs that have per-prompt JS data."""
    runs_with_js = [r for r in runs if r["js_mean"] is not None and r["examples"]
                    and any(e.get("js", -1) >= 0 for e in r["examples"])]
    if not runs_with_js:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(runs_with_js))
    data = []
    for r in runs_with_js:
        js_values = [e["js"] for e in r["examples"] if e.get("js", -1) >= 0]
        data.append(js_values)

    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=1.5))
    for patch, run in zip(bp["boxes"], runs_with_js):
        patch.set_facecolor(run["color"])
        patch.set_alpha(0.6)

    rng = np.random.default_rng(42)
    for i, (run, vals) in enumerate(zip(runs_with_js, data)):
        jitter = rng.normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=run["color"], edgecolor="black", linewidth=0.3,
                   s=30, zorder=3, alpha=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels([r["label"] for r in runs_with_js], fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("JS Divergence")
    ax.set_title("Per-Prompt JS Divergence Distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_js_distribution.png")
    plt.close(fig)
    print("  Saved fig3_js_distribution.png")


def plot_examples_table(runs):
    """Render example tables for select models."""
    # Pick a local model and a few interesting API models
    local_chat = next((r for r in runs if not r["is_api"]
                       and r["top1_agreement"] < 0.5), None)
    api_models = sorted([r for r in runs if r["is_api"] and r["examples"]],
                        key=lambda r: r["top1_agreement"])

    models_to_show = []
    if local_chat:
        models_to_show.append((local_chat, f"{local_chat['short_model']} (local)"))
    # Show lowest and highest API agreement
    if len(api_models) >= 2:
        models_to_show.append((api_models[0], f"{api_models[0]['short_model']} (API, lowest)"))
        models_to_show.append((api_models[-1], f"{api_models[-1]['short_model']} (API, highest)"))

    if not models_to_show:
        return

    n = 10

    def escape_token(t):
        if not t:
            return "''"
        return repr(t) if t.strip() == "" else t

    fig, axes = plt.subplots(len(models_to_show), 1,
                             figsize=(12, 3.2 * len(models_to_show)))
    if len(models_to_show) == 1:
        axes = [axes]

    for ax, (run, title) in zip(axes, models_to_show):
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)

        cell_text = []
        cell_colors = []
        has_js = run["js_mean"] is not None
        for e in run["examples"][:n]:
            prompt = e["prompt"][:40] + ("..." if len(e["prompt"]) > 40 else "")
            direct = escape_token(e.get("top1_direct", ""))
            wrapped = escape_token(e.get("top1_wrapped", ""))
            same = e.get("top1_direct", "") == e.get("top1_wrapped", "")
            js_str = f"{e['js']:.3f}" if has_js and e.get("js", -1) >= 0 else "n/a"
            cell_text.append([prompt, js_str, direct, wrapped])
            bg = "#d4edda" if same else "#f8d7da"
            cell_colors.append(["white", "white", bg, bg])

        table = ax.table(
            cellText=cell_text,
            colLabels=["Prompt", "JS Div", "Top-1 Direct", "Top-1 Wrapped"],
            cellColours=cell_colors,
            colColours=["#e9ecef"] * 4,
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.auto_set_column_width([0, 1, 2, 3])
        table.scale(1, 1.4)

    fig.suptitle("Qualitative Examples: Direct vs Wrapped Top-1 Predictions",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "fig4_examples_table.png")
    plt.close(fig)
    print("  Saved fig4_examples_table.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    runs = load_runs(WANDB_DIR)
    print(f"Found {len(runs)} experiment runs:")
    for r in runs:
        js_str = f"JS={r['js_mean']:.4f}" if r["js_mean"] is not None else "JS=n/a"
        print(f"  {r['label']:40s} {js_str:15s} Top1={r['top1_agreement']:.3f}"
              f"  {'(API)' if r['is_api'] else '(local)'}")
    print()
    print("Generating figures...")
    plot_top1_all(runs)
    plot_local_comparison(runs)
    plot_js_distribution(runs)
    plot_examples_table(runs)
    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
