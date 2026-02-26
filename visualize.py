#!/usr/bin/env python
"""Visualize self-prediction wrapper experiment results.

Reads data from local wandb run directories and produces comparison charts
in the figures/ directory.
"""

import json
import glob
import os
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

WANDB_DIR = Path(__file__).parent / "wandb"
OUTPUT_DIR = Path(__file__).parent / "figures"

COLORS = {"gpt2": "#4C72B0", "qwen": "#DD8452"}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def load_runs(wandb_dir):
    runs = []
    for run_dir in sorted(wandb_dir.glob("run-*")):
        files_dir = run_dir / "files"
        summary_path = files_dir / "wandb-summary.json"
        config_path = files_dir / "config.yaml"
        if not summary_path.exists() or not config_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)
        if "js_mean" not in summary:
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        model = config["model"]["value"]
        n_prompts = config["n_prompts"]["value"]
        use_chat = config.get("use_chat_template", {}).get("value", False)

        # Load examples table
        table_files = list((files_dir / "media" / "table").glob("*.table.json"))
        examples = []
        if table_files:
            with open(table_files[0]) as f:
                table = json.load(f)
            for row in table["data"]:
                examples.append(dict(zip(table["columns"], row)))

        short_model = model.split("/")[-1]
        label = f"{short_model} (n={n_prompts})"

        runs.append({
            "model": model,
            "short_model": short_model,
            "label": label,
            "n_prompts": n_prompts,
            "use_chat": use_chat,
            "js_mean": summary["js_mean"],
            "js_median": summary["js_median"],
            "top1_agreement": summary["top1_agreement"],
            "examples": examples,
            "color": COLORS["qwen"] if "wen" in model.lower() else COLORS["gpt2"],
        })

    return runs


def plot_metrics_comparison(runs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    labels = [r["label"] for r in runs]
    colors = [r["color"] for r in runs]
    x = np.arange(len(runs))

    # JS Divergence
    js_means = [r["js_mean"] for r in runs]
    js_medians = [r["js_median"] for r in runs]
    bars1 = ax1.bar(x, js_means, 0.5, color=colors, edgecolor="black", linewidth=0.5)
    for i, (mean, median) in enumerate(zip(js_means, js_medians)):
        ax1.plot(i, median, "D", color="white", markersize=6, markeredgecolor="black", zorder=3)
    ax1.set_ylabel("JS Divergence")
    ax1.set_title("JS Divergence (mean)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax1.set_ylim(0, max(js_means) * 1.3)
    for i, v in enumerate(js_means):
        ax1.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)

    # Top-1 Agreement
    agreements = [r["top1_agreement"] * 100 for r in runs]
    bars2 = ax2.bar(x, agreements, 0.5, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Top-1 Agreement (%)")
    ax2.set_title("Top-1 Agreement")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylim(0, 115)
    for i, v in enumerate(agreements):
        ax2.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9)

    fig.suptitle("Self-Prediction Wrapper: Aggregate Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_metrics_comparison.png")
    plt.close(fig)
    print("  Saved fig1_metrics_comparison.png")


def plot_js_distribution(runs):
    fig, ax = plt.subplots(figsize=(8, 5))

    positions = np.arange(len(runs))
    data = []
    for r in runs:
        js_values = [e["js"] for e in r["examples"]]
        data.append(js_values)

    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=1.5))
    for patch, run in zip(bp["boxes"], runs):
        patch.set_facecolor(run["color"])
        patch.set_alpha(0.6)

    # Overlay individual points with jitter
    rng = np.random.default_rng(42)
    for i, (run, vals) in enumerate(zip(runs, data)):
        jitter = rng.normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=run["color"], edgecolor="black", linewidth=0.3,
                   s=30, zorder=3, alpha=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels([r["label"] for r in runs], fontsize=10)
    ax.set_ylabel("JS Divergence")
    ax.set_title("Per-Prompt JS Divergence Distribution (first 20 prompts)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_js_distribution.png")
    plt.close(fig)
    print("  Saved fig2_js_distribution.png")


def plot_per_prompt_comparison(runs):
    # Compare GPT-2 (first run with 200 prompts) vs Qwen on the 10 base prompts
    gpt2_run = next((r for r in runs if "gpt2" in r["model"] and r["n_prompts"] == 200), None)
    qwen_run = next((r for r in runs if "wen" in r["model"].lower()), None)
    if not gpt2_run or not qwen_run:
        print("  Skipping fig3: need both GPT-2 (200) and Qwen runs")
        return

    n = 10  # base prompts only
    gpt2_js = [gpt2_run["examples"][i]["js"] for i in range(n)]
    qwen_js = [qwen_run["examples"][i]["js"] for i in range(n)]
    prompts = [e["prompt"][:35] + ("..." if len(e["prompt"]) > 35 else "")
               for e in gpt2_run["examples"][:n]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)
    ax.plot(x, gpt2_js, "o-", color=COLORS["gpt2"], label="GPT-2", linewidth=2, markersize=7)
    ax.plot(x, qwen_js, "s-", color=COLORS["qwen"], label="Qwen2.5-0.5B-Instruct", linewidth=2, markersize=7)
    ax.fill_between(x, gpt2_js, qwen_js, alpha=0.15, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("JS Divergence")
    ax.set_title("Per-Prompt JS Divergence: Base Model vs Chat Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_per_prompt_comparison.png")
    plt.close(fig)
    print("  Saved fig3_per_prompt_comparison.png")


def plot_examples_table(runs):
    gpt2_run = next((r for r in runs if "gpt2" in r["model"] and r["n_prompts"] == 200), None)
    qwen_run = next((r for r in runs if "wen" in r["model"].lower()), None)
    if not gpt2_run or not qwen_run:
        print("  Skipping fig4: need both GPT-2 (200) and Qwen runs")
        return

    n = 10

    def escape_token(t):
        return repr(t) if t.strip() == "" else t

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    for ax, run, title in [
        (ax1, gpt2_run, "GPT-2 (base model)"),
        (ax2, qwen_run, "Qwen2.5-0.5B-Instruct (chat model)"),
    ]:
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=10)

        cell_text = []
        cell_colors = []
        for e in run["examples"][:n]:
            prompt = e["prompt"][:40] + ("..." if len(e["prompt"]) > 40 else "")
            direct = escape_token(e["top1_direct"])
            wrapped = escape_token(e["top1_wrapped"])
            same = e["top1_direct"] == e["top1_wrapped"]
            cell_text.append([prompt, f"{e['js']:.3f}", direct, wrapped])
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
        print(f"  {r['label']}: JS={r['js_mean']:.4f}, Top1={r['top1_agreement']:.3f}")
    print()
    print("Generating figures...")
    plot_metrics_comparison(runs)
    plot_js_distribution(runs)
    plot_per_prompt_comparison(runs)
    plot_examples_table(runs)
    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
