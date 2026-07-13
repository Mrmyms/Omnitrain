"""
plot_results.py — Publication-quality figures for OmniTrain paper
Addresses:
  #5  task metric panels (MSE + TTF/success-rate side by side)
  #27 Shaded regions explicitly labelled ±1σ
  #28 Timeseries with multiple seeds + mean as thick line
  #29 Latency unit clarified in table captions
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_DATA = os.path.join(os.path.dirname(__file__), "data")
_COLORS = {"LSTM": "#e74c3c", "GRU": "#e67e22", "CfC": "#2980b9", "CfCFull": "#27ae60"}
_LS     = {"LSTM": "--", "GRU": "-.", "CfC": "-", "CfCFull": ":"}
_MARKERS= {"LSTM": "x",  "GRU": "^",  "CfC": "o", "CfCFull": "s"}
_LEVELS = [0, 20, 60]


def load_results():
    path = os.path.join(_DATA, "results_full.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run train_and_compare.py first to generate {path}")
    with open(path) as f:
        return json.load(f)


def fig_temporal_resilience(results):
    """
    Dual-panel: left = MSE (diagnostic), right = Mean TTF (primary task metric).
    Shaded regions = ±1σ (fix #27).  Only LSTM / GRU / CfC — CfCFull = ablation only.
    """
    mse = results["mse"]
    ttf = results["ttf"]
    models_plot = ["LSTM", "GRU", "CfC"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Temporal Resilience under Simulated Packet Loss (30 Seeds)", fontsize=13, fontweight='bold')

    # ── Left: MSE ────────────────────────────────────────────────────────────
    ax = axes[0]
    for name in models_plot:
        means = [np.mean(mse[name][str(l)]) for l in _LEVELS]
        stds  = [np.std( mse[name][str(l)]) for l in _LEVELS]
        ax.plot(_LEVELS, means, label=name, color=_COLORS[name],
                linestyle=_LS[name], marker=_MARKERS[name], linewidth=2.3, markersize=8)
        ax.fill_between(_LEVELS,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        color=_COLORS[name], alpha=0.18,
                        label=f"_{name} ±1σ")   # prefix _ prevents legend duplicate

    ax.set_title("(a) Predictive MSE — Force (N)\n[secondary / diagnostic metric]", fontsize=11)
    ax.set_xlabel("Simulated Packet Loss (%)", fontsize=11)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax.set_xticks(_LEVELS)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.text(0.02, 0.97, "Shaded = ±1σ across 30 seeds\n(weights & ZOH mask resampled independently)",
            transform=ax.transAxes, fontsize=8, va='top', color='gray')

    # ── Right: TTF ───────────────────────────────────────────────────────────
    ax = axes[1]
    for name in models_plot:
        means = [np.mean(ttf[name][str(l)]) for l in _LEVELS]
        stds  = [np.std( ttf[name][str(l)]) for l in _LEVELS]
        ax.plot(_LEVELS, means, label=name, color=_COLORS[name],
                linestyle=_LS[name], marker=_MARKERS[name], linewidth=2.3, markersize=8)
        ax.fill_between(_LEVELS,
                        np.maximum(0, np.array(means) - np.array(stds)),
                        np.minimum(500, np.array(means) + np.array(stds)),
                        color=_COLORS[name], alpha=0.18)

    ax.axhline(500, color='gray', linestyle=':', linewidth=1.5, label="Full episode (success)")
    ax.set_title("(b) Mean Time-To-Failure (steps)\n[primary task metric; 500 = full success]", fontsize=11)
    ax.set_xlabel("Simulated Packet Loss (%)", fontsize=11)
    ax.set_ylabel("Steps before pole fell (max 500)", fontsize=11)
    ax.set_xticks(_LEVELS)
    ax.set_ylim(0, 520)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.text(0.02, 0.97, "Shaded = ±1σ across 30 seeds\n(500 steps × 0.02s = 10s max episode)",
            transform=ax.transAxes, fontsize=8, va='top', color='gray')

    plt.tight_layout()
    out = os.path.join(_DATA, "temporal_resilience_chart.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved: {out}")
    plt.close()


def fig_statistics_heatmap(results):
    """Heatmap of Holm-corrected p-values for all 6 comparisons (fix #6)."""
    stats = results.get("stats", {})
    if not stats or "holm_p_mse" not in stats:
        print("[!] Holm-Bonferroni data not found — skipping heatmap (statsmodels needed).")
        return

    comparisons = stats["comparisons"]
    p_mse  = stats["holm_p_mse"]
    p_ttf  = stats["holm_p_ttf"]
    d_mse  = stats["cohen_d_mse"]
    d_ttf  = stats["cohen_d_ttf"]

    labels_short = [c.replace(" vs CfC @ ", "\n@").replace("% loss","% ") for c in comparisons]
    matrix = np.array([p_mse, p_ttf])

    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.1)
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(labels_short, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Holm p (MSE)", "Holm p (TTF)"], fontsize=10)

    for j, (pm, pt, dm, dt) in enumerate(zip(p_mse, p_ttf, d_mse, d_ttf)):
        ax.text(j, 0, f"p={pm:.3f}\nd={dm:.2f}", ha='center', va='center', fontsize=8)
        ax.text(j, 1, f"p={pt:.3f}\nd={dt:.2f}", ha='center', va='center', fontsize=8)

    plt.colorbar(im, ax=ax, label="Holm-corrected p-value")
    ax.set_title("Statistical Comparison: Holm-Bonferroni p-values + Cohen's d\n"
                 "(green = significant; all 6 comparisons vs. CfC)", fontsize=10)
    plt.tight_layout()
    out = os.path.join(_DATA, "stats_heatmap.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved: {out}")
    plt.close()


if __name__ == "__main__":
    res = load_results()
    fig_temporal_resilience(res)
    fig_statistics_heatmap(res)
    print("[✓] All figures generated.")
