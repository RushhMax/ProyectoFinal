"""
Gráficos adicionales para las tablas de Resultados que todavía no tenían
un gráfico propio. Lee los JSON ya generados en results/, no recalcula
ni reentrena nada.

Uso: python -m visualization.plot_results_summary
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from config import RESULTS_DIR

MODEL_PALETTE = {"LSTM": "#6C5CE7", "Transformer": "#00B894",
                 "RandomForest": "#E17055", "Persistencia": "#8C8C8C"}
PAIR_PALETTE = {"LSTM_Transformer": "#6C5CE7", "LSTM_RandomForest": "#00B894",
                "Transformer_RandomForest": "#E17055"}
PAIR_LABELS = {"LSTM_Transformer": "LSTM-Transformer",
               "LSTM_RandomForest": "LSTM-Random Forest",
               "Transformer_RandomForest": "Transformer-Random Forest"}

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#333333",
})


def _load(name):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def _style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.25, linestyle="-", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _bar_labels(ax, bars, fmt="{:.4f}", fontsize=8.5, offset_frac=0.02):
    ymax = ax.get_ylim()[1]
    for b in bars:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = ymax * offset_frac * (1 if h >= 0 else -1)
        ax.text(b.get_x() + b.get_width() / 2, h + offset, fmt.format(h),
                ha="center", va=va, fontsize=fontsize, color="#222222")


def plot_metrics_barplot(metrics: dict, save_path: Path, title: str,
                         subtitle: str, extra_first: tuple = None) -> None:
    """3 subplots (MSE, MAE, RMSE), barras por modelo (y opcionalmente
    un baseline adicional, p.ej. persistencia, como primera barra)."""
    order = ["lstm", "transformer", "random_forest"]
    labels = ["LSTM", "Transformer", "RandomForest"]
    if extra_first:
        extra_key, extra_label = extra_first
        order = [extra_key] + order
        labels = [extra_label] + labels

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6))
    for ax, metric in zip(axes, ["mse", "mae", "rmse"]):
        vals = [metrics[k][metric] for k in order]
        colors = [MODEL_PALETTE.get(lbl, "#333333") for lbl in labels]
        bars = ax.bar(labels, vals, color=colors, alpha=0.92,
                      edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_ylim(top=max(vals) * 1.18)
        _bar_labels(ax, bars, fmt="{:.4f}")
        ax.set_title(metric.upper())
        ax.set_ylabel("Error")
        ax.tick_params(axis="x", rotation=15)
        _style_ax(ax)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11.5, y=1.05)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_consistency_robustness(save_path: Path) -> None:
    """3 paneles: por target, por referencia, por régimen de mercado."""
    expl = _load("explainability_results.json")["consistency_spearman"]
    logret = _load("logreturn_explainability_results.json")["consistency_spearman"]
    add = _load("additional_results.json")
    by_ref = add["reference_sensitivity"]["consistency_by_reference"]
    by_regime = {k: v["consistency_spearman"] for k, v in add["market_regime"].items()}

    pairs = ["LSTM_Transformer", "LSTM_RandomForest", "Transformer_RandomForest"]

    panels = [
        ("Variable objetivo", {"Close": expl, "Log_Return": logret}),
        ("Vector de referencia", {"Media": by_ref["mean"], "Mediana": by_ref["median"],
                                   "Cero": by_ref["zero"]}),
        ("Régimen de mercado", {"Bajista\n2022": by_regime["bajista_2022"],
                                 "Alcista\n2023-24": by_regime["alcista_2023_2024"]}),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, (panel_title, conditions) in zip(axes, panels):
        cond_names = list(conditions.keys())
        x = np.arange(len(cond_names))
        width = 0.25
        for i, pair in enumerate(pairs):
            vals = [conditions[c][pair] for c in cond_names]
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=PAIR_LABELS[pair], color=PAIR_PALETTE[pair],
                          edgecolor="white", linewidth=0.5, zorder=3)
            _bar_labels(ax, bars, fmt="{:.2f}", fontsize=7.5, offset_frac=0.015)
        ax.axhline(0, color="black", linewidth=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_names, fontsize=9.5)
        ax.set_title(panel_title)
        _style_ax(ax)
    axes[0].set_ylabel(r"Spearman $\rho$")
    axes[0].set_ylim(-0.15, 0.92)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.04), fontsize=9.5, frameon=False)
    fig.suptitle("LSTM-Transformer se mantiene por encima de cualquier par con "
                 "Random Forest en las 7 condiciones evaluadas",
                 fontsize=11, y=1.1)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_directional_accuracy(save_path: Path) -> None:
    dir_acc = _load("additional_results.json")["directional_accuracy"]
    models = ["LSTM", "Transformer", "RandomForest"]
    close_vals = [dir_acc["target_close"][m] * 100 for m in models]
    logret_vals = [dir_acc["target_logreturn"][m] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    b1 = ax.bar(x - width / 2, close_vals, width, label="Target Close",
               color=[MODEL_PALETTE[m] for m in models], alpha=0.95,
               edgecolor="white", linewidth=0.6, zorder=3)
    b2 = ax.bar(x + width / 2, logret_vals, width, label="Target Log_Return",
               color=[MODEL_PALETTE[m] for m in models], alpha=0.55, hatch="//",
               edgecolor="white", linewidth=0.6, zorder=3)
    _bar_labels(ax, b1, fmt="{:.1f}%", fontsize=8, offset_frac=0.028)
    _bar_labels(ax, b2, fmt="{:.1f}%", fontsize=8, offset_frac=0.008)
    ax.axhline(50, color="#B22222", linestyle="--", linewidth=1.4, zorder=2,
               label="Azar (50%)")
    ax.set_xlim(-0.55, 2.75)
    ax.annotate("Los seis valores caen\ndentro de $\\pm$4 puntos del azar",
                xy=(2.4, 55), xytext=(1.25, 61),
                fontsize=9, color="#B22222", ha="center",
                arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Exactitud direccional (%)")
    ax.set_ylim(0, 68)
    ax.set_title("Exactitud direccional sobre el conjunto de prueba")
    ax.legend(fontsize=8.5, loc="lower right")
    _style_ax(ax)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_consistency(save_path: Path) -> None:
    d = _load("temporal_order_ablation.json")["real_sp500"]
    pairs = ["LSTM_Transformer", "LSTM_RandomForest", "Transformer_RandomForest"]
    fo_vals = [d["consistency_fo"][p] for p in pairs]
    shuf_vals = [d["consistency_shuffled"][p] for p in pairs]
    labels = [PAIR_LABELS[p] for p in pairs]

    x = np.arange(len(pairs))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8, 4.8))
    b1 = ax.bar(x - width / 2, fo_vals, width, label="FO (respeta el orden)",
               color="#4C72B0", edgecolor="white", linewidth=0.6, zorder=3)
    b2 = ax.bar(x + width / 2, shuf_vals, width, label="Shuffled (viola el orden)",
               color="#C44E52", edgecolor="white", linewidth=0.6, zorder=3)
    _bar_labels(ax, b1, fmt="{:.3f}")
    _bar_labels(ax, b2, fmt="{:.3f}")
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.annotate("Se vuelve negativa",
                xy=(1 + width / 2, shuf_vals[1]), xytext=(1.55, -0.13),
                fontsize=9.5, color="#C44E52", ha="center",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.3))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(-0.16, 0.85)
    ax.set_title("Consistencia inter-modelo, FO frente a violar el orden temporal")
    ax.legend(fontsize=9.5)
    _style_ax(ax)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    metrics = _load("metrics.json")
    plot_metrics_barplot(
        metrics, RESULTS_DIR / "metrics_barplot.png",
        "Error de predicción sobre el conjunto de prueba (target = Close)",
        "Random Forest comete aproximadamente 35x más error que las redes "
        "neuronales (MSE)")
    print("metrics_barplot.png listo")

    logret_metrics = _load("logreturn_metrics.json")
    plot_metrics_barplot(
        logret_metrics, RESULTS_DIR / "logreturn_metrics_barplot.png",
        "Error de predicción con target = retorno logarítmico",
        "Ningún modelo entrenado supera a la persistencia ingenua (barra gris)",
        extra_first=("persistence", "Persistencia"))
    print("logreturn_metrics_barplot.png listo")

    plot_consistency_robustness(RESULTS_DIR / "consistency_robustness.png")
    print("consistency_robustness.png listo")

    plot_directional_accuracy(RESULTS_DIR / "dir_accuracy_barplot.png")
    print("dir_accuracy_barplot.png listo")

    plot_ablation_consistency(RESULTS_DIR / "ablation_consistency_barplot.png")
    print("ablation_consistency_barplot.png listo")


if __name__ == "__main__":
    main()
