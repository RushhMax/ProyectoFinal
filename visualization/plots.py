"""Visualizaciones del framework de explicabilidad temporal."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURES, RESULTS_DIR

PALETTE = {"LSTM": "#6C5CE7", "Transformer": "#00B894", "RandomForest": "#E17055"}


def plot_importance_heatmap(alpha: np.ndarray,
                            title: str = "",
                            features: List[str] = FEATURES,
                            save_path: Optional[Path] = None) -> None:
    """Heatmap de la matriz de importancia promedio T×V."""
    T = alpha.shape[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        alpha.T,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=[f"t-{T-1-i}" for i in range(T)],
        yticklabels=features,
        cbar_kws={"label": r"$\hat{\alpha}(t,v)$"},
    )
    ax.set_xlabel("Paso de tiempo (días hacia atrás)")
    ax.set_ylabel("Variable")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_variable_importance(alphas_dict: Dict[str, np.ndarray],
                              features: List[str] = FEATURES,
                              save_path: Optional[Path] = None) -> None:
    """Barras de importancia por variable para cada modelo."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(features))
    width = 0.25
    for idx, (name, alphas) in enumerate(alphas_dict.items()):
        importance = alphas.mean(axis=(0, 1)) if alphas.ndim == 3 else alphas.mean(axis=0)
        color = PALETTE.get(name, None)
        ax.bar(x + idx * width, importance, width, label=name, color=color)
    ax.set_xticks(x + width)
    ax.set_xticklabels(features)
    ax.set_ylabel("Importancia promedio")
    ax.set_title("Importancia por variable")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_temporal_importance(alphas_dict: Dict[str, np.ndarray],
                              window_size: int = None,
                              save_path: Optional[Path] = None) -> None:
    """Curva de importancia por paso de tiempo para cada modelo."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, alphas in alphas_dict.items():
        # importancia media por paso de tiempo
        imp = alphas.mean(axis=(0, 2)) if alphas.ndim == 3 else alphas.mean(axis=1)
        T = len(imp)
        lags = list(range(T - 1, -1, -1))  # t-19 … t-0
        color = PALETTE.get(name, None)
        ax.plot(lags, imp, marker="o", markersize=3, label=name, color=color)
    ax.set_xlabel("Días hacia atrás (lag)")
    ax.set_ylabel("Importancia promedio")
    ax.set_title("Importancia temporal por modelo")
    ax.invert_xaxis()
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_exp1_barplot(results: dict,
                      scenario: str = "rare_feature",
                      metrics: List[str] = None,
                      save_path: Optional[Path] = None) -> None:
    """
    Barras agrupadas de AUP / AUR / IM(A) / SM(A) para el Experimento 1.
    Compatible con el formato de las Tablas 1 y 2 de Dynamask.
    """
    if metrics is None:
        metrics = ["AUP", "AUR", "IM_A", "SM_A"]

    methods = list(results[scenario].keys())
    n_metrics = len(metrics)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    palette = plt.cm.tab10.colors
    for ax, metric in zip(axes, metrics):
        means = [results[scenario][m][metric]["mean"] for m in methods]
        stds  = [results[scenario][m][metric]["std"]  for m in methods]
        bars = ax.bar(methods, means, yerr=stds, capsize=4,
                      color=palette[:n_methods], alpha=0.85)
        ax.set_title(metric.replace("_", "(") + (")" if "_" in metric else ""),
                     fontsize=11)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(f"Experimento 1 — {scenario.replace('_', ' ').title()}\n"
                 "(AUP/AUR: ↑ mejor; IM_A: ↑ mejor; SM_A: ↓ mejor)",
                 fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps_comparison(alphas_dict: Dict[str, np.ndarray],
                              features: List[str] = FEATURES,
                              save_path: Optional[Path] = None) -> None:
    """Tres heatmaps lado a lado (uno por modelo) para comparar."""
    n = len(alphas_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, alphas) in zip(axes, alphas_dict.items()):
        mat = alphas.mean(axis=0) if alphas.ndim == 3 else alphas
        T = mat.shape[0]
        sns.heatmap(
            mat.T,
            ax=ax,
            cmap="YlOrRd",
            xticklabels=[f"t-{T-1-i}" for i in range(T)],
            yticklabels=features,
            cbar=ax is axes[-1],
        )
        ax.set_title(name)
        ax.set_xlabel("Paso de tiempo")
    axes[0].set_ylabel("Variable")
    fig.suptitle("Comparación de matrices de importancia promedio")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
