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

PALETTE = {"LSTM": "#4C72B0", "Transformer": "#DD8452", "RandomForest": "#55A868"}


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
