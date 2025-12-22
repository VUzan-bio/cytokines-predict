"""Plotting utilities for scVI cytokine analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import scanpy as sc


def plot_latent_umap(adata, color: Iterable[str], output_dir: str | Path, *, neighbors_key: str = "X_scvi") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep=neighbors_key)
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)
    for covariate in color:
        if covariate not in adata.obs and covariate not in adata.var_names:
            continue
        ax = sc.pl.umap(
            adata,
            color=covariate,
            show=False,
            frameon=False,
            return_fig=True,
        )
        out_path = output_dir / f"umap_{covariate}.png"
        ax.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(ax)


def plot_training_history(model, output_dir: str | Path) -> Optional[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = getattr(model, "history_", None)
    if history is None:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if "elbo_train" in history:
        ax.plot(history["elbo_train"], label="elbo_train")
    if "elbo_validation" in history:
        ax.plot(history["elbo_validation"], label="elbo_val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.legend()
    out_path = output_dir / "training_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_expression_recovery(true_values, reconstructed, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(true_values, reconstructed, s=4, alpha=0.5)
    ax.set_xlabel("True expression")
    ax.set_ylabel("Reconstructed expression")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def create_scvi_plots(model, adata, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_latent_umap(adata, ["cytokine_type", "donor_id", "batch_id"], output_dir)
    plot_training_history(model, output_dir)
