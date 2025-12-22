from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _get_2d_embedding(adata: anndata.AnnData) -> np.ndarray:
    if "X_umap" in adata.obsm:
        return np.asarray(adata.obsm["X_umap"])
    if "X_scvi" in adata.obsm:
        return np.asarray(adata.obsm["X_scvi"])[:, :2]
    raise ValueError("AnnData lacks X_umap or X_scvi embeddings.")


def plot_latent_overlay(
    adata_real: anndata.AnnData,
    adata_virtual: anndata.AnnData,
    output_path: str,
    max_points: int = 20000,
) -> None:
    """Plot latent scatter of real vs counterfactual cells."""
    embed_real = _get_2d_embedding(adata_real)
    embed_virtual = _get_2d_embedding(adata_virtual)

    real_df = pd.DataFrame(embed_real, columns=["dim1", "dim2"])
    real_df["label"] = "real"
    virt_df = pd.DataFrame(embed_virtual, columns=["dim1", "dim2"])
    virt_df["label"] = "counterfactual"
    plot_df = pd.concat([real_df, virt_df], axis=0)

    if plot_df.shape[0] > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=42)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=plot_df,
        x="dim1",
        y="dim2",
        hue="label",
        s=5,
        alpha=0.6,
        palette={"real": "#1f77b4", "counterfactual": "#d62728"},
    )
    plt.title("Latent space: real vs counterfactual")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_gene_scatter(
    gene_stats: pd.DataFrame,
    output_path: str,
    nonlinear_genes: Optional[Sequence[str]] = None,
) -> None:
    """Scatter of mean_real vs mean_virtual per gene."""
    df = gene_stats.copy()
    df["is_nonlinear"] = df["gene"].isin(nonlinear_genes or [])

    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=df,
        x="mean_real",
        y="mean_virtual",
        hue="is_nonlinear",
        s=6,
        alpha=0.6,
        palette={False: "#1f77b4", True: "#d62728"},
    )
    max_val = max(df["mean_real"].max(), df["mean_virtual"].max())
    plt.plot([0, max_val], [0, max_val], ls="--", color="gray")
    plt.xlabel("Mean expression (real)")
    plt.ylabel("Mean expression (virtual)")
    plt.title("Gene-level agreement")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pathway_bars(
    pathway_stats: pd.DataFrame,
    output_path: str,
    sort_by: str = "linearity_score",
) -> None:
    """Bar plot of pathway linearity scores."""
    if pathway_stats.empty:
        return
    df = pathway_stats.sort_values(sort_by, ascending=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="pathway", y=sort_by, color="#1f77b4")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(sort_by)
    plt.title("Pathway agreement")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pathway_heatmap(
    pathway_stats_per_cell: dict,
    output_path: str,
    value_col: str = "mean_abs_logFC",
) -> None:
    """Heatmap of pathway agreement across cell types."""
    if not pathway_stats_per_cell:
        return
    frames = []
    for cell_type, df in pathway_stats_per_cell.items():
        if df is None or df.empty:
            continue
        tmp = df[["pathway", value_col]].copy()
        tmp["cell_type"] = cell_type
        frames.append(tmp)
    if not frames:
        return
    mat = pd.concat(frames)
    pivot = mat.pivot(index="pathway", columns="cell_type", values=value_col)

    plt.figure(figsize=(8, max(3, 0.3 * pivot.shape[0])))
    sns.heatmap(pivot, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title(f"{value_col} per pathway / cell type")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
