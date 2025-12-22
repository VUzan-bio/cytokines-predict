from __future__ import annotations

from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cell_type_counts(adata: anndata.AnnData, output_path: str) -> None:
    """Bar plot of cell counts per cell type and cytokine."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = adata.obs.copy()
    if "cytokine_type" not in df:
        df["cytokine_type"] = "unknown"
    counts = (
        df.groupby(["cell_type", "cytokine_type"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=counts, x="cell_type", y="n", hue="cytokine_type")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_cell_type_umap(adata: anndata.AnnData, output_path: str) -> None:
    """UMAP colored by cell type using X_scvi embedding."""
    if "X_umap" not in adata.obsm and "X_scvi" in adata.obsm:
        # compute quick neighbor graph/umap if missing
        import scanpy as sc

        sc.pp.neighbors(adata, use_rep="X_scvi")
        sc.tl.umap(adata)

    if "X_umap" not in adata.obsm:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = adata.obs.copy()
    df[["umap1", "umap2"]] = adata.obsm["X_umap"][:, :2]
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=df,
        x="umap1",
        y="umap2",
        hue="cell_type",
        s=6,
        alpha=0.6,
        legend=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
