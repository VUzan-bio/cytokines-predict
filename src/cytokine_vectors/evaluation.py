from __future__ import annotations

from typing import Dict, Iterable, Optional

import anndata
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def compute_gene_level_agreement(
    adata_real: anndata.AnnData,
    adata_virtual: anndata.AnnData,
    cell_type: str,
    target_cytokine: str = "IL-6",
) -> pd.DataFrame:
    """Compare real vs virtual expression per gene for a cell type."""
    real_mask = (adata_real.obs.get("cell_type") == cell_type) & (
        adata_real.obs.get("cytokine_type") == target_cytokine
    )
    virt_mask = adata_virtual.obs.get("cell_type") == cell_type

    real = adata_real[real_mask]
    virt = adata_virtual[virt_mask]

    if real.n_obs == 0 or virt.n_obs == 0:
        raise ValueError(f"No cells for cell_type={cell_type}")

    real_mean = np.asarray(real.X.mean(axis=0)).ravel()
    virt_mean = np.asarray(virt.X.mean(axis=0)).ravel()
    logfc = np.log1p(virt_mean) - np.log1p(real_mean)

    # Optional correlation if same number of cells
    corr_values = []
    min_cells = min(real.n_obs, virt.n_obs)
    for gene_idx in range(real.n_vars):
        if min_cells < 2:
            corr_values.append(np.nan)
            continue
        try:
            r, _ = pearsonr(
                np.asarray(real.X[:min_cells, gene_idx]).ravel(),
                np.asarray(virt.X[:min_cells, gene_idx]).ravel(),
            )
        except Exception:
            r = np.nan
        corr_values.append(r)

    return pd.DataFrame(
        {
            "gene": real.var_names,
            "mean_real": real_mean,
            "mean_virtual": virt_mean,
            "logFC_virtual_vs_real": logfc,
            "pearson_r": corr_values,
        }
    )


def compute_pathway_agreement(
    gene_stats: pd.DataFrame,
    pathway_db: Dict[str, Iterable[str]],
    abs_logfc_threshold: float = 0.25,
) -> pd.DataFrame:
    """Aggregate gene-level stats to pathway scores."""
    rows = []
    stats = gene_stats.set_index("gene")
    for pathway, genes in pathway_db.items():
        genes_in = stats.loc[stats.index.intersection(pd.Index(genes))]
        if genes_in.empty:
            continue
        mean_abs_logfc = np.abs(genes_in["logFC_virtual_vs_real"]).mean()
        linearity = (np.abs(genes_in["logFC_virtual_vs_real"]) < abs_logfc_threshold).mean()
        rows.append(
            {
                "pathway": pathway,
                "n_genes": len(genes_in),
                "mean_abs_logFC": mean_abs_logfc,
                "linearity_score": linearity,
            }
        )
    return pd.DataFrame(rows)


def identify_nonlinear_genes(
    gene_stats: pd.DataFrame, abs_logfc_threshold: float = 0.5
) -> pd.DataFrame:
    """Genes with large discrepancies between virtual and real."""
    mask = np.abs(gene_stats["logFC_virtual_vs_real"]) > abs_logfc_threshold
    return gene_stats[mask].sort_values("logFC_virtual_vs_real", key=np.abs, ascending=False)
