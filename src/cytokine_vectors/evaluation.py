from __future__ import annotations

from typing import Dict, Iterable, Optional

import anndata
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def _select_expression(adata: anndata.AnnData, prefer_layer: str = "normalized"):
    """Return expression matrix, preferring a specific layer for scale alignment."""
    if prefer_layer and prefer_layer in adata.layers:
        return adata.layers[prefer_layer]
    return adata.X


def compute_gene_level_agreement(
    adata_real: anndata.AnnData,
    adata_virtual: anndata.AnnData,
    cell_type: str,
    target_cytokine: str = "IL-6",
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Compare real vs virtual expression per gene for a cell type on the same scale."""
    real_mask = (adata_real.obs.get("cell_type") == cell_type) & (
        adata_real.obs.get("cytokine_type") == target_cytokine
    )
    virt_mask = adata_virtual.obs.get("cell_type") == cell_type

    real = adata_real[real_mask]
    virt = adata_virtual[virt_mask]

    if real.n_obs == 0 or virt.n_obs == 0:
        raise ValueError(f"No cells for cell_type={cell_type}")

    real_expr = _select_expression(real, prefer_layer="normalized")
    virt_expr = _select_expression(virt, prefer_layer="normalized")
    real_mean = np.asarray(real_expr.mean(axis=0)).ravel()
    virt_mean = np.asarray(virt_expr.mean(axis=0)).ravel()
    logfc = np.log2((virt_mean + eps) / (real_mean + eps))

    # Optional correlation if same number of cells
    corr_values = []
    min_cells = min(real.n_obs, virt.n_obs)
    for gene_idx in range(real.n_vars):
        if min_cells < 2:
            corr_values.append(np.nan)
            continue
        try:
            r, _ = pearsonr(
                np.asarray(real_expr[:min_cells, gene_idx]).ravel(),
                np.asarray(virt_expr[:min_cells, gene_idx]).ravel(),
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


def compute_gene_level_agreement_for_all_cell_types(
    adata_real: anndata.AnnData,
    adata_virtual: anndata.AnnData,
    cell_types: list,
    target_cytokine: str,
) -> dict:
    """Compute gene-level stats for each cell type."""
    results = {}
    for ct in cell_types:
        try:
            results[ct] = compute_gene_level_agreement(
                adata_real=adata_real,
                adata_virtual=adata_virtual,
                cell_type=ct,
                target_cytokine=target_cytokine,
            )
        except Exception:
            results[ct] = None
    return results
