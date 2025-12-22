"""Interpretation helpers for scVI cytokine latent space."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def latent_traversal(
    model,
    adata,
    *,
    groupby: str = "cytokine_type",
    start: str = "control",
    end: str = "stimulated",
    steps: int = 5,
) -> np.ndarray:
    """Interpolate between two cytokine states in latent space."""
    latent = model.get_latent_representation(adata)
    adata.obsm["X_scvi"] = latent
    start_centroid = latent[adata.obs[groupby] == start].mean(axis=0)
    end_centroid = latent[adata.obs[groupby] == end].mean(axis=0)
    alphas = np.linspace(0, 1, steps)
    return np.stack([start_centroid * (1 - a) + end_centroid * a for a in alphas])


def run_differential_expression(
    model,
    adata,
    *,
    groupby: str = "cytokine_type",
    group1: str = "stimulated",
    group2: str = "control",
    n_top_genes: int = 50,
):
    """Wrapper around scVI DE with sensible defaults."""
    return model.differential_expression(
        adata=adata,
        groupby=groupby,
        group1=group1,
        group2=group2,
        n_samples=1,
        batch_correction=True,
    ).head(n_top_genes)


def gene_loadings_by_correlation(model, adata, *, max_cells: int = 1000, top_n: int = 15) -> Dict[int, list[Tuple[str, float]]]:
    """Rank genes by correlation with each latent dimension."""
    sample_size = min(max_cells, adata.n_obs)
    indices = np.random.choice(adata.n_obs, size=sample_size, replace=False)
    latent = model.get_latent_representation(adata[indices])
    expr = model.get_normalized_expression(adata=adata[indices])
    expr = np.asarray(expr)
    loadings: Dict[int, list[Tuple[str, float]]] = {}
    for dim in range(latent.shape[1]):
        correlations = []
        for g in range(expr.shape[1]):
            gene_expr = expr[:, g]
            corr = np.corrcoef(latent[:, dim], gene_expr)[0, 1]
            correlations.append((adata.var_names[g], corr))
        sorted_genes = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:top_n]
        loadings[dim] = sorted_genes
    return loadings


def cytokine_pseudotime(model, adata, *, key: str = "cytokine_type", reference: str = "control") -> np.ndarray:
    """Order cells by distance from reference state in latent space."""
    latent = model.get_latent_representation(adata)
    reference_centroid = latent[adata.obs[key] == reference].mean(axis=0)
    pseudotime = np.linalg.norm(latent - reference_centroid, axis=1)
    adata.obs["cytokine_pseudotime"] = pseudotime
    return pseudotime
