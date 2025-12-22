"""Evaluation metrics for cytokine scVI models."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

logger = logging.getLogger(__name__)


def _safe_silhouette(latent: np.ndarray, labels) -> Optional[float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None
    return float(silhouette_score(latent, labels))


def _neighbor_entropy(adata, batch_key: str) -> Optional[float]:
    """Approximate batch mixing using entropy of neighbor batches."""
    if batch_key not in adata.obs:
        return None
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=15)
    connectivities = adata.obsp["connectivities"]
    batch_categories = adata.obs[batch_key].astype("category")
    entropy_values = []
    for i in range(adata.n_obs):
        neighbors = connectivities[i].indices
        if len(neighbors) == 0:
            continue
        neighbor_batches = batch_categories.iloc[neighbors].values
        _, counts = np.unique(neighbor_batches, return_counts=True)
        probs = counts / counts.sum()
        entropy_values.append(-np.sum(probs * np.log(probs + 1e-8)))
    if not entropy_values:
        return None
    return float(np.mean(entropy_values))


def expression_recovery(model, adata, n_cells: int = 500) -> Optional[float]:
    if adata.n_obs == 0:
        return None
    sample_size = min(n_cells, adata.n_obs)
    indices = np.random.choice(adata.n_obs, size=sample_size, replace=False)
    recon = model.get_normalized_expression(indices=indices)
    true = adata.layers.get("normalized")
    if true is None:
        return None
    true_subset = np.asarray(true[indices].toarray() if hasattr(true, "toarray") else true[indices]).ravel()
    recon_subset = np.asarray(recon).ravel()
    if true_subset.std() == 0 or recon_subset.std() == 0:
        return None
    return float(np.corrcoef(true_subset, recon_subset)[0, 1])


def evaluate_scvi_model(
    model,
    adata,
    *,
    label_key: str = "cell_type",
    cytokine_key: str = "cytokine_type",
    batch_key: str = "donor_id",
    resolution: float = 0.8,
) -> Dict[str, Optional[float]]:
    """Compute key scVI evaluation metrics."""
    metrics: Dict[str, Optional[float]] = {}
    latent = model.get_latent_representation(adata)
    adata.obsm["X_scvi"] = latent
    metrics["cytokine_silhouette"] = _safe_silhouette(latent, adata.obs.get(cytokine_key, ["unknown"]))

    if label_key in adata.obs:
        sc.pp.neighbors(adata, use_rep="X_scvi")
        sc.tl.leiden(adata, resolution=resolution, key_added="leiden_scvi")
        metrics["ari"] = float(
            adjusted_rand_score(adata.obs[label_key], adata.obs["leiden_scvi"])
        )
        metrics["nmi"] = float(
            normalized_mutual_info_score(adata.obs[label_key], adata.obs["leiden_scvi"])
        )
    else:
        metrics["ari"] = None
        metrics["nmi"] = None

    metrics["batch_entropy"] = _neighbor_entropy(adata, batch_key=batch_key)
    recon = model.get_reconstruction_error(adata=adata)
    if isinstance(recon, dict):
        recon_val = recon.get("reconstruction_loss", list(recon.values())[0])
    else:
        recon_val = recon
    metrics["reconstruction_error"] = float(recon_val)
    metrics["expression_recovery"] = expression_recovery(model, adata)
    return metrics
