from __future__ import annotations

from typing import Dict, Iterable, Optional

import anndata
import numpy as np
import pandas as pd

from .reference import get_default_marker_defs


def _mean_expr(adata: anndata.AnnData, genes: Iterable[str]) -> np.ndarray:
    genes = list(genes)
    if not genes:
        return np.zeros(adata.n_obs)
    intersect = adata.var_names.intersection(genes)
    if len(intersect) == 0:
        return np.zeros(adata.n_obs)
    X = adata[:, intersect].X
    mean_vals = np.asarray(X.mean(axis=1)).ravel()
    return mean_vals


def annotate_cell_types(
    adata: anndata.AnnData,
    method: str = "marker_score",
    min_score_diff: float = 0.1,
    unknown_label: str = "unknown",
    overwrite: bool = True,
    marker_defs: Optional[Dict[str, Dict[str, Iterable[str]]]] = None,
) -> anndata.AnnData:
    """Add or refine adata.obs['cell_type'] using marker scores."""
    if marker_defs is None:
        marker_defs = get_default_marker_defs()

    if method != "marker_score":
        raise NotImplementedError(f"Method '{method}' not implemented.")

    # Ensure log-transformed layer if counts layer exists
    if "log1p" in adata.layers:
        ad = adata.copy()
        ad.X = ad.layers["log1p"]
    else:
        ad = adata

    scores = {}
    for ct, defs in marker_defs.items():
        pos = defs.get("positive", [])
        neg = defs.get("negative", [])
        pos_score = _mean_expr(ad, pos)
        neg_score = _mean_expr(ad, neg)
        scores[ct] = pos_score - neg_score

    score_df = pd.DataFrame(scores, index=adata.obs_names)
    best = score_df.idxmax(axis=1)
    sorted_scores = np.sort(score_df.values, axis=1)
    top = sorted_scores[:, -1]
    second = sorted_scores[:, -2] if score_df.shape[1] > 1 else np.zeros_like(top)
    confident = (top - second) >= min_score_diff

    new_labels = best.where(confident, other=unknown_label).astype(str)
    if overwrite or "cell_type" not in adata.obs:
        adata.obs["cell_type"] = new_labels
    else:
        # Keep existing labels where present
        adata.obs["cell_type"] = adata.obs["cell_type"].astype(str).fillna(new_labels)

    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    return adata
