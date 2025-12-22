"""Data loading and preprocessing tailored for scVI cytokine analysis."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def load_cytokine_data(data_path: str) -> AnnData:
    """Load AnnData from disk."""
    logger.info("Loading AnnData from %s", data_path)
    return sc.read_h5ad(data_path)


def annotate_mitochondrial_genes(adata: AnnData, mito_prefix: str = "MT-") -> AnnData:
    adata.var["mt"] = adata.var_names.str.upper().str.startswith(mito_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    return adata


def filter_cells_and_genes(
    adata: AnnData,
    *,
    min_genes: int,
    max_genes: int,
    max_mito: float,
) -> AnnData:
    mask = (
        (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["n_genes_by_counts"] <= max_genes)
        & (adata.obs["pct_counts_mt"] <= (max_mito * 100))
    )
    filtered = adata[mask].copy()
    sc.pp.filter_genes(filtered, min_cells=3)
    logger.info("Filtered cells: %d -> %d", adata.n_obs, filtered.n_obs)
    return filtered


def detect_doublets_lightweight(adata: AnnData, threshold: float = 0.25) -> AnnData:
    """Approximate doublet detection (lightweight proxy for scDblFinder)."""
    if "doublet_score" not in adata.obs:
        total = adata.obs["total_counts"].astype(float)
        genes = adata.obs["n_genes_by_counts"].astype(float)
        normalized_score = (total / (total.median() + 1e-8) + genes / (genes.median() + 1e-8)) / 2
        score = (normalized_score - normalized_score.min()) / (normalized_score.max() - normalized_score.min() + 1e-8)
        adata.obs["doublet_score"] = score
    adata.obs["is_doublet"] = adata.obs["doublet_score"] > threshold
    return adata


def normalize_and_log_transform(adata: AnnData) -> AnnData:
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    adata.layers["normalized"] = adata.X.copy()
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()
    return adata


def select_hvgs(adata: AnnData, n_top_genes: int = 2000) -> AnnData:
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        layer="counts",
        subset=True,
    )
    return adata


def encode_cytokine_metadata(
    adata: AnnData,
    *,
    cytokine_key_map,
    donor_key: str,
) -> AnnData:
    if hasattr(cytokine_key_map, "__dict__"):
        cytokine_key_map = dict(cytokine_key_map.__dict__)
    elif not isinstance(cytokine_key_map, dict):
        cytokine_key_map = {
            "cytokine_type": "cytokine_type",
            "concentration": "concentration",
            "stimulation_duration": "stimulation_duration",
        }
    defaults = {
        "cytokine_type": "unknown",
        "concentration": "unknown",
        "stimulation_duration": "unknown",
    }
    for target, source in cytokine_key_map.items():
        if source in adata.obs:
            adata.obs[target] = adata.obs[source].astype("category")
        else:
            adata.obs[target] = defaults.get(target)
    if donor_key in adata.obs:
        adata.obs[donor_key] = adata.obs[donor_key].astype("category")
    else:
        adata.obs[donor_key] = "donor_unknown"
    return adata


def stratified_splits(
    adata: AnnData,
    *,
    donor_key: str,
    cytokine_key: str,
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: int = 42,
) -> AnnData:
    strata = adata.obs[donor_key].astype(str) + "|" + adata.obs[cytokine_key].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_size, random_state=random_state)
    train_idx, holdout_idx = next(sss.split(np.zeros(len(strata)), strata))

    holdout_strata = strata.iloc[holdout_idx]
    val_fraction = val_size / (1 - train_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    val_idx_rel, test_idx_rel = next(sss_val.split(np.zeros(len(holdout_idx)), holdout_strata))
    holdout_idx = np.array(holdout_idx)
    val_idx = holdout_idx[val_idx_rel]
    test_idx = holdout_idx[test_idx_rel]

    adata.obs["split"] = "train"
    adata.obs.iloc[val_idx, adata.obs.columns.get_loc("split")] = "val"
    adata.obs.iloc[test_idx, adata.obs.columns.get_loc("split")] = "test"
    logger.info("Split cells into train/val/test: %s", adata.obs["split"].value_counts().to_dict())
    return adata


def preprocess_for_scvi(adata: AnnData, config) -> AnnData:
    """Full preprocessing chain to ready AnnData for scVI."""
    data_cfg = getattr(config, "data", config)
    training_cfg = getattr(config, "training", None)
    cytokine_key_map = getattr(data_cfg, "cytokine_keys", {"cytokine_type": "cytokine_type", "concentration": "concentration", "stimulation_duration": "stimulation_duration"})
    if hasattr(cytokine_key_map, "__dict__"):
        cytokine_key_map = dict(cytokine_key_map.__dict__)
    adata = annotate_mitochondrial_genes(adata, mito_prefix=getattr(data_cfg, "mito_prefix", "MT-"))
    adata = filter_cells_and_genes(
        adata,
        min_genes=getattr(data_cfg, "min_genes", 500),
        max_genes=getattr(data_cfg, "max_genes", 7500),
        max_mito=getattr(data_cfg, "max_mito", 0.15),
    )
    adata = detect_doublets_lightweight(adata, threshold=getattr(data_cfg, "doublet_score_threshold", 0.25))
    adata = adata[~adata.obs["is_doublet"]].copy()
    adata = normalize_and_log_transform(adata)
    adata = encode_cytokine_metadata(
        adata,
        cytokine_key_map=cytokine_key_map,
        donor_key=getattr(data_cfg, "donor_key", "donor_id"),
    )
    adata = select_hvgs(adata, n_top_genes=getattr(data_cfg, "hvg_genes", 2000))
    adata = stratified_splits(
        adata,
        donor_key=getattr(data_cfg, "donor_key", "donor_id"),
        cytokine_key=cytokine_key_map.get("cytokine_type", "cytokine_type"),
        train_size=getattr(training_cfg, "train_size", 0.8) if training_cfg is not None else 0.8,
        val_size=getattr(training_cfg, "validation_size", 0.1) if training_cfg is not None else 0.1,
        random_state=getattr(training_cfg, "random_seed", 42) if training_cfg is not None else 42,
    )
    return adata
