from __future__ import annotations

from typing import List

import anndata
import numpy as np
import pandas as pd
import torch

from .vectors import CytokineVector


def _decode_latent(
    model,
    latent_z: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Decode latent representations to normalized expression."""
    try:
        expr = model.get_normalized_expression(
            adata=None,
            latent_z=latent_z,
            batch_size=batch_size,
            return_numpy=True,
        )
        expr = np.asarray(expr)
        if expr.shape[0] != latent_z.shape[0]:
            expr = expr[: latent_z.shape[0]]
        return expr
    except Exception:
        # Fallback to direct decoder call
        model.module.eval()
        with torch.no_grad():
            z = torch.tensor(latent_z, device=model.device, dtype=torch.float32)
            zero_lib = torch.zeros((z.shape[0], 1), device=model.device)
            batch_index = torch.zeros((z.shape[0], 1), device=model.device, dtype=torch.int64)
            outputs = model.module.decoder(z, zero_lib, batch_index)
            px_scale = outputs["px_scale"]
            expr = px_scale.cpu().numpy()
            if expr.shape[0] != latent_z.shape[0]:
                expr = expr[: latent_z.shape[0]]
            return expr


def generate_counterfactuals(
    adata: anndata.AnnData,
    model,
    vectors: List[CytokineVector],
    source_cytokine: str = "IFN-beta",
    target_cytokine: str = "IL-6",
    max_cells_per_stratum: int = 5000,
) -> anndata.AnnData:
    """
    Generate virtual target-cytokine cells from source-cytokine cells.
    Returns an AnnData containing only virtual cells.
    """
    if "X_scvi" not in adata.obsm:
        raise ValueError("adata.obsm['X_scvi'] is required")

    obs_records = []
    latent_blocks = []
    expr_blocks = []

    for vec in vectors:
        mask_source = (adata.obs["cell_type"] == vec.cell_type) & (
            adata.obs["cytokine_type"] == source_cytokine
        )
        if vec.donor_id != "all":
            mask_source &= adata.obs["donor_id"] == vec.donor_id
        idx = np.where(mask_source)[0]
        if idx.size == 0:
            continue
        if idx.size > max_cells_per_stratum:
            idx = np.random.choice(idx, size=max_cells_per_stratum, replace=False)

        z_src = np.asarray(adata.obsm["X_scvi"][idx])
        shift = vec.v_il6_minus_ifn if target_cytokine == "IL-6" else vec.v_ifn_minus_il6
        z_tgt = z_src + shift

        decoded = _decode_latent(model, z_tgt)
        expr_blocks.append(decoded)
        latent_blocks.append(z_tgt)

        obs_block = adata.obs.iloc[idx].copy()
        obs_block["cytokine_type_real"] = source_cytokine
        obs_block["cytokine_type_virtual"] = target_cytokine
        obs_block["is_counterfactual"] = True
        obs_block["source_index"] = idx
        obs_records.append(obs_block)

    if not expr_blocks:
        raise ValueError("No counterfactual cells were generated; check input strata.")

    expr = np.vstack(expr_blocks)
    latent = np.vstack(latent_blocks)
    obs = pd.concat(obs_records, axis=0)
    obs.index = [f"cf_{i}" for i in range(expr.shape[0])]

    adata_virtual = anndata.AnnData(
        X=expr,
        obs=obs,
        var=adata.var.copy(),
        dtype=expr.dtype,
    )
    adata_virtual.layers["normalized"] = expr
    adata_virtual.obsm["X_scvi"] = latent
    return adata_virtual


def merge_real_and_virtual(
    adata_real: anndata.AnnData,
    adata_virtual: anndata.AnnData,
    target_cytokine: str = "IL-6",
) -> anndata.AnnData:
    """Concatenate real target-cytokine cells with counterfactual cells."""
    mask_real = (
        (adata_real.obs["cytokine_type"] == target_cytokine)
        if "cytokine_type" in adata_real.obs
        else np.ones(adata_real.n_obs, dtype=bool)
    )
    real_subset = adata_real[mask_real].copy()
    real_subset.obs["is_counterfactual"] = False
    return anndata.concat([real_subset, adata_virtual], join="outer", label="source")
