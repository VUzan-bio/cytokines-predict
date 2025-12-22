from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import anndata
import numpy as np
import pandas as pd

from .utils import get_valid_strata


@dataclass
class CytokineVector:
    cell_type: str
    donor_id: str
    n_ifn: int
    n_il6: int
    mu_ifn: np.ndarray
    mu_il6: np.ndarray
    v_il6_minus_ifn: np.ndarray
    v_ifn_minus_il6: np.ndarray

    def to_jsonable(self) -> dict:
        payload = asdict(self)
        for key in ["mu_ifn", "mu_il6", "v_il6_minus_ifn", "v_ifn_minus_il6"]:
            payload[key] = payload[key].tolist()
        return payload

    @staticmethod
    def from_jsonable(data: dict) -> "CytokineVector":
        fields = {}
        for key in ["cell_type", "donor_id", "n_ifn", "n_il6"]:
            fields[key] = data[key]
        for key in ["mu_ifn", "mu_il6", "v_il6_minus_ifn", "v_ifn_minus_il6"]:
            fields[key] = np.asarray(data[key])
        return CytokineVector(**fields)


def compute_cytokine_vectors(
    adata: anndata.AnnData, min_cells: int = 100
) -> List[CytokineVector]:
    """Compute IL-6 vs IFN-beta latent vectors per (cell_type, donor_id).

    Falls back to a single global vector if no donor-level strata qualify.
    """
    if "X_scvi" not in adata.obsm:
        raise ValueError("adata.obsm['X_scvi'] is required")

    valid = get_valid_strata(adata, min_cells=min_cells)
    vectors: List[CytokineVector] = []

    for _, row in valid.iterrows():
        cell_type = row["cell_type"]
        donor_id = row["donor_id"]

        mask_stratum = (adata.obs["cell_type"] == cell_type) & (
            adata.obs["donor_id"] == donor_id
        )
        stratum = adata[mask_stratum]

        mask_ifn = stratum.obs["cytokine_type"] == "IFN-beta"
        mask_il6 = stratum.obs["cytokine_type"] == "IL-6"

        z_ifn = np.asarray(stratum.obsm["X_scvi"][mask_ifn])
        z_il6 = np.asarray(stratum.obsm["X_scvi"][mask_il6])

        if z_ifn.shape[0] < min_cells or z_il6.shape[0] < min_cells:
            continue

        mu_ifn = z_ifn.mean(axis=0)
        mu_il6 = z_il6.mean(axis=0)
        v_il6_minus_ifn = mu_il6 - mu_ifn
        v_ifn_minus_il6 = mu_ifn - mu_il6

        vectors.append(
            CytokineVector(
                cell_type=cell_type,
                donor_id=donor_id,
                n_ifn=z_ifn.shape[0],
                n_il6=z_il6.shape[0],
                mu_ifn=mu_ifn,
                mu_il6=mu_il6,
                v_il6_minus_ifn=v_il6_minus_ifn,
                v_ifn_minus_il6=v_ifn_minus_il6,
            )
        )

    # Fallback: global vector if nothing passed the filter
    if not vectors:
        mask_ifn = adata.obs["cytokine_type"] == "IFN-beta"
        mask_il6 = adata.obs["cytokine_type"] == "IL-6"
        z_ifn = np.asarray(adata.obsm["X_scvi"][mask_ifn])
        z_il6 = np.asarray(adata.obsm["X_scvi"][mask_il6])
        if z_ifn.size == 0 or z_il6.size == 0:
            return []
        mu_ifn = z_ifn.mean(axis=0)
        mu_il6 = z_il6.mean(axis=0)
        vectors.append(
            CytokineVector(
                cell_type="unknown",
                donor_id="all",
                n_ifn=z_ifn.shape[0],
                n_il6=z_il6.shape[0],
                mu_ifn=mu_ifn,
                mu_il6=mu_il6,
                v_il6_minus_ifn=mu_il6 - mu_ifn,
                v_ifn_minus_il6=mu_ifn - mu_il6,
            )
        )

    return vectors


def save_cytokine_vectors(vectors: List[CytokineVector], path: str) -> None:
    """Serialize cytokine vectors to JSON."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = [v.to_jsonable() for v in vectors]
    pd.Series(payload).to_json(path_obj, orient="values", indent=2)


def load_cytokine_vectors(path: str) -> List[CytokineVector]:
    """Load cytokine vectors from JSON."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Cytokine vector file not found at {path_obj}")
    data = pd.read_json(path_obj, typ="series")
    return [CytokineVector.from_jsonable(item) for item in data.tolist()]


def get_vector_for_cell(
    vectors: List[CytokineVector], cell_type: str, donor_id: str
) -> Optional[CytokineVector]:
    """Return vector matching cell_type and donor_id."""
    for vec in vectors:
        if vec.cell_type == cell_type and vec.donor_id == donor_id:
            return vec
    return None
