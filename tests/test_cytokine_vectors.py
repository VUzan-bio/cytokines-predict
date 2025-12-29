import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cytokine_vectors.counterfactuals import generate_counterfactuals
from src.cytokine_vectors.vectors import compute_cytokine_vectors


class DummyModel:
    device = "cpu"

    def __init__(self, n_genes: int):
        self.n_genes = n_genes

    def get_normalized_expression(self, adata=None, latent_z=None, **kwargs):
        # Return zeros with correct gene dimension to mimic decoder output
        return np.zeros((latent_z.shape[0], self.n_genes))

    class module:
        @staticmethod
        def decoder(z, library, batch_index):
            # Match shape: (cells, genes)
            return {"px_scale": z}


def _make_toy_adata(n_cells=40, n_genes=5):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_cells, n_genes))
    adata = anndata.AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs["cytokine_type"] = (
        ["IFN-beta"] * (n_cells // 4)
        + ["IL-6"] * (n_cells // 4)
        + ["IFN-beta"] * (n_cells // 4)
        + ["IL-6"] * (n_cells - 3 * n_cells // 4)
    )
    adata.obs["donor_id"] = ["d1"] * (n_cells // 2) + ["d2"] * (n_cells - n_cells // 2)
    adata.obs["cell_type"] = ["T"] * n_cells
    adata.obsm["X_scvi"] = rng.normal(size=(n_cells, 3))
    return adata


def test_compute_vectors_and_counterfactuals():
    adata = _make_toy_adata()
    vectors = compute_cytokine_vectors(adata, min_cells=5)
    assert vectors, "Expected non-empty cytokine vectors"

    model = DummyModel(n_genes=adata.n_vars)
    adata_virtual = generate_counterfactuals(
        adata=adata,
        model=model,
        vectors=vectors,
        source_cytokine="IFN-beta",
        target_cytokine="IL-6",
        max_cells_per_stratum=10,
    )
    assert adata_virtual.n_obs > 0
    assert "is_counterfactual" in adata_virtual.obs
    assert "cytokine_type_virtual" in adata_virtual.obs
    assert (adata_virtual.var_names == adata.var_names).all()
    assert adata_virtual.obsm["X_scvi"].shape[0] == adata_virtual.n_obs
