import numpy as np
import scanpy as sc
from types import SimpleNamespace

from src.data.preprocess import preprocess_for_scvi
from src.training.train_scvi import train_scvi_model


def _config():
    return SimpleNamespace(
        model=SimpleNamespace(
            n_latent=10,
            n_hidden=64,
            n_layers=1,
            dropout_rate=0.1,
            dispersion="gene-batch",
        ),
        training=SimpleNamespace(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=64,
            max_epochs=2,
            patience=2,
            kl_warmup_epochs=1,
            train_size=0.8,
            validation_size=0.1,
            random_seed=0,
        ),
        data=SimpleNamespace(
            hvg_genes=200,
            mito_prefix="MT-",
            min_genes=10,
            max_genes=1000,
            max_mito=0.3,
            doublet_score_threshold=0.8,
            donor_key="donor_id",
            cytokine_keys=SimpleNamespace(
                cytokine_type="cytokine_type",
                cytokine_concentration="cytokine_concentration",
                stimulation_duration="stimulation_duration",
            ),
        ),
    )


def create_test_adata(n_cells: int = 128, n_genes: int = 300) -> sc.AnnData:
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.2, size=(n_cells, n_genes))
    adata = sc.AnnData(counts)
    adata.var_names = [f"GENE{i}" for i in range(n_genes - 2)] + ["MT-ND1", "MT-CO1"]
    adata.obs["donor_id"] = rng.choice(["D1", "D2", "D3"], size=n_cells)
    adata.obs["cytokine_type"] = rng.choice(["control", "stimulated"], size=n_cells)
    adata.obs["cytokine_concentration"] = rng.choice(["low", "high"], size=n_cells)
    adata.obs["stimulation_duration"] = rng.choice(["1h", "6h"], size=n_cells)
    adata.obs["cell_type"] = rng.choice(["T", "B"], size=n_cells)
    return adata


def test_scvi_pipeline_runs():
    adata = create_test_adata()
    config = _config()
    adata = preprocess_for_scvi(adata, config)
    model = train_scvi_model(adata, config)

    latent = model.get_latent_representation(adata)
    assert latent.shape == (adata.n_obs, config.model.n_latent)
    recon_error = model.get_reconstruction_error(adata=adata)
    assert np.isfinite(recon_error)
