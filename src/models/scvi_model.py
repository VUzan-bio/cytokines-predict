"""scVI model tailored for cytokine stimulation analysis."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import scvi
from anndata import AnnData

logger = logging.getLogger(__name__)


class CytokineSCVI(scvi.model.SCVI):
    """scVI model optimized for cytokine stimulation analysis."""

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 20,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        dispersion: str = "gene-batch",
        categorical_covariate_keys: Optional[Iterable[str]] = None,
        batch_key: str = "donor_id",
    ):
        super().__init__(
            adata,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood="zinb",
            latent_distribution="normal",
        )

    @classmethod
    def setup_anndata(cls, adata, **kwargs):
        categorical_covariate_keys = kwargs.get(
            "categorical_covariate_keys",
            ["cytokine_type", "concentration", "stimulation_duration"],
        )
        batch_key = kwargs.get("batch_key", "donor_id")

        if "counts" not in adata.layers:
            raise ValueError("AnnData missing 'counts' layer required for scVI.")

        # Ensure required covariate columns exist
        for key in categorical_covariate_keys:
            if key not in adata.obs:
                adata.obs[key] = "unknown"
            adata.obs[key] = adata.obs[key].astype("category")

        if batch_key not in adata.obs:
            adata.obs[batch_key] = "batch_unknown"
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")

        for key in list(adata.uns):
            if str(key).startswith("_scvi"):
                del adata.uns[key]

        logger.info("Running CytokineSCVI.setup_anndata with batch_key=%s", batch_key)
        super().setup_anndata(
            adata,
            **{
                "layer": "counts",
                "batch_key": batch_key,
                "categorical_covariate_keys": categorical_covariate_keys,
                **{k: v for k, v in kwargs.items() if k not in {"layer", "batch_key", "categorical_covariate_keys"}},
            },
        )


def build_model_from_config(adata: AnnData, config) -> CytokineSCVI:
    model_cfg = getattr(config, "model", config)
    CytokineSCVI.setup_anndata(adata)
    return CytokineSCVI(
        adata,
        n_latent=getattr(model_cfg, "n_latent", 20),
        n_hidden=getattr(model_cfg, "n_hidden", 128),
        n_layers=getattr(model_cfg, "n_layers", 2),
        dropout_rate=getattr(model_cfg, "dropout_rate", 0.1),
        dispersion=getattr(model_cfg, "dispersion", "gene-batch"),
    )
