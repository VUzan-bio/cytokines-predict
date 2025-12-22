"""Training loop for the cytokine scVI model."""

from __future__ import annotations

import logging

import sys
import types

# Some Windows wheels of jaxlib miss the xla_extension module; add a minimal stub
try:  # pragma: no cover - environment-specific guard
    import jaxlib

    if not hasattr(jaxlib, "xla_extension"):
        stub = types.SimpleNamespace(Device=object)
        sys.modules.setdefault("jaxlib.xla_extension", stub)
except Exception:
    pass

import scvi

from src.models.scvi_model import CytokineSCVI, build_model_from_config

logger = logging.getLogger(__name__)


def train_scvi_model(adata, config) -> CytokineSCVI:
    """Train scVI with laptop-optimized settings."""
    training_cfg = getattr(config, "training", config)
    scvi.settings.seed = getattr(training_cfg, "random_seed", 42)

    model = build_model_from_config(adata, config)
    logger.info("Starting scVI training on CPU")
    model.train(
        max_epochs=getattr(training_cfg, "max_epochs", 100),
        batch_size=getattr(training_cfg, "batch_size", 256),
        plan_kwargs={
            "lr": getattr(training_cfg, "learning_rate", 1e-3),
            "weight_decay": getattr(training_cfg, "weight_decay", 1e-4),
            "n_epochs_kl_warmup": getattr(training_cfg, "kl_warmup_epochs", 40),
        },
        check_val_every_n_epoch=1,
        train_size=getattr(training_cfg, "train_size", 0.8),
        validation_size=getattr(training_cfg, "validation_size", 0.1),
        early_stopping=True,
        early_stopping_monitor="elbo_validation",
        early_stopping_patience=getattr(training_cfg, "patience", 20),
        accelerator="cpu",
        devices=1,
    )
    return model
