from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Optional, Tuple

import anndata
import numpy as np
import pandas as pd


def _ensure_jax_stub() -> None:
    """Stub jax/jaxlib/flax modules when not available (common on Windows CPUs)."""
    try:
        import jaxlib.xla_extension  # type: ignore
    except Exception:
        jaxlib_mod = types.ModuleType("jaxlib")
        xla_extension = types.ModuleType("jaxlib.xla_extension")
        xla_extension.Device = object
        version_mod = types.ModuleType("jaxlib.version")
        version_mod.__version__ = "0.7.2"
        sys.modules["jaxlib"] = jaxlib_mod
        sys.modules["jaxlib.xla_extension"] = xla_extension
        sys.modules["jaxlib.version"] = version_mod
        jaxlib_mod.xla_extension = xla_extension
        jaxlib_mod.version = version_mod

    # Minimal jax stub to avoid dependency on real jax/jaxlib
    if "jax" not in sys.modules:
        def device_put(x, device=None):
            return x

        def devices(kind=None):
            return [None]

        class _Random:
            @staticmethod
            def PRNGKey(i):
                return i

        class _ShapedArray:
            def __init__(self, shape=None, dtype=None, **kwargs):
                self.shape = shape
                self.dtype = dtype

        jax_module = types.ModuleType("jax")
        jax_module.device_put = device_put
        jax_module.device_get = lambda x: x
        jax_module.devices = devices
        jax_module.random = _Random()
        jax_module.numpy = np  # type: ignore
        jax_module.jit = lambda fn, *args, **kwargs: fn
        jax_module.Array = object
        sys.modules["jax"] = jax_module

        random_module = types.ModuleType("jax.random")
        random_module.PRNGKey = _Random.PRNGKey
        sys.modules["jax.random"] = random_module
        sys.modules["jax.numpy"] = np

        core_module = types.ModuleType("jax.core")
        core_module.ShapedArray = _ShapedArray
        sys.modules["jax.core"] = core_module

        lax_module = types.ModuleType("jax.lax")
        lax_module.scan = lambda f, init, xs: (init, xs)
        sys.modules["jax.lax"] = lax_module

        sharding_module = types.ModuleType("jax.sharding")
        class _PartitionSpec:
            def __init__(self, *args, **kwargs):
                self.spec = args
        sharding_module.PartitionSpec = _PartitionSpec
        sys.modules["jax.sharding"] = sharding_module

        extend_module = types.ModuleType("jax.extend")
        lu_module = types.ModuleType("jax.extend.linear_util")
        extend_module.linear_util = lu_module
        sys.modules["jax.extend"] = extend_module
        sys.modules["jax.extend.linear_util"] = lu_module

        interpreters_module = types.ModuleType("jax.interpreters")
        pe_module = types.ModuleType("jax.interpreters.partial_eval")
        interpreters_module.partial_eval = pe_module
        sys.modules["jax.interpreters"] = interpreters_module
        sys.modules["jax.interpreters.partial_eval"] = pe_module
        tree_util_module = types.ModuleType("jax.tree_util")
        class _PyTreeDef:
            pass
        tree_util_module.PyTreeDef = _PyTreeDef
        tree_util_module.tree_flatten = lambda x: ([], None)
        tree_util_module.tree_unflatten = lambda treedef, leaves: leaves
        sys.modules["jax.tree_util"] = tree_util_module

    # Minimal flax stub
    if "flax" not in sys.modules:
        import dataclasses

        linen_module = types.ModuleType("flax.linen")
        class _DummyLayer:
            def __init__(self, *args, **kwargs):
                pass
        linen_module.Module = object
        linen_module.Dense = _DummyLayer
        linen_module.Dropout = _DummyLayer
        linen_module.LayerNorm = _DummyLayer
        init_module = types.ModuleType("flax.linen.initializers")
        init_module.variance_scaling = lambda *args, **kwargs: None
        sys.modules["flax.linen"] = linen_module
        sys.modules["flax.linen.initializers"] = init_module
        flax_module = types.ModuleType("flax")
        struct_module = types.ModuleType("flax.struct")
        struct_module.dataclass = dataclasses.dataclass
        training_module = types.ModuleType("flax.training")
        train_state_module = types.ModuleType("flax.training.train_state")

        class _TrainState:
            pass

        train_state_module.TrainState = _TrainState
        training_module.train_state = train_state_module

        flax_module.struct = struct_module
        flax_module.training = training_module
        flax_module.linen = linen_module
        linen_module.initializers = init_module

        sys.modules["flax"] = flax_module
        sys.modules["flax.struct"] = struct_module
        sys.modules["flax.training"] = training_module
        sys.modules["flax.training.train_state"] = train_state_module

    # Minimal numpyro stub to satisfy optional imports
    if "numpyro" not in sys.modules:
        numpyro_module = types.ModuleType("numpyro")
        distributions_module = types.ModuleType("numpyro.distributions")

        class _Distribution:
            pass

        class _Constraints:
            positive = object()
            real = object()
            nonnegative_integer = object()
            unit_interval = object()

        class _NegativeBinomial2:
            pass

        distributions_module.Distribution = _Distribution
        distributions_module.constraints = _Constraints()
        distributions_module.NegativeBinomial2 = _NegativeBinomial2
        util_module = types.ModuleType("numpyro.distributions.util")
        util_module.promote_shapes = lambda *args, **kwargs: args
        util_module.validate_sample = lambda *args, **kwargs: None
        sys.modules["numpyro.distributions.util"] = util_module
        numpyro_module.distributions = distributions_module

        sys.modules["numpyro"] = numpyro_module
        sys.modules["numpyro.distributions"] = distributions_module

    # Minimal optax stub to avoid heavy JAX deps
    if "optax" not in sys.modules:
        optax_module = types.ModuleType("optax")
        class _GT:
            pass
        optax_module.GradientTransformation = _GT
        optax_module.contrib = types.ModuleType("optax.contrib")
        sys.modules["optax"] = optax_module
        sys.modules["optax.contrib"] = optax_module.contrib


def load_adata(path: str) -> anndata.AnnData:
    """Load AnnData and ensure required columns exist and are strings."""
    adata_path = Path(path)
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found at {adata_path}")

    adata = anndata.read_h5ad(adata_path)

    required = ["cytokine_type", "donor_id", "cell_type"]
    for col in required:
        if col not in adata.obs:
            # Fall back to placeholder to allow downstream processing
            adata.obs[col] = "unknown"
        adata.obs[col] = adata.obs[col].astype(str)

    return adata


def load_scvi_model(model_dir: str, adata: anndata.AnnData) -> scvi.model.SCVI:
    """Load a trained SCVI model and attach to the provided AnnData."""
    _ensure_jax_stub()
    import scvi

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"SCVI model directory not found at {model_path}")

    try:
        from src.models.scvi_model import CytokineSCVI
        loader_cls = CytokineSCVI
    except Exception:
        loader_cls = scvi.model.SCVI

    try:
        model = loader_cls.load(model_path, adata=adata)
    except Exception:
        model = scvi.model.SCVI.load(model_path, adata=adata)
    return model


def _count_by_stratum(adata: anndata.AnnData) -> pd.DataFrame:
    """Return counts per (cell_type, donor_id, cytokine_type)."""
    return (
        adata.obs.groupby(["cell_type", "donor_id", "cytokine_type"])
        .size()
        .rename("n")
        .reset_index()
    )


def get_valid_strata(
    adata: anndata.AnnData, min_cells: int = 100
) -> pd.DataFrame:
    """Strata with at least min_cells per cytokine."""
    if "cell_type" not in adata.obs:
        raise ValueError("adata.obs must contain 'cell_type'")
    counts = _count_by_stratum(adata)
    pivot = counts.pivot_table(
        index=["cell_type", "donor_id"],
        columns="cytokine_type",
        values="n",
        fill_value=0,
    )
    pivot = pivot.rename_axis(None, axis=1).reset_index()
    pivot["n_ifn"] = pivot.get("IFN-beta", 0)
    pivot["n_il6"] = pivot.get("IL-6", 0)
    valid = pivot[(pivot["n_ifn"] >= min_cells) & (pivot["n_il6"] >= min_cells)]
    return valid[["cell_type", "donor_id", "n_ifn", "n_il6"]]
