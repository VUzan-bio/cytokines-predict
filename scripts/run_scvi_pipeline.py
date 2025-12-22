"""End-to-end execution script for the scVI cytokine dictionary pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.preprocess import load_cytokine_data, preprocess_for_scvi
from src.evaluation.evaluate import evaluate_scvi_model
from src.interpretation.interpret import run_differential_expression
from src.training.train_scvi import train_scvi_model
from src.utils.config import load_config, namespace_to_dict
from src.visualization.plots import create_scvi_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="scVI Cytokine Dictionary Pipeline")
    parser.add_argument("--data_path", required=True, help="Path to raw h5ad file.")
    parser.add_argument("--output_dir", default="results/scvi", help="Directory to store outputs.")
    parser.add_argument("--config", default="configs/model/scvi.yaml", help="YAML config for scVI model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    adata = load_cytokine_data(args.data_path)
    adata = preprocess_for_scvi(adata, config)

    model = train_scvi_model(adata, config)
    adata.obsm["X_scvi"] = model.get_latent_representation(adata)

    metrics = evaluate_scvi_model(model, adata)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    try:
        de_results = run_differential_expression(
            model,
            adata,
            groupby="cytokine_type",
            group1="IL-6",
            group2="IFN-beta",
        )
        de_results.to_csv(output_dir / "de_il6_vs_ifnb.csv", index=False)
    except Exception as exc:
        with open(output_dir / "de_error.log", "w", encoding="utf-8") as handle:
            handle.write(f"DE computation failed: {exc}\n")

    create_scvi_plots(model, adata, output_dir)
    model.save(output_dir / "model", overwrite=True)

    # h5py is strict about mixed/object dtypes; coerce obs to strings where needed
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str)

    adata.write(output_dir / "adata_with_scvi.h5ad")

    with open(output_dir / "config_used.json", "w", encoding="utf-8") as handle:
        json.dump(namespace_to_dict(config), handle, indent=2)


if __name__ == "__main__":
    main()
