from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.celltyping.annotate import annotate_cell_types
from src.celltyping.plots import plot_cell_type_counts, plot_cell_type_umap
from src.cytokine_vectors.counterfactuals import (
    generate_counterfactuals,
    merge_real_and_virtual,
)
from src.cytokine_vectors.evaluation import (
    compute_gene_level_agreement,
    compute_pathway_agreement,
    identify_nonlinear_genes,
)
from src.cytokine_vectors.plots import (
    plot_gene_scatter,
    plot_latent_overlay,
    plot_pathway_bars,
    plot_pathway_heatmap,
)
from src.cytokine_vectors.utils import load_adata, load_scvi_model
from src.cytokine_vectors.vectors import (
    compute_cytokine_vectors,
    save_cytokine_vectors,
)


def _default_pathways() -> dict:
    return {
        "IL6_JAK_STAT": [
            "STAT1",
            "STAT3",
            "SOCS1",
            "SOCS3",
            "JAK1",
            "JAK2",
            "IL6ST",
        ],
        "IFN_beta_ISG": [
            "IFI44L",
            "IFI44",
            "IFIT1",
            "IFIT3",
            "MX1",
            "ISG15",
            "OAS1",
            "OAS2",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cytokine response vectors and counterfactuals on scVI latent space"
    )
    parser.add_argument(
        "--adata_path",
        default="data/processed/scp_cytokine_with_scvi.h5ad",
        help="Path to AnnData with X_scvi",
    )
    parser.add_argument(
        "--model_dir",
        default="models/scvi_scp_cytokine",
        help="Directory containing trained scVI model",
    )
    parser.add_argument(
        "--output_dir",
        default="results/scp_cytokine_vectors",
        help="Directory for outputs",
    )
    parser.add_argument("--min_cells", type=int, default=50)
    parser.add_argument("--max_cells_per_stratum", type=int, default=5000)
    parser.add_argument("--source_cytokine", default="IFN-beta")
    parser.add_argument("--target_cytokine", default="IL-6")
    parser.add_argument(
        "--annotate_cell_types",
        action="store_true",
        help="Run marker-based cell type annotation before vectors",
    )
    parser.add_argument("--celltype_method", default="marker_score")
    parser.add_argument("--skip_plots", action="store_true", help="Skip plotting to avoid backend issues")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = load_adata(args.adata_path)
    model = load_scvi_model(args.model_dir, adata)

    # Cell type annotation if requested or missing
    if args.annotate_cell_types or ("cell_type" not in adata.obs) or (
        adata.obs["cell_type"].nunique() <= 1
    ):
        adata = annotate_cell_types(
            adata,
            method=args.celltype_method,
            min_score_diff=0.1,
            unknown_label="unknown",
            overwrite=True,
        )
        annotated_path = output_dir / "adata_with_celltypes.h5ad"
        adata.write(annotated_path)
        try:
            plot_cell_type_counts(adata, str(output_dir / "cell_type_counts.png"))
            plot_cell_type_umap(adata, str(output_dir / "cell_type_umap.png"))
        except Exception as plot_err:
            print(f"Skipping cell type QC plots due to error: {plot_err}")

    vectors = compute_cytokine_vectors(
        adata, min_cells=args.min_cells, source_cytokine=args.source_cytokine, target_cytokine=args.target_cytokine
    )
    vectors_path = output_dir / "cytokine_vectors.json"
    save_cytokine_vectors(vectors, vectors_path)

    adata_virtual = generate_counterfactuals(
        adata=adata,
        model=model,
        vectors=vectors,
        source_cytokine=args.source_cytokine,
        target_cytokine=args.target_cytokine,
        max_cells_per_stratum=args.max_cells_per_stratum,
    )
    virtual_path = output_dir / f"virtual_{args.source_cytokine}_to_{args.target_cytokine}.h5ad"
    adata_virtual.write(virtual_path)

    merged = merge_real_and_virtual(
        adata_real=adata,
        adata_virtual=adata_virtual,
        target_cytokine=args.target_cytokine,
    )
    merged_path = output_dir / "real_plus_virtual.h5ad"
    merged.write(merged_path)

    # Plot latent overlay
    if not args.skip_plots:
        try:
            plot_latent_overlay(
                adata_real=adata,
                adata_virtual=adata_virtual,
                output_path=str(output_dir / "latent_real_vs_counterfactual.png"),
            )
        except Exception as plot_err:
            print(f"Skipping latent overlay plot due to error: {plot_err}")

    pathway_db = _default_pathways()
    pathway_heatmap_input = {}
    summary = {"cell_types": []}

    for cell_type in sorted(adata.obs["cell_type"].unique()):
        try:
            gene_stats = compute_gene_level_agreement(
                adata_real=adata,
                adata_virtual=adata_virtual,
                cell_type=cell_type,
                target_cytokine=args.target_cytokine,
            )
        except ValueError:
            continue

        gene_stats_path = output_dir / f"gene_stats_{cell_type}.csv"
        gene_stats.to_csv(gene_stats_path, index=False)

        pathway_stats = compute_pathway_agreement(gene_stats, pathway_db)
        pathway_stats_path = output_dir / f"pathway_stats_{cell_type}.csv"
        pathway_stats.to_csv(pathway_stats_path, index=False)
        pathway_heatmap_input[cell_type] = pathway_stats

        nonlinear = identify_nonlinear_genes(gene_stats)
        nonlinear_path = output_dir / f"nonlinear_genes_{cell_type}.csv"
        nonlinear.to_csv(nonlinear_path, index=False)

        if not args.skip_plots:
            try:
                plot_gene_scatter(
                    gene_stats,
                    output_path=str(output_dir / f"gene_scatter_{cell_type}.png"),
                    nonlinear_genes=nonlinear["gene"].head(20).tolist(),
                )
            except Exception as plot_err:
                print(f"Skipping gene scatter plot for {cell_type}: {plot_err}")
            try:
                plot_pathway_bars(
                    pathway_stats,
                    output_path=str(output_dir / f"pathway_{cell_type}.png"),
                )
            except Exception as plot_err:
                print(f"Skipping pathway plot for {cell_type}: {plot_err}")

        summary["cell_types"].append(
            {
                "cell_type": cell_type,
                "n_real": int((adata.obs["cell_type"] == cell_type).sum()),
                "n_virtual": int((adata_virtual.obs["cell_type"] == cell_type).sum()),
            }
        )

    if not args.skip_plots:
        try:
            plot_pathway_heatmap(
                pathway_heatmap_input,
                output_path=str(output_dir / "pathway_heatmap.png"),
            )
        except Exception as plot_err:
            print(f"Skipping pathway heatmap: {plot_err}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Completed counterfactual analysis. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
