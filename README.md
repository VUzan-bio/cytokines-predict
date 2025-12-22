# A single-cell cytokine dictionary of human peripheral blood (scVI-only)

Laptop-friendly, reproducible pipeline for building a cytokine response dictionary using scVI as the sole modeling backbone.

## Quickstart
- Create a Python 3.10+ environment.
- Install CPU-only PyTorch for maximum compatibility:
  ```
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```
- Install dependencies:
  ```
  pip install -r requirements/base.txt
  ```
- Stream data (example GEO accession placeholder):
  ```
  python scripts/download_cytokine_data.py --accession GSEXXXXXX --raw_dir data/raw
  ```
- Run the full pipeline:
  ```
  python scripts/run_scvi_pipeline.py \
    --data_path data/raw/cytokine_dictionary.h5ad \
    --output_dir results/scvi_analysis \
    --config configs/model/scvi.yaml
  ```

## What you get
- Memory-efficient preprocessing (mitochondrial filtering, lightweight doublet detection, HVG selection).
- scVI model configured with cytokine covariates (type, concentration, duration) and donor batches.
- Training tuned for laptops (CPU-only, small hidden size, early stopping).
- Evaluation: ARI/NMI, silhouette by cytokine, reconstruction error, batch-mixing entropy, expression recovery.
- Interpretation: DE contrasts (stimulated vs control), latent traversals, gene loadings, pseudotime.
- Visuals: UMAPs colored by cell type/cytokine/donor plus training curves.

## Runtime and resources
- Expected runtime: ~2–4 hours on a 16 GB RAM laptop for a full-sized dataset; minutes for the toy test.
- Peak memory: ~8 GB during preprocessing and HVG selection; scVI training fits comfortably on CPU with batch size 256.
- Disk: raw downloads dominate; processed AnnData objects and checkpoints live in `models/checkpoints` and `data/processed`.

## Configuration
- Model/training defaults live in `configs/model/scvi.yaml`.
- Dataset metadata in `configs/data/dataset.yaml` (GEO accession, columns, streaming settings).
- Override paths and config via CLI flags on `scripts/run_scvi_pipeline.py`.

## Project layout
- `src/data`: download (GEO/SRA streaming) and preprocessing for scVI (QC, HVGs, stratified splits).
- `src/models`: scVI wrapper and cytokine response predictor on latent space.
- `src/training`: training loop with early stopping tuned for CPU.
- `src/evaluation`: metrics (clustering, silhouette, batch mixing, reconstruction, expression recovery).
- `src/interpretation`: DE, latent traversal, loadings, pseudotime.
- `src/visualization`: UMAPs and training curves.
- `scripts`: pipeline runner and data download helper.
- `notebooks`: ordered exploration notebooks (stubs ready for use).
- `tests`: lightweight end-to-end smoke test with synthetic data.

## Troubleshooting
- **Out of memory during preprocessing:** Lower `data.hvg_genes` and `training.batch_size` in `configs/model/scvi.yaml`; ensure swap/virtual memory is available.
- **Slow training:** Reduce `training.max_epochs` and `training.kl_warmup_epochs`; start from a subsampled AnnData to validate the pipeline.
- **Missing GPU:** The defaults already force CPU (`use_gpu=False`); if you add a GPU, set CUDA-enabled PyTorch and pass `use_gpu=True` in `train_scvi_model`.
- **Empty or sparse metadata:** The preprocessing step backfills missing cytokine covariates with "unknown"; update `configs/data/dataset.yaml` to match your columns.

## Testing
- Run the fast smoke test:
  ```
  pytest -q tests/test_scvi_pipeline.py
  ```
- The test trains for two epochs on synthetic data to verify that preprocessing, training, and latent extraction work end-to-end.
