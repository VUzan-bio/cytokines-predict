# Scaling the cytokine dictionary on higher-RAM machines

The current environment cannot load the full SCP (SCP2554) matrix or all GSE202186 GSMs. To extend the project to full cytokine and cell-type resolution, run these steps on a machine with ample RAM, then bring back compact artifacts.

## One-time preprocessing on a larger machine
1) **Download raw data**
   - SCP: counts (`matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`) + `metadata.txt.gz`.
   - Optional extra GSMs from GSE202186 (10x triples) for additional cytokines.

2) **Load + QC + HVG**
   - `scanpy.read_10x_mtx` on the full SCP directory (or combined GSMs).
   - Mito QC, min genes/cells, normalize_total, log1p, HVGs (2k).

3) **Attach metadata**
   - Join per-cell metadata (`cytokine`, `cell_type`, `donor_id`/`mouse_id`).
   - Ensure `obs` has `cytokine_type`, `cell_type`, `donor_id`.

4) **Train scVI**
   - `batch_key=donor_id`, `categorical_covariate_keys=['cytokine_type','cell_type']`.
   - CPU or GPU; latent 20, hidden 128, layers 2, dropout 0.1; epochs ~50–100.

5) **Save artifacts to share back**
   - `data/processed/scp_cytokine_with_scvi.h5ad` (compressed, includes `X_scvi`).
   - `models/scvi_scp_cytokine/` (scVI saved model).

## Using artifacts in this repo
1) Copy the two artifacts into this repo.
2) Run the cytokine-vector pipeline (plots optional if backend issues):
   ```
   $env:PYTHONPATH='.'
   python scripts/run_cytokine_vectors.py ^
     --adata_path data/processed/scp_cytokine_with_scvi.h5ad ^
     --model_dir models/scvi_scp_cytokine ^
     --output_dir results/scp_cytokine_vectors ^
     --min_cells 50 ^
     --source_cytokine IFN-beta ^
     --target_cytokine IL-6 ^
     --skip_plots
   ```
3) For additional cytokine pairs, adjust `--source_cytokine/--target_cytokine` and rerun.

## Optional: controlled downsampling
If full SCP is still too large, downsample on the bigger machine to ~50–100k cells while preserving cytokine and cell-type diversity, save as `scp_cytokine_subset_with_scvi.h5ad`, and reuse the same pipeline here.
