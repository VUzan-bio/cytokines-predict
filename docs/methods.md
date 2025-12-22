# Methods: scVI-only cytokine dictionary

## Generative model
- scVI assumes counts `x_{ng}` arise from a ZINB distribution conditioned on latent `z_n` and library size `l_n`.
- Encoder: amortized inference network parameterizing `q(z_n | x_n)` with `n_layers` feed-forward blocks and dropout.
- Decoder: gene-level dispersion (`dispersion="gene-batch"`) to capture cytokine- and donor-specific variance; likelihood `p(x_n | z_n, l_n)` uses softplus-activated mean and learned dropout logits.
- Priors: standard normal on `z_n`, log-normal on `l_n`; KL annealed over `kl_warmup_epochs` for stable training.

## Cytokine covariates
- `cytokine_type`, `concentration`, `stimulation_duration` encoded as categorical covariates in `setup_anndata`.
- `donor_id` used as `batch_key` to learn shared cytokine manifolds while correcting donor effects.
- Layers: `counts` (raw), `normalized` (library-size normalized), `log1p` (log-transformed working matrix).

## Preprocessing
- QC: mitochondrial fraction (`MT-` prefix), gene-count bounds, lightweight doublet heuristic (proxy for scDblFinder).
- Normalization: `scanpy.pp.normalize_total` then `sc.pp.log1p`.
- HVGs: `scanpy.pp.highly_variable_genes` with `n_top_genes=2000`, `flavor="seurat_v3"`, subsetting to reduce memory.
- Splits: `StratifiedShuffleSplit` on donor × cytokine condition to keep evaluation balanced.

## Training (CPU-friendly)
- Hidden size 128, 2 layers, latent dim 20; dropout 0.1.
- Optimizer via scVI `TrainingPlan` with AdamW (`lr=1e-3`, `weight_decay=1e-4`).
- Early stopping on `elbo_validation` with patience 20; validation every epoch; `train_size=0.8`, `validation_size=0.1`.
- KL warmup over 40 epochs to avoid posterior collapse on small laptops.

## Evaluation metrics
- **Latent quality:** reconstruction error, cytokine-condition silhouette on `X_scvi`.
- **Clustering:** Leiden on `X_scvi`; ARI/NMI vs. annotated cell types.
- **Batch mixing:** neighbor entropy proxy for kBET/LISI using donor labels.
- **Expression recovery:** Pearson correlation of normalized vs. reconstructed expression on held-out cells.
- **DE:** scVI `differential_expression` contrasting stimulated vs. control per cytokine type.

## Interpretation
- Latent traversal: interpolate between control and stimulated centroids to map response trajectories.
- Gene loadings: correlation-based ranking of genes per latent dimension (top 15).
- Pseudotime: Euclidean distance from control centroid in latent space as cytokine response intensity.
