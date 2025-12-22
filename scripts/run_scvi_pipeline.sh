#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${1:-data/raw/cytokine_dictionary.h5ad}
OUTPUT_DIR=${2:-results/scvi_analysis}
CONFIG=${3:-configs/model/scvi.yaml}

python scripts/run_scvi_pipeline.py \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --config "${CONFIG}"
