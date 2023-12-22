#! /bin/bash
# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="gs://v5_path/"

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR="gs://improved-t5/t5x/stable_t5"
T5X_DIR="/home/natha/t5x"  # directory where the T5X repo is cloned.
# # TFDS_DATA_DIR="..."
PROJECT_DIR="/home/natha/stable_t5x/t5x/examples/scalable_t5/t5_1_1/stable_t5"
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file=base_stable_t5_pretrain.gin \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}