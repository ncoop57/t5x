#! /bin/bash
# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="gs://v5_path/"

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR="gs://improved-t5/t5x/models"
T5X_DIR="/home/natha/t5x"  # directory where the T5X repo is cloned.
# # TFDS_DATA_DIR="..."

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file=t5x/examples/t5/t5_1_1/stable_t5/base_stable_t5_pretrain.gin \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}