#! /bin/bash
# Data dir to save the processed dataset in "gs://data_dir" format.
BUCKET_ROOT_DIR="gs://improved-t5/t5x"
TFDS_DATA_DIR="gs://stablelm-datasets/starcoder-raw-tfds"
# PROJECT_DIR=${HOME}"/dir1/user_dir"

# Make sure that dataset package is up-to-date.
# python3 -m pip install --upgrade tfds-nightly

# Pre-download dataset.
# tfds build wmt_t2t_translate

# gsutil cp -r /home/natha/tensorflow_datasets gs://improved-t5/t5x/data/

# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR="gs://improved-t5/t5x/models"
T5X_DIR="/home/natha/t5x"  # directory where the T5X repo is cloned.
# # TFDS_DATA_DIR="..."
PROJECT_DIR="/home/natha/t5x/t5x/examples/t5/t5_1_1/stable_code"
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="base_stable_code_pretrain.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}