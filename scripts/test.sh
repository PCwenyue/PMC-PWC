#!/bin/bash

# model and checkpoint
MODEL=$1
CHECKPOINT=$2

# datasets
DATASET_NAME=$3
DATASET_HOME=$4

# training configuration
python main.py \
--evaluation=False \
--test=True \
--cuda=$7 \
--checkpoint=$CHECKPOINT \
--model=$MODEL \
--validation_dataset=$DATASET_NAME \
--validation_dataset_root=$DATASET_HOME \
--test_output_path=$5 \
--test_save_occ=$6

