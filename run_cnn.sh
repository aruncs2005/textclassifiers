#!/bin/bash

export OUTPUT_DIR=./cnn_based_training
# export log_file=tmp/train_log_${TASK_NAME}.txt

python ./scripts/cnn_classification.py \
    --max_seq_length 128 \
    --train_file glue_data/medabstracts/train.csv \
    --validation_file glue_data/medabstracts/test.csv \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR \
    --save_steps 10000 \
    --logging_steps 10 \
# 2>&1 | tee -a $log_file
# mv $log_file $OUTPUT_DIR 

#tmp/ELECTRA_lamb/checkpoint-1500/checkpoint-d-1500\

