#!/bin/bash
export GLUE_DIR=glue_data
export TASK_NAME=MRPC
export OUTPUT_DIR=./${TASK_NAME}_no_label_embedding
# export log_file=tmp/train_log_${TASK_NAME}.txt

python ./scripts/run_training_bert.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --pretrained_model bert-base-uncased \
    --task_name $TASK_NAME \
    --train_file data/medabstracts/train.csv \
    --validation_file data/medabstracts/test.csv \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 256 \
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

