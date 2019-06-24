#!/bin/sh

for testset in "abortion" "cloning" "death penalty" "gun control" "marijuana legalization" "minimum wage" "nuclear energy" "school uniforms"
do
    python train.py --task_name ukp-topic-sentence --do_train --do_eval --seed 1 --do_lower_case --binarize_labels 0 --data_dir ./datasets/ukp/data/complete/ --bert_model bert-base-uncased --max_seq_length 64 --train_batch_size 16 --test_set "$testset" --learning_rate 2e-5 --num_train_epochs 2.0 --output_dir "bert_output/ukp/bert-base-topic-sentence/${testset}_test_topic/"

done



