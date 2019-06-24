#!/bin/sh

python train.py --task_name ukp-topic-sentence --do_train --use_all_data=1 --seed 1 --do_lower_case --binarize_labels 0 --data_dir ./datasets/ukp/data/complete/ --bert_model bert-base-uncased --max_seq_length 64 --train_batch_size 16 --test_set "" --learning_rate 2e-5 --num_train_epochs 2.0 --output_dir "bert_output/ukp/bert-base-topic-sentence/all_data/"




