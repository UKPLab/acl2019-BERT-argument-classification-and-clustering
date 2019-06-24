#!/bin/sh
python train.py --task_name ukp_aspects --do_train --seed 1 --do_eval --do_lower_case --data_dir ./datasets/ukp_aspect/splits/ --train_file "all_data.tsv"  --bert_model bert-base-uncased --max_seq_length 64 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir "./bert_output/ukp_aspects_all"


