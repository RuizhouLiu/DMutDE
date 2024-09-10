#!/bin/bash
python run.py   --dataset FB237 \
                --model EnsembleModel \
                --rank 500 \
                --regularizer N3 \
                --reg 0.0 \
                --optimizer Adagrad \
                --max_epochs 500 \
                --patience 100 \
                --valid 5 \
                --batch_size 1000 \
                --neg_sample_size 50 \
                --init_size 0.001 \
                --learning_rate 0.01 \
                --gamma 0.0 \
                --bias learn \
                --dtype double \
                --multi_c \
                --KGE_weight 0.7 \
                --pretained