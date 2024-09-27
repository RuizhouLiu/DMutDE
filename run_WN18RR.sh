#!/bin/bash
python run.py   --dataset WN18RR \
                --model EnsembleModel \
                --regularizer N3 \
                --reg 0.0 \
                --optimizer Adam \
                --max_epochs 300 \
                --patience 100 \
                --valid 5 \
                --batch_size 500 \
                --neg_sample_size 250 \
                --init_size 0.001 \
                --learning_rate 0.0005 \
                --gamma 0.0 \
                --bias learn \
                --dtype double \
                --double_neg \
                --multi_c \
                --global_model RefH \
                --global_model_rank 32 \
                --local_model RotH \
                --local_model_rank 32 \
                --KD_iter 40 \
                --KD_SLF_weight 1.0 \
                --KD_EED_weight 0.2 \
                --KD_omega1 0.6 \
                --KD_omega2 0.4 \
                --cuda_id 0