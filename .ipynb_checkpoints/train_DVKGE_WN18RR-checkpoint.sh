#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 80 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 500 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 2000 \
                                        --neg_sample_size 50 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 7