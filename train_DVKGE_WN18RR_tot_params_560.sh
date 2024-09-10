#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 560 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 300 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 2000 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 1

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 280 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 300 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 2000 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 2

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 112 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 300 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 2000 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 5

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 80 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 300 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 2000 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 7

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 40 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 300 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 2000 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 14

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 28 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 300 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 2000 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 20

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 16 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 300 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 2000 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 35

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 4 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 300 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 2000 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 63

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 7 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 300 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 2000 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 80

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 4 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 300 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 2000 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 140

