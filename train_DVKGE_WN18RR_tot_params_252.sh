#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 252 \
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
                                        --rank 126 \
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
                                        --rank 84 \
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
                                        --n_space 3

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 42 \
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
                                        --n_space 6

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 21 \
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
                                        --n_space 12

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 12 \
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
                                        --n_space 21

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 6 \
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
                                        --n_space 42

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
                                        --rank 3 \
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
                                        --n_space 84

CUDA_VISIBLE_DEVICES=0 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 2 \
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
                                        --n_space 126

