#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 960 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 500 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 1500 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 1

CUDA_VISIBLE_DEVICES=1 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 480 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 500 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 1500 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 2

CUDA_VISIBLE_DEVICES=1 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 160 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 500 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 1500 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 6

CUDA_VISIBLE_DEVICES=1 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 96 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 500 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 1500 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 10

CUDA_VISIBLE_DEVICES=1 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 40 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 500 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 1500 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 24

CUDA_VISIBLE_DEVICES=1 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 24 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 500 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 1500 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 40

CUDA_VISIBLE_DEVICES=1 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 15 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 500 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 1500 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 64

CUDA_VISIBLE_DEVICES=1 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 6 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 500 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 1500 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 160

CUDA_VISIBLE_DEVICES=1 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 4 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 500 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 1500 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 240

CUDA_VISIBLE_DEVICES=1 python run.py    --dataset WN18RR \
                                        --model DVKGE \
                                        --rank 3 \
                                        --regularizer N3 \
                                        --reg 0.0 \
                                        --optimizer Adam \
                                        --max_epochs 500 \
                                        --patience 15 \
                                        --valid 5 \
                                        --batch_size 1500 \
                                        --neg_sample_size 10 \
                                        --init_size 0.001 \
                                        --learning_rate 0.001 \
                                        --gamma 0.0 \
                                        --bias learn \
                                        --dtype double \
                                        --double_neg \
                                        --multi_c \
                                        --n_space 320

