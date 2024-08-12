#!/bin/bash
# bash ./scripts/daily_exps/2024/08/09/1238/gradient_scaling.sh 7 infoloss 0.1 none 2.5

PROJECT=20240807_PDA

DEVICE=${1}
PRUNING_METHOD=${2}
RATIO=${3}
AUGMENT_METHOD=${4}
THRESHOLD=${5}

CUDA_VISIBLE_DEVICES=${DEVICE} python main.py \
    --data-dir /SSD/ILSVRC2012 \
    --dataset imagenet \
    --model resnet18 \
    --sched step \
    --smoothing 0. \
    --decay-epochs 30 \
    --decay-rate 0.1 \
    --epochs 90 \
    --warmup-epochs 0 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --batch-size 256 \
    --workers 8 \
    --experiment ${PROJECT} \
    --pruning-method ${PRUNING_METHOD} \
    --pruning-ratio ${RATIO} \
    --augment-method ${AUGMENT_METHOD} \
    --aa original \
    --rescaling \
    --threshold ${THRESHOLD} \
    --log-wandb \

