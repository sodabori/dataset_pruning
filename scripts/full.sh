#!/bin/bash
# bash ./scripts/full.sh 0 20240809_PDA none
# bash ./scripts/full.sh 0 20240809_PDA all

DEVICE=${1}
PROJECT=${2}
PRUNING_METHOD=full
RATIO=1.0
AUGMENT_METHOD=${3}

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
    --aa rand-m9-mstd0.5 \
    --log-wandb \

