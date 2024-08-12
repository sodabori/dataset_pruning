#!/bin/bash
# bash ./scripts/infoloss_no_scaling.sh 0 20240809_PDA 0.1 none 2.0
# bash ./scripts/infoloss_no_scaling.sh 0 20240809_PDA 0.1 well 2.0
# bash ./scripts/infoloss_no_scaling.sh 0 20240809_PDA 0.1 all 2.0
# bash ./scripts/infoloss_no_scaling.sh 7 20240809_PDA 0.5 none 2.0

DEVICE=${1}
PROJECT=${2}
PRUNING_METHOD=infoloss
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
    --aa rand-m9-mstd0.5 \
    --threshold ${THRESHOLD} \
    --log-wandb \

