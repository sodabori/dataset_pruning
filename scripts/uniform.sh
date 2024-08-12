#!/bin/bash
# bash ./scripts/uniform.sh 0 DEBUG 0.45 well 0.2
# bash ./scripts/uniform.sh 3 20240809_PDA 0.4 none 0.0
# bash ./scripts/uniform.sh 4 20240809_PDA 0.4 well 0.2
# bash ./scripts/uniform.sh 5 20240809_PDA 0.4 all 1.0

DEVICE=${1}
PROJECT=${2}
PRUNING_METHOD=uniform
RATIO=${3}
AUGMENT_METHOD=${4}
AUGMENT_RATIO=${5}

START_EPOCH=-1
END_EPOCH=90

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
    --augment-ratio ${AUGMENT_RATIO} \
    --augment-method ${AUGMENT_METHOD} \
    --aa rand-m9-mstd0.5 \
    --pruning-start-epoch ${START_EPOCH} \
    --pruning-end-epoch ${END_EPOCH} \
    --log-wandb \
