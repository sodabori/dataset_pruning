#!/bin/bash
# bash ./scripts/debug.sh 0

PROJECT=DEBUG

DEVICE=${1}
PRUNING_METHOD=infobatch
RATIO=0.00001
AUGMENT_METHOD=none

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