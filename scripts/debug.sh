#!/bin/bash
# bash ./scripts/debug.sh


# torchrun --nproc_per_node=4 main.py \
#     --data-dir /SSD/ILSVRC2012 \
#     --dataset imagenet \
#     --model resnet18 \
#     --sched cosine \
#     --epochs 150 \
#     --warmup-epochs 5 \
#     --lr 0.4 \
#     --reprob 0.5 \
#     --remode pixel \
#     --batch-size 256 \
#     --amp \
#     --workers 4 \

CUDA_VISIBLE_DEVICES=0 python main.py \
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
    --batch-size 32 \
    --workers 4 \
    --experiment 20240805_DEBUG \
    --pruning-method full \
    --augment-method all \


# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --data-dir /SSD/CIFAR100 \
#     --dataset torch/cifar100 \
#     --model resnet18 \
#     --sched cosine \
#     --epochs 150 \
#     --warmup-epochs 5 \
#     --lr 0.4 \
#     --reprob 0.5 \
#     --remode pixel \
#     --batch-size 32 \
#     --workers 4 \

