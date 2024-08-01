#!/bin/bash
# bash ./scripts/main.sh


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
    --sched cosine \
    --epochs 150 \
    --warmup-epochs 5 \
    --lr 0.4 \
    --reprob 0.5 \
    --remode pixel \
    --batch-size 32 \
    --workers 4 \

