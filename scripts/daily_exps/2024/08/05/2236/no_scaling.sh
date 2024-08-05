#!/bin/bash
# bash ./scripts/daily_exps/2024/08/05/2236/no_scaling.sh 1 infobatch 0.1 none
# bash ./scripts/daily_exps/2024/08/05/2236/no_scaling.sh 3 infobatch 0.1 well
# bash ./scripts/daily_exps/2024/08/05/2236/no_scaling.sh 5 infobatch 0.1 all
# bash ./scripts/daily_exps/2024/08/05/2236/no_scaling.sh 6 full 1.0 none
# bash ./scripts/daily_exps/2024/08/05/2236/no_scaling.sh 7 full 1.0 all

PROJECT=20240805_PDA

DEVICE=${1}
PRUNING_METHOD=${2}
RATIO=${3}
AUGMENT_METHOD=${4}

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
    --batch-size 32 \
    --workers 4 \
    --experiment ${PROJECT} \
    --pruning-method ${PRUNING_METHOD} \
    --pruning-ratio ${RATIO} \
    --augment-method ${AUGMENT_METHOD} \
    --log-wandb \

