#!/bin/bash
# bash ./scripts/daily_exps/2024/08/09/2152/gpu0.sh

DEVICE=0
PROJECT=20240809_PDA
RATIO=0.1
AUGMENT_METHOD=none

bash ./scripts/infobatch_gradient_scaling.sh ${DEVICE} ${PROJECT} ${RATIO} ${AUGMENT_METHOD}

THRESHOLD=2.5

bash ./scripts/infoloss_gradient_scaling.sh ${DEVICE} ${PROJECT} ${RATIO} ${AUGMENT_METHOD} ${THRESHOLD}