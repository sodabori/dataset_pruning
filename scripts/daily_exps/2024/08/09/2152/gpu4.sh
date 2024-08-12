#!/bin/bash
# bash ./scripts/daily_exps/2024/08/09/2152/gpu4.sh

DEVICE=4
PROJECT=20240809_PDA
RATIO=0.1
AUGMENT_METHOD=well

bash ./scripts/infobatch_no_scaling.sh ${DEVICE} ${PROJECT} ${RATIO} ${AUGMENT_METHOD}

THRESHOLD=2.5

bash ./scripts/infoloss_no_scaling.sh ${DEVICE} ${PROJECT} ${RATIO} ${AUGMENT_METHOD} ${THRESHOLD}