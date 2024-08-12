#!/bin/bash
# bash ./scripts/daily_exps/2024/08/09/2152/gpu7.sh

DEVICE=6
PROJECT=20240809_PDA
RATIO=0.1
AUGMENT_METHOD=all

bash ./scripts/full.sh ${DEVICE} ${PROJECT} ${AUGMENT_METHOD}