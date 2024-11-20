#!/bin/bash

set -e
set -x

PREF_DATA_NUM=20
PG_NUM_ITERS=1000
REG_COEF=0.01
STATE_DIM=1

for seed in 2021 
do
    python -m experiments.run_linear_bandit_copy \
    --state_dim ${STATE_DIM} \
    --pref_data_num ${PREF_DATA_NUM} \
    --sppo_adaptive \
    --seed ${seed} \
    --flip_feature \
    --logdir "log"
done