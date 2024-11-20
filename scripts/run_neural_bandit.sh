#!/bin/bash

set -e
set -x

STATE_DIM=1
ACTION_NUM=10
PREF_DATA_NUM=50

for seed in 5 
do
    python -m experiments.run_neural_bandit \
    --state_dim ${STATE_DIM} \
    --pref_data_num ${PREF_DATA_NUM} \
    --rl_data_ratio 0.5 \
    --reg_coef 0.01 \
    --seed ${seed} \
    --logdir "log"
done
