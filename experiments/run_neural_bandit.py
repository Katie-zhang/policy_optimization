import argparse
import collections
from typing import List
import numpy as np
import math
import torch.cuda
import torch
import torch.nn as nn
from utils.collect_data import collect_preference_data
from torch.utils.tensorboard import SummaryWriter
import os

from copy import deepcopy

from algos.neural_bandit.mle import RewardModel, MaximumLikelihoodEstimator
from algos.neural_bandit.pg import (
    PolicyModel,
    PolicyGradientOptimizer,
    UniformPolicyModel,
)
from algos.neural_bandit.dpo import DirectPreferenceOptimizer
from envs.neural_bandit import NeuralBandit
from utils.io_utils import save_code, save_config, create_log_dir
from utils.logger import Logger



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="neural_bandit")
    parser.add_argument("--state_dim", type=int, default=1)
    parser.add_argument("--actions", type=np.ndarray, default=[-10, 0, 10])

    parser.add_argument("--agent", type=str, default="pg")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--logdir", type=str, default="log")

    parser.add_argument("--pref_data_num", type=int, default=50)
    parser.add_argument("--mle_num_iters", type=int, default=20)

    parser.add_argument("--rl_data_ratio", type=float, default=4)
    parser.add_argument("--reg_coef", type=float, default=1.0)
    parser.add_argument("--pg_num_iters", type=int, default=50)

    return parser.parse_args()


def main(args=parse_args()):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    log_dir = create_log_dir(args)
    save_code(log_dir)
    save_config(args.__dict__, log_dir)

    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    state_dim = args.state_dim
    actions = args.actions

    
    env = NeuralBandit(
        state_dim,
        actions,     
        device=device,
        num_trials_for_eval=5000,
    )

    # opt_policy = env.get_opt_policy()
    uniform_policy = UniformPolicyModel(len(actions), device)
    
    # Collect preference data
    pref_dataset, p_list = collect_preference_data(env)

    logger.info(f"Preference data: ")
    for i, transition in enumerate(pref_dataset):
        logger.info(
            f"Transition {i+1}: state={transition.state}, action_0={transition.action_0}, action_1={transition.action_1}, "
            f" pref={transition.pref}, "
            f"chosen_probs={transition.chosen_probs:.4f}"
        )
    logger.info(f"P: ")
    logger.info(p_list)
    
    learned_reward_model = RewardModel(
        state_dim,
        actions,
        is_target=False,
        hidden_dim=64,
        num_layers=2,
        device=device,
    )

    mle_learner = MaximumLikelihoodEstimator(
        actions,
        learned_reward_model,
        learning_rate=1e-3,
        batch_size=64,
        logger=logger,
    )

    action_to_index = {-10: 0, 0: 1, 10: 2}

    states = torch.cat([torch.tensor(x.state) for x in pref_dataset], dim=0)
    
    positive_actions = torch.cat(
        [torch.tensor(action_to_index[x.action_1] if x.pref == 1 else action_to_index[x.action_0]).unsqueeze(0) for x in pref_dataset],
        dim=0
    )
    
    negative_actions = torch.cat(
        [torch.tensor(action_to_index[x.action_0] if x.pref == 1 else action_to_index[x.action_1]).unsqueeze(0) for x in pref_dataset],
        dim=0
    )

    mle_learner.optimize(
        states, positive_actions, negative_actions, num_epochs=args.mle_num_iters
    )

    # Train policy on preference data
    logger.info("========Train on preference data [DPO]===========")
    policy = PolicyModel(
        state_dim,
        actions,
        hidden_dim=64,
        num_layers=2,
        device=device,
    )
    policy2 = deepcopy(policy)
    policy2.load_state_dict(policy.state_dict())
    policy3 = deepcopy(policy)
    policy3.load_state_dict(policy.state_dict())

    learned_env = NeuralBandit(
        state_dim,
        actions,
        num_trials_for_eval=5000,
        device=device,
    )

    dpo = DirectPreferenceOptimizer(
        policy,
        ref_policy=uniform_policy,
        env=env,
        learned_env=learned_env,
        learning_rate=1e-3,
        batch_size=64,
        beta=args.reg_coef,
        logger=logger,
    )
    
    states = torch.cat([torch.tensor(x.state) for x in pref_dataset], dim=0).to(device)
    positive_actions = torch.cat(
        [torch.tensor(action_to_index[x.action_1] if x.pref == 1 else action_to_index[x.action_0]).unsqueeze(0) for x in pref_dataset],
        dim=0
    ).to(device)
    negative_actions = torch.cat(
        [torch.tensor(action_to_index[x.action_0] if x.pref == 1 else action_to_index[x.action_1]).unsqueeze(0) for x in pref_dataset],
        dim=0
    ).to(device)

    dpo.optimize(
        states,
        positive_actions,
        negative_actions,
        num_epochs=args.pg_num_iters,
    )
   

    # RMB-PO
    logger.info("========Train on preference data [RMB-PO]===========")
    pg = PolicyGradientOptimizer(
        policy2,
        ref_policy=uniform_policy,
        env=env,
        learned_env=learned_env,
        learning_rate=1e-3,
        batch_size=64,
        beta=args.reg_coef,
        logger=logger,
    )
    pg.optimize(
        states,
        positive_actions,
        negative_actions,
        num_epochs=args.pg_num_iters,
    )




if __name__ == "__main__":
    main()
