import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import yaml
from algos.linear_bandit.mle import MLERewardLearning
from algos.linear_bandit.pg import PolicyGradient
from algos.linear_bandit.dpo import DirectPolicyOptimization
from algos.linear_bandit.sppo import SelfPlayPreferenceOptimization
from envs.linear_bandit import LinearBandit, ret_feature_func
from utils.io_utils import save_code, save_config, create_log_dir
from utils.logger import Logger
from utils.collect_data import (
    ret_uniform_policy,
    collect_preference_data,
    collect_rl_data,
    merge_datasets,
    pref_to_rl,
)
from utils.plot import plot_model_accuracies

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="linear_bandit")
    parser.add_argument("--state_dim", type=int, default=1)
    parser.add_argument("--action_num", type=int, default=4)

    parser.add_argument("--agent", type=str, default="pg")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--flip_feature", action="store_true")

    parser.add_argument("--pref_data_num", type=int, default=500)
    parser.add_argument("--mle_num_iters", type=int, default=100)
    parser.add_argument("--mle_adaptive", action="store_true")
    parser.add_argument("--mle_ada_coef", type=float, default=1.0)
    parser.add_argument("--mle_step_size", type=float, default=0.1)
    
    parser.add_argument("--reg_coef", type=float, default=1.0)
    parser.add_argument("--dpo_num_iters", type=int, default=200)
    parser.add_argument("--dpo_adaptive", action="store_true")
    parser.add_argument("--dpo_ada_coef", type=float, default=1.0)
    parser.add_argument("--dpo_step_size", type=float, default=0.1)

    
    parser.add_argument("--sppo_num_iters", type=int, default=1000)
    parser.add_argument("--sppo_adaptive", action="store_true")
    parser.add_argument("--sppo_ada_coef", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--sppo_step_size", type=float, default=0.005)

    parser.add_argument("--pg_num_iters", type=int, default=1000)
    parser.add_argument("--pg_adaptive", action="store_true")
    parser.add_argument("--pg_ada_coef", type=float, default=1.0)
    parser.add_argument("--pg_step_size", type=float, default=0.1)

    return parser.parse_args()


def get_reward_func(reward_param: np.ndarray, feature_func):
    def reward_func(state, action):
        feature = feature_func(state, action)
        rew = np.dot(feature, reward_param)

        return rew

    return reward_func


def main(args):
    np.random.seed(args.seed)
    log_dir = create_log_dir(args)
    save_code(log_dir)
    save_config(args.__dict__, log_dir)

    logger = Logger(log_dir)
    writer = SummaryWriter(log_dir)

    state_dim = args.state_dim
    action_num = args.action_num

    feature_dim = 2 * args.state_dim
    num_trials_for_eval = 10000
    feature_func = ret_feature_func(num_action=action_num, state_dim=state_dim)

    reward_param = np.array([1.0, 2.0], np.float32)
    
    score_param = np.array([1.0, 2.0, 1.0, 2.0], np.float32)
    env = LinearBandit(
        state_dim,
        action_num,
        reward_param,
        score_param,
        feature_func,
        num_trials_for_eval=num_trials_for_eval,
    )
    opt_policy = env.get_opt_policy()
    opt_reward = env.evaluate_reward(policy=opt_policy)

    uniform_policy = ret_uniform_policy(action_num)
    unif_policy_rew = env.evaluate_reward(policy=uniform_policy)
    pref_data,P = collect_preference_data(args.pref_data_num, env, uniform_policy)
    logger.info(f"Preference data: ")
    for i, transition in enumerate(pref_data):
        logger.info(
            f"Transition {i+1}: state={transition.state}, action_0={transition.action_0}, action_1={transition.action_1}, "
            f" pref={transition.pref}, "
            f"chosen_probs={transition.chosen_probs:.4f}"
        )
    logger.info(f"P: ")
    logger.info(P)
    
    logger.info(
        f"optimal policy reward: {opt_reward: .4f}, uniform policy reward: {unif_policy_rew: .4f}."
    )

    # learn the reward function
    reward_model = MLERewardLearning(
        feature_func,
        feature_dim,
        args.mle_step_size,
        args.mle_num_iters,
        args.mle_adaptive,
        args.mle_ada_coef,
    )
    loss, l2_dist, acc = reward_model.train_by_cvxpy(
        dataset=pref_data, true_reward_param=reward_param
    )
    logger.info(f"Reward loss: {loss:.4f}, l2 distance: {l2_dist:.4f}, acc: {acc:.2f}.")

    learned_reward_func = reward_model.get_reward_func
    learned_reward_param = reward_model.get_reward_param
    logger.info("True reward parameter: {}".format(reward_param))
    logger.info("Learned reward parameter: {}".format(learned_reward_param))

    learned_env = LinearBandit(
        state_dim,
        action_num,
        learned_reward_param,
        score_param,
        feature_func,
        num_trials_for_eval=num_trials_for_eval,
    )

    # learn the policy
    policy_feature_func = ret_feature_func(
        num_action=action_num, state_dim=state_dim, is_flip=args.flip_feature
    )
    
    # RMB-PO
    logger.info(
        f"Train a policy on the preference data with policy-generated data (RMB-PO)."
    )
    agent = PolicyGradient(
        policy_feature_func,
        learned_reward_func,
        uniform_policy,
        feature_dim,
        action_num,
        args.reg_coef,
        args.pg_step_size,
        args.pg_num_iters,
        args.pg_adaptive,
        args.pg_ada_coef,
        logger=logger,
    )

    reward,RMB_PO_accuracy = agent.train(dataset=pref_data, env=env, learned_env=learned_env)
    rew_error = float(opt_reward - reward)
    policy_param = agent.get_param
    logger.info(f"Policy parameter (RMB-PO): {policy_param}.")
    logger.info(
        f"Training solely on the preference data (RMB-PO), dataset size: {len(pref_data): d}, optimal reward: {opt_reward: .4f}, reward: {reward: .4f}, reward error: {rew_error: .4f}."
    )
    rew_err_dict, rew_dict = dict(), dict()
    rew_err_dict[args.pref_data_num] = rew_error
    rew_dict[args.pref_data_num] = float(reward)
    save_path = os.path.join(log_dir, "reward_error_rmb_po.yml")
    yaml.dump(rew_err_dict, open(save_path, "w"), default_flow_style=False)
    save_path = os.path.join(log_dir, "reward_rmb_po.yml")
    yaml.dump(rew_dict, open(save_path, "w"), default_flow_style=False)
    
    

     # Train the RL on the preference data
    agent = DirectPolicyOptimization(
        state_dim=state_dim,
        action_num=action_num,
        feature_dim=feature_dim,
        feature_func=policy_feature_func,
        ref_policy=uniform_policy,
        reg_coef=args.reg_coef,
        step_size=args.dpo_step_size,
        num_iters=args.dpo_num_iters,
        is_adaptive=args.dpo_adaptive,
        ada_coef=args.dpo_ada_coef,
        logger=logger,
    )

    reward,dpo_accuracy = agent.train(dataset=pref_data, env=env)
    rew_error = float(opt_reward - reward)
    policy_param = agent.get_param
    logger.info(
        f"Policy parameter learned solely on the preference data (DPO): {policy_param}."
    )
    logger.info(
        f"Training solely on the preference data (DPO), dataset size: {len(pref_data): d}, optimal reward: {opt_reward: .4f}, reward: {reward: .4f}, reward error: {rew_error: .4f}."
    )
    rew_err_dict, rew_dict = dict(), dict()
    rew_err_dict[args.pref_data_num] = rew_error
    rew_dict[args.pref_data_num] = float(reward)
    save_path = os.path.join(log_dir, "reward_error_dpo.yml")
    yaml.dump(rew_err_dict, open(save_path, "w"), default_flow_style=False)
    save_path = os.path.join(log_dir, "reward_dpo.yml")
    yaml.dump(rew_dict, open(save_path, "w"), default_flow_style=False)
    
    #SPPO
    agent = SelfPlayPreferenceOptimization(
        state_dim=state_dim,
        action_num=action_num,
        feature_dim=feature_dim,
        feature_func=policy_feature_func,
        beta=args.beta,
        step_size=args.sppo_step_size,
        num_iters=args.sppo_num_iters,
        is_adaptive=args.sppo_adaptive,
        ada_coef=args.sppo_ada_coef,
        logger=logger,
    )
        
    reward,sppo_accuracy = agent.train(dataset=pref_data, env=env)
    rew_error = float(opt_reward - reward)
    policy_param = agent.get_param
    logger.info(
        f"Policy parameter learned solely on the preference data (SPPO): {policy_param}."
    )
    logger.info(
        f"Training solely on the preference data (SPPO), dataset size: {len(pref_data): d}, optimal reward: {opt_reward: .4f}, reward: {reward: .4f}, reward error: {rew_error: .4f}."
    )
    rew_err_dict, rew_dict = dict(), dict()
    rew_err_dict[args.pref_data_num] = rew_error
    rew_dict[args.pref_data_num] = float(reward)
    save_path = os.path.join(log_dir, "reward_error_sppo.yml")
    yaml.dump(rew_err_dict, open(save_path, "w"), default_flow_style=False)
    save_path = os.path.join(log_dir, "reward_sppo.yml")
    yaml.dump(rew_dict, open(save_path, "w"), default_flow_style=False)
    
    model_names = ['SPPO', 'DPO','RMB_PO']
    accuracies = [sppo_accuracy, dpo_accuracy,RMB_PO_accuracy]
    plot_model_accuracies(model_names, accuracies, os.path.join(log_dir, "model_accuracies.png"))
    


if __name__ == "__main__":
    main(parse_args())
