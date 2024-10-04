import collections
import numpy as np
import math
from typing import List
from envs.linear_bandit import LinearBandit

Transition = collections.namedtuple(
    "Transition", ["state", "action_0", "action_1", "reward_0", "reward_1", "pref", "chosen_probs"]
)


def sigmoid(x: float):
    return 1.0 / (1.0 + math.exp(-x))


def ret_uniform_policy(action_num: int = 0):
    assert action_num > 0, "The number of actions should be positive."

    def uniform_policy(state: np.ndarray = None):
        action_prob = np.full(shape=action_num, fill_value=1.0 / action_num)
        return action_prob

    return uniform_policy


def collect_preference_data(
    num: int, env: LinearBandit, policy_func
) -> List[Transition]:
    pref_dataset = []
    action_num = env.action_num
    for _ in range(num):
        state = env.reset()
        action_prob = policy_func(state)
        sampled_actions = np.random.choice(
            a=action_num, size=2, replace=False, p=action_prob  # replace=True
        )
        action_one, action_two = sampled_actions[0], sampled_actions[1]
        reward_one, reward_two = env.sample(action_one), env.sample(action_two)

        bernoulli_param = sigmoid(reward_two - reward_one)
        # pref=1 means that the second action is preferred over the first one
        pref = np.random.binomial(1, bernoulli_param, 1)[0]
        
        # define p(action_pref > action_nonpref|x)
        if pref == 0:
            score = env.score(action_one, action_two)   
        elif pref == 1:
            score = env.score(action_two, action_one)   
        chosen_probs = sigmoid(score)
        
        transition = Transition(
            state, action_one, action_two, reward_one, reward_two, pref, chosen_probs
        )
        pref_dataset.append(transition)
    return pref_dataset


def collect_rl_data(num: int, env) -> List[float]:
    rl_dataset = []
    for _ in range(num):
        state = env.reset()
        rl_dataset.append(state)

    return rl_dataset


def merge_datasets(pref_dataset: List[Transition], rl_dataset: List[float]):
    merged_rl_dataset = rl_dataset
    for transition in pref_dataset:
        state = transition.state
        merged_rl_dataset.append(state)

    return merged_rl_dataset


def pref_to_rl(pref_dataset: List[Transition]):
    rl_dataset = []
    for transition in pref_dataset:
        state = transition.state
        rl_dataset.append(state)

    return rl_dataset
