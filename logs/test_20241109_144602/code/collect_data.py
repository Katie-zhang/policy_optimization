import collections
import numpy as np
import math
from typing import List
import torch
import torch.nn as nn
from envs.linear_bandit import LinearBandit

torch.manual_seed(5) 
Transition = collections.namedtuple(
    "Transition", ["state", "action_0", "action_1", "pref", "chosen_probs"]
)


def sigmoid(x: float):
    return 1.0 / (1.0 + math.exp(-x))


def ret_uniform_policy(actions: np.ndarray):
    
    def uniform_policy(state: np.ndarray = None):
        action_prob = np.full(shape=len(actions), fill_value=1.0 / len(actions))
        return action_prob

    return uniform_policy


class NonMonotonicScalarToVectorNN(nn.Module):
    def __init__(self):
        super(NonMonotonicScalarToVectorNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)     # Input layer (1 -> 10)
        self.fc2 = nn.Linear(10, 20)    # Hidden layer (10 -> 20)
        self.fc3 = nn.Linear(20, 2)     # Output layer (20 -> 2)      
    def forward(self, x):
        x = torch.tanh(self.fc1(x))      # Using sin activation for non-monotonicity
        x = torch.tanh(self.fc2(x))     # Tanh activation adds more non-linearity
        x = self.fc3(x)                 # Output layer
        return x
        

def get_score(action_one,action_two,feature_func):
    feature_one = feature_func(torch.tensor([[action_one]], dtype=torch.float))[0].detach().numpy()
    feature_two = feature_func(torch.tensor([[action_two]], dtype=torch.float))[0].detach().numpy()
    score_param =  np.array([[0.0, -1.0],[1, 0]], np.float32) 
    score = feature_one@score_param@feature_two
    return score

def get_p(action_one,action_two,feature_func):
    feature_one = feature_func(torch.tensor([[action_one]], dtype=torch.float))[0].detach().numpy()
    feature_two = feature_func(torch.tensor([[action_two]], dtype=torch.float))[0].detach().numpy()
    score_param =  np.array([[0.0, -1.0],[1, 0]], np.float32) 
    score = feature_one@score_param@feature_two 
    p = sigmoid(score)
    return p

        
        
def collect_preference_data(
    env: LinearBandit
) -> List[Transition]:
    pref_dataset = []
    actions = env.actions
    cur_state = np.array([0])   
    feature_func = NonMonotonicScalarToVectorNN()
    
    p_list = np.zeros([len(actions),len(actions)])
    for i in range(len(actions)):
        for j in range(len(actions)):
            action_one = actions[i]
            action_two = actions[j]
            p = get_p(action_one,action_two,feature_func)
            if i==j:
                assert p==0.5
            p_list[i][j] = p
            
            if i==j:
                continue
            else:
                transition = Transition(
                cur_state, action_one, action_two, 1, p
                )
                pref_dataset.append(transition)
                transition = Transition(
                cur_state, action_two, action_one, 0, p 
                )
                pref_dataset.append(transition)
           
    return pref_dataset,p_list


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
