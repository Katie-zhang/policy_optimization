import collections
import itertools
from typing import List
import numpy as np
import math
import torch.cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from utils.logger import Logger

    

from utils.io_utils import save_code
from utils.logger import Logger

torch.manual_seed(5)

Transition = collections.namedtuple(
    "Transition", ["state", "action_0", "action_1", "pref", "chosen_probs"]
)


def sigmoid(x: float):
    return 1.0 / (1.0 + math.exp(-x))

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
        

def get_score(action_one,action_two, p_list):   
    score = p_list[action_one][action_two]
    return score

def get_p(action_one,action_two,feature_func):
    feature_one = feature_func(torch.tensor([[action_one]], dtype=torch.float))[0].detach().numpy()
    feature_two = feature_func(torch.tensor([[action_two]], dtype=torch.float))[0].detach().numpy()
    score_param =  np.array([[0.0, -1.0],[1, 0]], np.float32) 
    score = feature_one@score_param@feature_two 
    
    temperature = 0.1
    p = 1 / (1 + np.exp(-score / temperature))
   
    return p

        
        
def collect_preference_data(  #TODO if the policy should be the reference policy
    actions:np.ndarray,
    sample_size: int,
    feature_func: nn.Module,
) -> List[Transition]:
    pref_dataset = []
    actions = actions
    cur_state = np.array([0])   
    
    p_list = np.zeros([len(actions),len(actions)])
    for i in range(len(actions)):
        for j in range(len(actions)):
            action_one = actions[i]
            action_two = actions[j]
            p = get_p(action_one,action_two,feature_func)
            if i==j:
                assert p==0.5
            p_list[i][j] = p
            
    for i in range(sample_size):
        idx_one, idx_two = np.random.choice(len(actions), 2, replace=False)
        action_one = actions[idx_one]
        action_two = actions[idx_two]

        bernoulli_param = p_list[idx_one][idx_two]
        
        if np.random.random() < bernoulli_param:  
            transition = Transition(
                cur_state, action_one, action_two, 0, p_list[idx_one][idx_two]
            )
        else:
            transition = Transition(
                cur_state, action_one, action_two, 1, p_list[idx_two][idx_one]
            )
        pref_dataset.append(transition)

           
    return pref_dataset,p_list


def generate_dataset_from_policy(
    actions:np.ndarray,
    p_list:np.ndarray,
    policy: nn.Module,
    num_samples=60):
    
    pref_dataset = []
    cur_state = np.array([0])   
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for _ in range(num_samples):
        state = torch.tensor(cur_state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            actions_prob = policy(state) 
        idx_one = np.random.choice(len(actions), p=actions_prob.squeeze(0).detach().cpu().numpy())
        while True:
            idx_two = np.random.choice(len(actions), p=actions_prob.squeeze(0).detach().cpu().numpy())
            if idx_one != idx_two:
                break
        action_one = actions[idx_one]
        action_two = actions[idx_two]
        
        bernoulli_param = p_list[idx_one][idx_two]
        
        if np.random.random() < bernoulli_param:
            transition = Transition(
                cur_state, action_one, action_two, 0, p_list[idx_one][idx_two]
            )
        else:
            transition = Transition(
                cur_state, action_one, action_two, 1, p_list[idx_two][idx_one]
            )
        pref_dataset.append(transition)
    states =  torch.cat([torch.tensor(x.state , dtype=torch.float32) for x in pref_dataset], dim=0).to(device)
    action_to_index = {-10: 0, 0: 1, 10: 2}
    
    positive_actions = torch.cat(
    [torch.tensor(action_to_index[x.action_1] if x.pref == 1 else action_to_index[x.action_0]).unsqueeze(0) for x in pref_dataset],
    dim=0
    ).to(device)

    negative_actions = torch.cat(
        [torch.tensor(action_to_index[x.action_0] if x.pref == 1 else action_to_index[x.action_1]).unsqueeze(0) for x in pref_dataset],
        dim=0
    ).to(device)
    return pref_dataset, states, positive_actions, negative_actions
   



def collect_preference_intransitive_data(
    p_list:np.ndarray,
    actions:np.ndarray,
    sample_size: int,
)-> List[Transition]:
    pref_dataset = []
    actions = actions
    cur_state = np.array([0])   

    for i in range(sample_size):
        idx_one, idx_two = np.random.choice(len(actions), 2, replace=False)
        action_one = actions[idx_one]
        action_two = actions[idx_two] 
          
        bernoulli_param = p_list[idx_one][idx_two]
        
        if np.random.random() < bernoulli_param:
            transition = Transition(
                cur_state, action_one, action_two, 0, p_list[idx_one][idx_two]
            )
        else:
            transition = Transition(
                cur_state, action_one, action_two, 1, p_list[idx_two][idx_one]
            )
        pref_dataset.append(transition)
    return pref_dataset

def check_data_consistency(pref_dataset):
    consistent = 0
    total = len(pref_dataset)
    
    for t in pref_dataset:
      
        if (t.chosen_probs > 0.5):
            consistent += 1
    
    return consistent / total


def check_cyclic_order(matrix, actions):
    items = [str(i) for i in actions]
    # Generate all permutations of the items
    for perm in itertools.permutations(items):
        # Extract preferences based on the current permutation
        if (matrix[items.index(perm[0])][items.index(perm[1])] > 0.5 and
            matrix[items.index(perm[1])][items.index(perm[2])] > 0.5 and
            matrix[items.index(perm[2])][items.index(perm[0])] > 0.5):
            return True, perm  # Found a cyclic order
    return False, None  # No cyclic order found