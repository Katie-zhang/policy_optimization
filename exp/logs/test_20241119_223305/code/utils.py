import numpy as np
import torch
from torch import nn
from ..collect_data import get_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def softmax(arr: np.ndarray) -> np.ndarray:
    assert len(np.shape(arr)) == 1, "The input array is not 1-dim."
    softmax_arr = np.exp(arr - np.max(arr))
    softmax_arr = softmax_arr / np.sum(softmax_arr)
    return softmax_arr


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

# output is a action from the distribution of model
def output_action(policy, state):
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        actions = policy(state).squeeze(0)
        actions = actions.cpu().numpy()
        chosen_action = np.random.choice(len(actions), p=actions)
    return chosen_action

def generate_outputs(policy,num_samples=10):
    outputs = []
    state = torch.zeros(1, dtype=torch.float32).to(device)
    for _ in range(num_samples):
        outputs.append(output_action(policy, state))
    return outputs


def model_comparison(policy, ref_policy,feature_func, num_samples=10):
    model_outputs = generate_outputs(policy, num_samples)
    ref_outputs = generate_outputs(ref_policy, num_samples)
    scores = []
    for i in range(num_samples):
        scores.append(get_score(model_outputs[i], ref_outputs[i],feature_func))
    return np.mean(scores)
        
  
  