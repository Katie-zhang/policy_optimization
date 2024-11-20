import collections
from typing import List
import numpy as np
import math
import torch.cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import copy

from utils.logger import Logger


from utils.io_utils import save_code
from utils.logger import Logger

class PolicyModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        actions: np.ndarray,
        hidden_dim: int = 128,
        num_layers: int = 2,
        device: str = "cpu",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_num = len(actions)

        self.device = torch.device(device)

        network = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            network.append(nn.Linear(hidden_dim, hidden_dim))
            network.append(nn.ReLU())
        network.append(nn.Linear(hidden_dim, self.action_num))

        self.network = nn.Sequential(*network)

    def forward(self, state: torch.tensor) -> torch.tensor:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        logits = self.network(state)
        return torch.softmax(logits, dim=-1)


class UniformPolicyModel(nn.Module):
    def __init__(self, action_num: int, device: str = "cpu"):
        super().__init__()
        self.action_num = action_num
        self.device = torch.device(device)

    def forward(self, state: torch.tensor) -> torch.tensor:
        logits = torch.zeros(
            [len(state), self.action_num], dtype=torch.float32, device=self.device
        )
        return torch.softmax(logits, dim=-1)

#TODO: other ref models       
class Ref_PolicyModel(nn.Module):
    def __init__(self,action_num:int, prob:torch.tensor, device: str = "cpu"):
        super().__init__()
        self.action_num = action_num
        self.device = torch.device(device)
        self.prob = prob.to(self.device)
        
    def forward(self,state:torch.tensor) -> torch.tensor:
        prob = self.prob
        
        assert prob.shape[1] == self.action_num      
        assert torch.all(prob >= 0)
        assert torch.all(prob <= 1)
        assert torch.sum(prob) == 1
        
        batch_size = state.shape[0]
        prob = prob.repeat(batch_size,1)
        return prob