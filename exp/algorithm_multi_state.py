import collections
from typing import List
import numpy as np
import math
from utils.utils import distribution_comparison, model_comparison
import torch.cuda
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import copy

from utils.logger import Logger
from utils.logger import Logger
from utils.plot import plot_scores, two_action_prob_plot
####################################################################################################
#                                      RLHF                                                        #
####################################################################################################
class RewardModel(nn.Module):
    def __init__(
        self,
        state_num,
        action_num,
        state_feature_extractor: nn.Module = None,
        action_feature_extractor: nn.Module = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        if state_feature_extractor is None:
            state_feature_extractor = [
                nn.Linear(state_num, hidden_dim),
                nn.Tanh(),
            ]
            for _ in range(num_layers - 1):
                state_feature_extractor.append(nn.Linear(hidden_dim, hidden_dim))
                state_feature_extractor.append(nn.Tanh())
        self.state_feature_extractor = nn.Sequential(*state_feature_extractor).to(self.device)
        
        
        if action_feature_extractor is None:
            action_feature_extractor = [
                nn.Linear(action_num, hidden_dim),
                nn.Tanh(),
            ]
            for _ in range(num_layers - 1):
                action_feature_extractor.append(nn.Linear(hidden_dim, hidden_dim))
                action_feature_extractor.append(nn.Tanh())
        self.action_feature_extractor = nn.Sequential(*action_feature_extractor).to(self.device)

        self.predict_layer = nn.Linear(hidden_dim , 1).to(self.device)

    def forward(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        assert len(state.shape) == len(action.shape)
        assert torch.all(action >= 0) and torch.all(action <= 1), f"{action}"

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
       
        hs = self.state_feature_extractor(state)
        ha = self.action_feature_extractor(action)
        h = torch.cat([hs, ha], dim=1)
        
        rew = self.predict_layer(h)
       
        return rew

        
class MaximumLikelihoodEstimator:
    def __init__(
        self,
        action_num: int,
        reward_model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        logger: Logger = None,
    ):
        self.action_num = action_num
        self.reward_model = reward_model
        self.batch_size = batch_size
        self.logger = logger

        self.optimizer = torch.optim.AdamW(
            self.reward_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def optimize_one_epoch(self, states, positive_actions, negative_actions):
        total_loss = 0.0
        total_acc = 0.0

        k = 0
        for i in range(0, len(states), self.batch_size):
            self.optimizer.zero_grad()

            _states = states[i : i + self.batch_size]
            _positive_actions = positive_actions[i : i + self.batch_size]
            _negative_actions = negative_actions[i : i + self.batch_size]

            _positive_actions = F.one_hot(
                _positive_actions, num_classes=self.action_num
            )
            _negative_actions = F.one_hot(
                _negative_actions, num_classes=self.action_num
            )
            
         
            _states = _states.unsqueeze(1) if _states.dim() == 1 else _states
            

            positive_rews = self.reward_model(_states, _positive_actions)
            negative_rews = self.reward_model(_states, _negative_actions)

            loss = -torch.log(torch.sigmoid(positive_rews - negative_rews)).mean()
            loss.backward()
            self.optimizer.step()

            acc = (positive_rews > negative_rews).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()
            k += 1

        return total_loss / k, total_acc / k

    def optimize(self, states, positive_actions, negative_actions, num_epochs):
        for epoch in range(num_epochs):
            loss, acc = self.optimize_one_epoch(
                states, positive_actions, negative_actions
            )
            if self.logger:
                if epoch % 2 == 0:
                    self.logger.info(
                        f"[Reward] Epoch {epoch} loss: {loss:.4f} acc: {acc:.2f}"
                    )

class PolicyGradientOptimizer:
    def __init__(
        self,
        policy: nn.Module,
        reward_model: nn.Module,
        ref_policy: nn.Module,  
        score_ref_policy: nn.Module,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        logger: Logger = None,
        nash_point: List[float] = None,
    ):
        self.policy = policy
        self.reward_model = reward_model
        self.ref_policy = ref_policy
        self.score_ref_policy = score_ref_policy
        
        self.batch_size = batch_size
        self.logger = logger
        self.nash_point = nash_point
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=learning_rate
        )

    def optimize_one_epoch(self, states):
        total_loss = 0.0
        k = 0
        for i in range(0, len(states), self.batch_size):
            self.optimizer.zero_grad()
            
            _states = states[i : i + self.batch_size]
           
            distributions = self.policy(_states)
            ref_distributions = self.ref_policy(_states)        
            
            rewards = self.reward_model(_states, distributions)
            ref_rewards = self.reward_model(_states, ref_distributions)
            
            
            loss = -torch.sum(distributions * rewards, dim=-1).mean()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            k += 1
            
            # record prob of choosing action 0 and action 1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            test_state = torch.tensor([[0.0]], dtype=torch.float32).to(device) 
            with torch.no_grad():
                action_probs = self.policy(test_state)
                action_probs = action_probs.cpu().numpy()[0] 
            action_0_prob = action_probs[0]
            action_1_prob = action_probs[1]
            
        return total_loss / k, rewards.mean().item(), ref_rewards.mean().item(), action_0_prob, action_1_prob

    def optimize(
        self,
        states: torch.tensor,
        p_list:List[List[float]],
        num_epochs=100):
        
        action_0_probs = []
        action_1_probs = []
        scores = []
        for epoch in range(num_epochs):
            loss, reward, ref_reward, action_0_prob, action_1_prob  = self.optimize_one_epoch(states)
            action_0_probs.append(action_0_prob)
            action_1_probs.append(action_1_prob)
            
            score = model_comparison(self.policy, self.score_ref_policy, p_list)
            scores.append(score)
            
            if epoch % 2 == 0:
                if self.logger:
                    self.logger.info(
                        f"[Policy] Epoch {epoch} "
                        f"loss: {loss:.4f} "
                        f"reward: {reward:.4f} "
                        f"ref_reward: {ref_reward:.4f} "
                        f"improvement: {(reward-ref_reward)/abs(ref_reward):.2%}"
                    )
        two_action_prob_plot(action_0_probs, action_1_probs,self.nash_point,'RLHF')
        plot_scores(scores, num_epochs)
####################################################################################################
#                                      DPO                                                         #
####################################################################################################
class DirectPreferenceOptimizer:
    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        score_ref_policy: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        beta: float = 1.0,
        logger: Logger = None,
        nash_point: List[float] = None
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.score_ref_policy = score_ref_policy
        
        self.batch_size = batch_size
        self.beta = beta
        self.logger = logger
        self.nash_point = nash_point
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def optimize_one_epoch(
        self,
        states: torch.tensor,
        positive_actions: torch.tensor,
        negative_actions: torch.tensor,
    ):
        total_loss = 0.0
        total_gradient_norm = 0.0
        k = 0
        for i in range(0, len(states), self.batch_size):
            self.optimizer.zero_grad()

            _states = states[i : i + self.batch_size]
            distributions = self.policy(_states)
            ref_distributions = self.ref_policy(_states)

            _positive_actions = positive_actions[i : i + self.batch_size]
            _negative_actions = negative_actions[i : i + self.batch_size]

            pi_positive_logprobs = distributions[
                np.arange(len(_states)), _positive_actions
            ]
            pi_negative_logprobs = distributions[
                np.arange(len(_states)), _negative_actions
            ]

            ref_positive_logprobs = ref_distributions[
                np.arange(len(_states)), _positive_actions
            ]
            ref_negative_logprobs = ref_distributions[
                np.arange(len(_states)), _negative_actions
            ]

            pi_log_ratios = pi_positive_logprobs - pi_negative_logprobs
            ref_log_ratios = ref_positive_logprobs - ref_negative_logprobs

            log_ratios = pi_log_ratios - ref_log_ratios

            loss = -F.logsigmoid(self.beta * log_ratios).mean()

            total_loss += loss.item()

            loss.backward()

            gradient_norm = 0.0
            for p in self.policy.parameters():
                param_norm = p.grad.detach().data.norm(2)
                gradient_norm += param_norm.item() ** 2
            gradient_norm = gradient_norm**0.5
            total_gradient_norm += gradient_norm

            self.optimizer.step()

            k += 1

            # record prob of choosing action 0 and action 1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            test_state = torch.tensor([[0.0]], dtype=torch.float32).to(device) 
            with torch.no_grad():
                action_probs = self.policy(test_state)
                action_probs = action_probs.cpu().numpy()[0] 
            action_0_prob = action_probs[0]
            action_1_prob = action_probs[1]
            
        return total_loss / k, total_gradient_norm / k, action_0_prob, action_1_prob

    def optimize(
        self,
        states: torch.tensor,
        positive_actions: torch.tensor,
        negative_actions: torch.tensor,
        p_list:List[List[float]],
        num_epochs: int = 10,
        
    ):
        eval_epoch_interval = 5
        action_0_probs = []
        action_1_probs = []
        scores = []
        for epoch in range(num_epochs):
            loss, gradient_norm,action_0_prob, action_1_prob = self.optimize_one_epoch(
                states, positive_actions, negative_actions
            )
            action_0_probs.append(action_0_prob)
            action_1_probs.append(action_1_prob)
            
            score = model_comparison(self.policy, self.score_ref_policy, p_list)
            scores.append(score)
            if epoch % eval_epoch_interval == 0:
                if self.logger:
                    self.logger.info(
                        f"[Policy] Epoch: {epoch} loss: {loss:.4f} grad norm: {gradient_norm:.4f} "
                    )
        two_action_prob_plot(action_0_probs, action_1_probs,self.nash_point,'DPO')
        plot_scores(scores, num_epochs)
####################################################################################################
#                                      SPPO                                                        #
####################################################################################################

class SelfPlayPreferenceOptimizer:
    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        score_ref_policy: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        eta: float = 1e-4,
        logger: Logger = None,
        nash_point: List[float] = None
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.score_ref_policy = score_ref_policy
        
        self.batch_size = batch_size
        self.eta = eta
        self.logger = logger
        self.nash_point = nash_point

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def optimize_one_epoch(
        self,
        states: torch.tensor,
        positive_actions: torch.tensor,
        negative_actions: torch.tensor,
        chsoen_probs: torch.tensor,
    ):
        total_loss = 0.0
        total_gradient_norm = 0.0
        k = 0
        for i in range(0, len(states), self.batch_size):
            self.optimizer.zero_grad()

            _states = states[i : i + self.batch_size]
            distributions = self.policy(_states)
            ref_distributions = self.ref_policy(_states)

            _positive_actions = positive_actions[i : i + self.batch_size]
            _negative_actions = negative_actions[i : i + self.batch_size]

            pi_positive_logprobs = distributions[
                np.arange(len(_states)), _positive_actions
            ]
            pi_negative_logprobs = distributions[
                np.arange(len(_states)), _negative_actions
            ]

            ref_positive_logprobs = ref_distributions[
                np.arange(len(_states)), _positive_actions
            ]
            ref_negative_logprobs = ref_distributions[
                np.arange(len(_states)), _negative_actions
            ]

            
            square_log_w = ((pi_positive_logprobs - ref_positive_logprobs) - self.eta * (chsoen_probs - 1 /2))**2
            square_log_l = ((pi_negative_logprobs - ref_negative_logprobs) - self.eta * (1 - chsoen_probs - 1 /2))**2

            loss = (square_log_w + square_log_l).mean()
            total_loss += loss.item()

            loss.backward()

            gradient_norm = 0.0
            for p in self.policy.parameters():
                param_norm = p.grad.detach().data.norm(2)
                gradient_norm += param_norm.item() ** 2
            gradient_norm = gradient_norm**0.5
            total_gradient_norm += gradient_norm

            self.optimizer.step()

            k += 1
            
            # record prob of choosing action 0 and action 1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            test_state = torch.tensor([[0.0]], dtype=torch.float32).to(device) 
            with torch.no_grad():
                action_probs = self.policy(test_state)
                action_probs = action_probs.cpu().numpy()[0] 
            action_0_prob = action_probs[0]
            action_1_prob = action_probs[1]

        return total_loss / k, total_gradient_norm / k, action_0_prob, action_1_prob

    def optimize(
        self,
        states: torch.tensor,
        positive_actions: torch.tensor,
        negative_actions: torch.tensor,
        chosen_probs: torch.tensor,
        p_list:List[List[float]],
        num_epochs: int = 10,
    ):
        eval_epoch_interval = 5
        action_0_probs = []
        action_1_probs = []
        
        scores = []
        for epoch in range(num_epochs):
            loss, gradient_norm, action_0_prob, action_1_prob = self.optimize_one_epoch(
                states, positive_actions, negative_actions,chosen_probs
            )
            action_0_probs.append(action_0_prob)
            action_1_probs.append(action_1_prob)
            score = model_comparison(self.policy, self.score_ref_policy, p_list)
            scores.append(score)
            if epoch % eval_epoch_interval == 0:
                if self.logger:
                    self.logger.info(
                        f"[Policy] Epoch: {epoch} loss: {loss:.4f} grad norm: {gradient_norm:.4f} "
                    )
            if epoch % 30 == 0:
                self.ref_policy = copy.deepcopy(self.policy)
        two_action_prob_plot(action_0_probs, action_1_probs,self.nash_point,'SPPO')
        plot_scores(scores, num_epochs)
####################################################################################################
#                                      SPPOClosedForm                                              #
####################################################################################################
class SPPOClosedForm:
   def __init__(
       self,
       actions: np.ndarray,
       score_ref_model: nn.Module,
       ref_policy: nn.Module,
       eta: float = 1e-4,
       batch_size: int = 64,
       logger: Logger = None,
       nash_point: List[float] = None,
       device: str = "cpu",
   ):
       self.actions = actions
       self.ref_policy = ref_policy
       self.score_ref_model = score_ref_model
       self.eta = eta
       self.batch_size = batch_size
       self.logger = logger
       self.nash_point = nash_point
       self.device = torch.device(device)

       state = torch.zeros(1, 1, dtype=torch.float32).to(self.device) # state = 0
       self.ref_distribution = self.ref_policy(state).squeeze(0)
       
   def compute_pi(
       self,
       p_list:List[List[float]],
   ):
      
        action_num = len(self.actions)
        exp_terms = []
        
        p_y_pi = []
        
        for i in range(action_num):
            p_y_pi = 0 
            for j in range(action_num):
                p_y_yi = self.ref_distribution[j] * p_list[i][j]
                p_y_pi += p_y_yi # p(y|pi)
                 
            exp_term = torch.exp(self.eta * p_y_pi)
            exp_terms.append(exp_term) 
        
        Z = torch.sum(self.ref_distribution * torch.stack(exp_terms), dim=-1, keepdim=True)    
   
        new_distribution = self.ref_distribution * torch.stack(exp_terms) / Z

        # record prob of choosing action 0 and action 1
        action_0_prob = new_distribution[0].cpu().numpy()
        action_1_prob = new_distribution[1].cpu().numpy()
                   
        return new_distribution, action_0_prob, action_1_prob
   
   def optimize(
       self,
       p_list:List[List[float]],
       num_iters: int = 3,
   ):
       
        action_0_probs = []
        action_1_probs = []
        
        scores = []
        for iter in range(num_iters):
           
           new_distribution,action_0_prob,action_1_prob = self.compute_pi(p_list)
           ref_distribution = self.ref_distribution

           action_0_probs.append(action_0_prob)
           action_1_probs.append(action_1_prob)
          
           if self.logger:
                self.logger.info(
                    f"Iteration {iter}: ref_distribution = {ref_distribution}, new_distribution = {new_distribution}"
                )

           score = distribution_comparison(new_distribution, self.score_ref_model, p_list)
           scores.append(score)
           self.ref_distribution = copy.deepcopy(new_distribution)
           
        two_action_prob_plot(action_0_probs, action_1_probs,self.nash_point,'SPPOClosedForm')
        plot_scores(scores, num_iters)
        return new_distribution