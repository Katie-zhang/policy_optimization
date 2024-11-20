import copy
import cvxpy as cp
import numpy as np
from typing import List
from envs.linear_bandit import LinearBandit
from utils.collect_data import Transition, ret_uniform_policy, collect_preference_data
from utils.utils import softmax, sigmoid
from utils.logger import Logger
from utils.plot import compare_pref_with_policy
class SelfPlayPreferenceOptimization:
    def __init__(
        self,
        state_dim: int,
        actions: np.ndarray,
        feature_dim: int,
        feature_func,
        beta: float,  # beta=1/eta has a different meaning, and is usually chosen around 1e-3.
        step_size: float,
        num_iters: int,
        is_adaptive: bool = False,
        ada_coef: float = None,
        logger: Logger = None,
        ) -> None:
        self.state_diim = state_dim
        self.actions = actions
        self.feature_dim = feature_dim
        self.feature_func = feature_func
        self.beta = beta
        self.step_size = step_size
        self.num_iters = num_iters
        self.logger = logger
        
        self.is_adaptive = is_adaptive
        self.ada_coef = ada_coef
        self.hist_grad_squared_norm = 0.0
        # initialize the policy parameter
        self.param = np.random.uniform(0, 1, self.feature_dim)


    def ret_action_prob(self, state: np.ndarray) -> np.ndarray:
        arr = np.zeros(len(self.actions), np.float32)  
        for idx, action in enumerate(self.actions):
            feature = self.feature_func(state, action)
            arr[idx] = np.dot(feature, self.param)
        prob = softmax(arr)
        return prob

    def ret_policy(self):
        feature_func = copy.deepcopy(self.feature_func)
        param = self.param

        def policy(state: np.ndarray) -> np.ndarray:
            arr = np.zeros(len(self.actions), np.float32)
            for idx, action in enumerate(self.actions):
                feature = feature_func(state, action)
                arr[idx] = np.dot(feature, param)
            prob = softmax(arr)

            return prob

        return policy
        
        
    def update_once(self, ref_policy, dataset: List[Transition]) -> float:
        grad = np.zeros_like(self.param)
        for transition in dataset:
            state, action_one, action_two, pref, chosen_probs = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.pref,
                transition.chosen_probs
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            pref_act_idx = np.where(self.actions == pref_act)[0]
            non_pref_act_idx = np.where(self.actions == non_pref_act)[0]

            feat_pref_act, feat_non_pref_act = (
                self.feature_func(state, pref_act),
                self.feature_func(state, non_pref_act),
            )
            cur_policy_act_prob = self.ret_action_prob(state)
            ref_policy_act_prob = ref_policy(state)
            
            logits_w = np.log(cur_policy_act_prob[pref_act_idx] + 1e-6) - np.log(ref_policy_act_prob[pref_act_idx] + 1e-6)
            logits_l = np.log(cur_policy_act_prob[non_pref_act_idx] + 1e-6) - np.log(ref_policy_act_prob[non_pref_act_idx] + 1e-6)
            
            log_ratio_w = (logits_w - (1 / self.beta)*(chosen_probs - 0.5)) * 2
            log_ratio_l = (logits_l - (1 / self.beta)*(1 - chosen_probs - 0.5)) * 2
            
            coef_w = 1 / np.log(cur_policy_act_prob[pref_act_idx] + 1e-6)
            coef_l = 1 / np.log(cur_policy_act_prob[non_pref_act_idx] + 1e-6)
            
            grad +=  log_ratio_w * coef_w * feat_pref_act + log_ratio_l * coef_l * feat_non_pref_act 

        grad /= len(dataset)
        self.hist_grad_squared_norm += np.sum(np.square(grad))
        if self.is_adaptive:
            step_size = self.ada_coef / np.sqrt(self.hist_grad_squared_norm)
        else:
            step_size = self.step_size
        self.param = self.param - step_size * grad
        return np.sqrt(np.sum(np.square(grad)))
        
        
    def evaluate_loss(self, ref_policy, dataset: List[Transition], policy=None) -> float:
        """
        Evaluate the loss on the dataset for any policy.
        """
        if policy is None:
            policy = self.ret_policy()

        loss = 0.0
        for transition in dataset:
            state, action_one, action_two, pref, chosen_probs = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.pref,
                transition.chosen_probs
            )
            pref_act = action_two if pref == 1 else action_one
            non_pref_act = action_two if pref == 0 else action_one

            pref_act_idx = self.actions.index(pref_act)
            non_pref_act_idx = self.actions.index(non_pref_act)
            
            eval_policy_act_prob = policy(state)
            ref_policy_act_prob = ref_policy(state)
          
            logits_w = np.log(eval_policy_act_prob[pref_act_idx] + 1e-6) - np.log(ref_policy_act_prob[pref_act_idx] + 1e-6)
            logits_l = np.log(eval_policy_act_prob[non_pref_act_idx] + 1e-6) - np.log(ref_policy_act_prob[non_pref_act_idx] + 1e-6)
            
            loss_w = (logits_w - (1 / self.beta)*(chosen_probs - 0.5)) ** 2
            loss_l = (logits_l - (1 / self.beta)*(1 - chosen_probs - 0.5)) ** 2
            loss += (loss_w + loss_l)/2
        loss /= len(dataset)
        return loss


    

    def train(self, dataset: List[Transition], env: LinearBandit) -> float:
        ref_policy = self.ret_policy()
        for step in range(self.num_iters):
            grad_norm = self.update_once(ref_policy,dataset)
            if step % 200 == 0:
                ref_policy = self.ret_policy()
                loss = self.evaluate_loss(ref_policy,dataset)
                rew = self.evaluate_reward(env)
                if self.logger:
                    self.logger.info(
                        f"Iteration: {step: d}, loss: {loss: .4f}, grad_norm :{grad_norm:.4f}, reward: {rew: .4f}."
                    )
                else:
                    print(
                        f"Iteration: {step: d}, loss: {loss: .4f}, grad_norm :{grad_norm:.4f}, reward: {rew: .4f}."
                    )
        accuracy = compare_pref_with_policy(env,self.ret_action_prob, dataset)
        self.logger.info(f"SPPO Preference accuracy: {accuracy:.4f}")
        rew = self.evaluate_reward(env)
        rew = float(rew)
        return rew, accuracy
    
    # def train_by_closed_form(self, dataset: List[Transition], env: LinearBandit) -> float:
    #     ref_policy = self.ret_policy()
    #     for transition in dataset:
    #         state, action_one, action_two, pref, chosen_probs = (
    #             transition.state,
    #             transition.action_0,
    #             transition.action_1,
    #             transition.pref,
    #             transition.chosen_probs
    #         )
            
    #         ref_policy_act_prob = self.ref_policy(state)
    #         (1 / self.beta) * chosen_probs
    #     return reward
  
    def evaluate_reward(self, env: LinearBandit) -> float:
        policy = self.ret_policy()
        rew = env.evaluate_reward(policy)

        return rew

    @property
    def get_param(self) -> np.ndarray:
        return self.param