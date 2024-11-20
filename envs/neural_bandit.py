import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralBandit:
    def __init__(
        self,
        state_dim: int,
        actions: np.ndarray,
        num_trials_for_eval: int = 5000,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.actions = actions
        self.action_space = [action_idx for action_idx in range(len(actions))]
        self.device = torch.device(device)

        self.cur_state = None
        self.num_trials_for_eval = num_trials_for_eval

        self._evaluation_states = self.sample_states(self.num_trials_for_eval)

    # def sample_states(self, n=1) -> torch.tensor:
    #     states = (
    #         torch.rand(n, self.state_dim, dtype=torch.float32, device=self.device) * 2
    #         - 1.0
    #     )
    #     return states


    # def get_state_action_score(
    #     self, state: torch.tensor, action: torch.tensor
    # ) -> torch.tensor:
    #     # assert action in self.action_space, "The input action is invalid."
    #     assert torch.all(0 <= action) and torch.all(
    #         action <= len(self.actions) - 1
    #     ), f"action is invalid: {action}"
    #     assert len(state.shape) == 2, f"{state.shape}"
    #     assert len(action.shape) == 1, f"{action.shape}"

    #     action_one_hot = F.one_hot(action, num_classes=len(self.actions))
    #     score = self.reward_model(state, action_one_hot)
    #     return score
    
    # def get_opt_policy(self):
    #     def opt_policy(states: torch.tensor) -> torch.tensor:
    #         assert len(states.shape) == 2, f"{states.shape}"
    #         distributions = []
    #         max_rews = []
    #         with torch.no_grad():
    #             for i in range(len(states)):
    #                 state = torch.as_tensor(
    #                     states[i], dtype=torch.float32, device=self.device
    #                 )
    #                 assert len(state) == self.state_dim
    #                 state = torch.repeat_interleave(
    #                     state[None], repeats=len(self.actions), dim=0
    #                 )

    #                 action = torch.as_tensor(
    #                     np.arange(len(self.actions)), device=self.device
    #                 )
    #                 rews = self.get_state_action_reward(state, action)
    #                 opt_action = torch.argmax(rews.flatten())
    #                 max_rews.append(rews.max().item())

    #                 # print(rews.min(), rews.max())

    #                 prob = torch.zeros(
    #                     [1, len(self.actions)], dtype=torch.float32, device=self.device
    #                 )
    #                 prob[0, opt_action.item()] = 1.0

    #                 distributions.append(prob)

    #         distributions = torch.cat(distributions, dim=0)
    #         assert distributions.shape == (len(states), len(self.actions))
    #         torch.testing.assert_close(
    #             torch.sum(distributions, dim=-1),
    #             torch.ones(len(distributions), dtype=torch.float32),
    #             rtol=1e-3,
    #             atol=1e-3,
    #         )

    #         return distributions

    #     return opt_policy

    def evaluate_policy(
        self, policy: nn.Module, states: torch.tensor = None, avg: bool = True
    ) -> torch.tensor:
        if states is None:
            distributions = policy(self._evaluation_states)
            print(f"self._evaluation_states device: {self._evaluation_states.device}")
        else:
            distributions = policy(states)
            
        torch.testing.assert_close(
            torch.sum(distributions, dim=-1),
            torch.ones(len(distributions), dtype=torch.float32),
            rtol=1e-3,
            atol=1e-3,
        )
        if states is None:
            rews = torch.sum(distributions * self._evaluation_reward, dim=-1)
        else:
            evaluation_rews = self.evaluate_states(states)
            assert evaluation_rews.shape == (len(states), len(self.actions))
            rews = torch.sum(distributions * evaluation_rews, dim=-1)
        if avg:
            rews = rews.mean()
        return rews
