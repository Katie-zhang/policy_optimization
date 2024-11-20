import numpy as np


class LinearBandit:
    def __init__(
        self,
        state_dim: int,
        actions: int,
        reward_param: np.ndarray,
        feature_func,
        num_trials_for_eval: int = None,
    ) -> None:
        self.state_dim = state_dim
        self.actions = actions
        self.action_space = [action_idx for action_idx in range(len(actions))]
        self.reward_param = reward_param
        self.feature_func = feature_func
        self.cur_state = np.random.uniform(0, 1, self.state_dim)
        

        self.num_trials_for_eval = num_trials_for_eval

    def reset(self) -> np.ndarray:
        self.cur_state = np.random.uniform(0, 1, self.state_dim)
        return self.cur_state    

    
    def get_opt_policy(self):
        def opt_policy(state: np.ndarray):
            # compute the optimal policy by enumerating the action space
         
            feature_mat = np.array(
            [
                self.feature_func(state, action)  
                for action in self.actions
            ],)
           
            assert np.shape(feature_mat) == (
                len(self.actions),
                self.reward_param.size,
            ), "The feature matrix is invalid."
            rew_vec = np.matmul(feature_mat, self.reward_param)
            optimal_action = np.argmax(rew_vec)
            action_prob = np.zeros(len(self.actions), np.float32)
            action_prob[optimal_action] = 1.0

            return action_prob

        return opt_policy

    def evaluate_reward(self, policy):
        """
        apply MC method to approximate the reward
        """
        rew = 0
        state_mat = np.random.uniform(
            0, 1, size=(self.num_trials_for_eval, self.state_dim)
        )
        feature_tensor = []
        action_mat = []
        for index in range(self.num_trials_for_eval):
            state = state_mat[index, :]
            action_prob = policy(state)
            assert np.all(action_prob >= 0.0) and np.allclose(
                np.sum(action_prob), 1.0
            ), "The policy is invalid."

            action_mat.append(action_prob)
            feature_mat = np.array(
            [
                self.feature_func(state, action)  
                for action in self.actions
            ],)
            feature_mat = np.stack(feature_mat, axis=0)
            feature_tensor.append(feature_mat)

        # feature_tensor has the shape of num * num_action * d
        feature_tensor = np.stack(feature_tensor, axis=0, dtype=np.float32)
        # reward_mat has the shape of num * num_action
        reward_mat = feature_tensor @ self.reward_param
        # action_mat has the shape of num * num_action
        action_mat = np.stack(action_mat, axis=0)

        rew = np.sum(np.multiply(reward_mat, action_mat)) / self.num_trials_for_eval

        return rew


def ret_feature_func(state_dim: int, is_flip: bool = False):
    """
    return the feature function for an arbitrary number of actions and any state dimension.
    """

    def feature_func(state: np.ndarray, action: int) -> np.ndarray:
        
        dim = 2 * state_dim
        feature = np.zeros(dim)
        for idx in range(state_dim):
            feature[2 * idx] = (action + 1) * np.sin(state[idx] * np.pi)
            feature[2 * idx + 1] = (1.0 / (action + 1)) * np.cos(state[idx] * np.pi)

        return feature

    return feature_func
