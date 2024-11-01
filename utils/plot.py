import copy
import numpy as np
from typing import List
from envs.linear_bandit import LinearBandit
from utils.collect_data import Transition, ret_uniform_policy, collect_preference_data
import matplotlib.pyplot as plt


def compare_pref_with_policy(ret_action_prob, dataset: List[Transition]) -> float:
        """
        Compare the preferences predicted by the policy with the actual preferences in the dataset.
        Returns the accuracy of the policy's predictions.
        """
        correct_predictions = 0
        for transition in dataset:
            state, action_one, action_two, actual_pref, _ = (
                transition.state,
                transition.action_0,
                transition.action_1,
                transition.pref,
                transition.chosen_probs
            )

            # Get action probabilities for both actions from the current policy
            action_probs = ret_action_prob(state)
            prob_one = action_probs[action_one]
            prob_two = action_probs[action_two]

            # Determine the predicted preferred action
            predicted_pref = 1 if prob_two > prob_one else 0

            # Compare with actual preference
            if predicted_pref == actual_pref:
                correct_predictions += 1

        # Return the accuracy as the percentage of correct predictions
        accuracy = correct_predictions / len(dataset)
        return accuracy
    
    
def plot_model_accuracies(model_names: List[str], accuracies: List[float], filename) -> None:
    """
    Plot the accuracies of different models.
    """
    plt.figure(figsize=(8, 6))
    colors = ['red', 'green', 'blue', 'purple']
    plt.bar(model_names, accuracies, color=colors)
    plt.ylim(0, 1)  
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.show()
    
    plt.savefig(filename)
    plt.close() 