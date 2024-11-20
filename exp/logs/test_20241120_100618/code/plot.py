import matplotlib.pyplot as plt


# def compare_pref_with_policy(ret_action_prob, dataset: List[Transition]) -> float:
#         """
#         Compare the preferences predicted by the policy with the actual preferences in the dataset.
#         Returns the accuracy of the policy's predictions.
#         """
#         correct_predictions = 0
#         for transition in dataset:
#             state, action_one, action_two, actual_pref, _ = (
#                 transition.state,
#                 transition.action_0,
#                 transition.action_1,
#                 transition.pref,
#                 transition.chosen_probs
#             )

#             # Get index of the preferred action and the non-preferred action
#             action_one_idx = actions.index(action_one)
#             action_two_idx = actions.index(action_two)
           
#             # Get action probabilities for both actions from the current policy
#             action_probs = ret_action_prob(state)
#             prob_one = action_probs[action_one_idx]
#             prob_two = action_probs[action_two_idx]

#             # Determine the predicted preferred action
#             predicted_pref = 1 if prob_two > prob_one else 0

#             # Compare with actual preference
#             if predicted_pref == actual_pref:
#                 correct_predictions += 1

#         # Return the accuracy as the percentage of correct predictions
#         accuracy = correct_predictions / len(dataset)
#         return accuracy
    
    
# def plot_model_accuracies(model_names: List[str], accuracies: List[float], filename) -> None:
#     """
#     Plot the accuracies of different models.
#     """
#     plt.figure(figsize=(8, 6))
#     colors = ['red', 'green', 'blue', 'purple']
#     plt.bar(model_names, accuracies, color=colors)
#     plt.ylim(0, 1)  
#     plt.xlabel("Models")
#     plt.ylabel("Accuracy")
#     plt.title("Model Accuracy Comparison")
#     plt.show()
    
#     plt.savefig(filename)
#     plt.close() 
    
    
def two_action_prob_plot(action_0, action_1,policy):
    p_action_0 = policy(action_0)
    p_action_1 = policy(action_1)
    
    plt.figure(figsize=(8, 6))
    plt.bar([0,1], [p_action_0, p_action_1], color=['red', 'green'])
    plt.ylim(0, 1)
    plt.xticks([0,1], ['Action 0', 'Action 1'])
    plt.ylabel("Probability")
    plt.title("Action Probabilities")
    plt.show()
    

def two_action_prob_plot(action_0, action_1,algorithm):
    plt.figure(figsize=(8, 8))          
    plt.scatter(action_0, action_1, c='blue', s=5, alpha=0.5)                   
    plt.plot(0.3, 0.45, 'b.', markersize=10)          
    plt.grid(True, linestyle='--', alpha=0.6) 
        
    plt.xlabel('x[0]')     
    plt.ylabel('x[1]')    
     
    plt.title(algorithm)          
    plt.xlim(0, 1)     
    plt.ylim(0, 1)         
    plt.show() 
     