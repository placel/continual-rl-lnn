import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import minigrid
import gymnasium as gym
import time
import datetime

# Iterate over every environment and play each 10 times to aquire perfromance
# Returns a list of average rewards for each environment 
# If doing few_shot learning in the future (more meta-learning) set few_shot to k trainable steps
def mean_reward(agent, envs, episodes=10, step_name='step', dir='results', few_shot=0):
    # Extract the device 
    device = next(agent.parameters()).device
    rewards = []
    # Loop over every environment
    for env_id in envs:

        env = gym.make(env_id)
        total_reward = []

        # Run each environment 10 times
        for i in range(episodes):
            state, _ = env.reset()
            done = False

            episode_reward = 0.0
            # Generate states here ( should be empty for initial pass)
            cfc_states = None
            while not done:
                # Normalize the image, reshape for cnn, and unsqueeze to add batch dimension
                img = (torch.tensor(state['image'], dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0).unsqueeze(0)

                # Get the action
                # Extract the states after implemented 
                # No gradient as we don't want the model learning here
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(img)

                # APply the action
                state, reward, term, trunc, _ = env.step(action)
                done = np.logical_or(term, trunc)

                episode_reward += reward
            
            total_reward.append(episode_reward)

        mean_reward = np.mean(total_reward)
        rewards.append(mean_reward)
    
    # Optionally save rewards here for logging
    return np.array(rewards)

# Follows the original paper (3,500 citations) instead of the 2018 revision (250 citation); just for coinsistency 
def compute_fwt(perf_matrix, b):
    # get the number of tasks
    n_tasks = perf_matrix.shape[0]

    # Difference between the performance of task i on Final model (j-1)
    # and performance task immediately after it's training phase 
    vals = [perf_matrix[i-1, i] - b[i] for i in range(1, n_tasks)]
    return np.mean(vals) if vals else 0.0

def compute_bwt(perf_matrix):
    n_tasks = perf_matrix.shape[0]
    vals = [perf_matrix[-1][i] - perf_matrix[i][i] for i in range(n_tasks-1)]

    return np.mean(vals) if vals else 0.0

def plot_perf_matrix(perf_matrix, save_path='./performance_matrix'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(perf_matrix, annot=True, fmt='.2f', cmap='viridis')
    plt.title('Performance Matrix (Train vs Eval)')
    plt.xlabel('Eval Task')
    plt.ylabel('Train Task')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metrics(fwt, bwt, mean_reward, save_path='./metrics'):
    plt.figure()
    bars = plt.bar(['Forward Transfer', 'Backward Transfer', 'Mean Reward'], [fwt, bwt, mean_reward])

    # Annotating bars: https://www.geeksforgeeks.org/python/how-to-annotate-bars-in-barplot-with-matplotlib-in-python/
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=12)
        
    plt.title('Metrics')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()