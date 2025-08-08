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
def mean_reward(agent, envs, episodes=10, step_name='step', dir='results', render_mode=None):
    
    # List of possible actions the model can take. Used for debugging
    action_list = [
        'left',
        'right',
        'forward',
        'pickup',
        'drop',
        'toggle',
        'done'
    ]
    # Extract the device 
    device = next(agent.parameters()).device
    rewards = []
    # Loop over every environment
    for env_id in envs:

        env = gym.make(env_id, render_mode=render_mode)
        total_reward = []

        # Run each environment 10 times
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            action_buffer = []
            episode_reward = 0.0 # Keep track of total reward throughout episode

            # If the model uses a CfC layer for actor or critic, generate states to use, or set to None
            if agent.actor_cfc or agent.critic_cfc:
                cfc_states = torch.zeros(1, agent.hidden_state_dim, device=device)
            else:
                cfc_states = None

            if agent.use_lstm:
                lstm_states = (
                    torch.zeros(1, 1, agent.hidden_state_dim, device=device), # (LSTM Layers, Batch Size, hidden_state_dim) 
                    torch.zeros(1, 1, agent.hidden_state_dim, device=device)
                )
            else:
                lstm_states = (None, None)
                
            while not done:
                # Normalize the image, reshape for cnn, and unsqueeze to add batch dimension
                img = torch.tensor(np.array(state['image']), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

                # Get the action to take
                # No gradient as we don't want the model learning here
                with torch.no_grad():
                    action, _, _, _, cfc_states, lstm_states, _ = agent.get_action_and_value(img, cfc_states, lstm_states, deterministic=True) # Deterministic set True as we don't want stochasticity to influence the model during eval
                
                action = action.item()
                # action_buffer.append(action)
                # print(f'Action taken: {action_list[action]}')
                # APply the action to current state
                state, reward, term, trunc, _ = env.step(action)
                done = np.logical_or(term, trunc)
                episode_reward += reward

                # if len(action_buffer) > 10:
                    

            total_reward.append(episode_reward) # Append the episode reward for averaging later

        mean_reward = np.mean(total_reward) # Average out the current environments reward
        rewards.append(mean_reward) # append reward to total rewards array
    
    return np.array(rewards) # Return rewards as numpy array

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

def plot_perf_matrix(perf_matrix, sequence=None, save_path='./performance_matrix'):
    
    # Extract the environment name alone, not 'MiniGrid' or 'v-0' for clean presentation
    labels = []
    for e in sequence:
        e = e.split('-')

        # If len(e) is >= 4, the environment has something like '5x5'or '6x6' which is relevant, and should be kept (e.g. Empty-5x5)
        if len(e) >= 4:
            labels.append('-'.join(e[1:3]))
        # Otherwise just take the environment name (e.g. DoorKey)
        else:
            labels.append(e[1])


    # Reverse the sequence and convert to list to display as the y_tick labels
    plt.figure(figsize=(6, 5))
    sns.heatmap(perf_matrix, annot=True, fmt='.2f', cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.title('Performance Matrix')
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
    plt.plot()
    plt.savefig(save_path)
    plt.close()