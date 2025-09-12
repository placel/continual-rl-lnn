import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import minigrid
import gymnasium as gym
import time
import datetime
import pandas as pd

"""
THIS SHOULD'VE BEEN A CLASS, NOT STATIC
"""

# Iterate over every environment and play each 10 times to aquire perfromance
# Returns a list of average rewards for each environment 
# If doing few_shot learning in the future (more meta-learning) set few_shot to k trainable steps
def mean_reward(agent, envs, seed=42, episodes=10, return_all=True, render_mode=None):

    # Extract the device 
    device = next(agent.parameters()).device
    rewards, stds, all_returns = [], [], []
    # Loop over every environment 
    for env_id in envs:

        env = gym.make(env_id, render_mode=render_mode)
        total_reward = []

        state, _ = env.reset(seed=seed)
        dones = torch.zeros(1).to(device)
        # Run each environment 10 times
        for i in range(episodes):
            if i > 0:
                state, _ = env.reset() # Don't overwrite the initial seed if not the first task
            
            done = False
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
                # Reshape image for cnn, and unsqueeze to add batch dimension
                img = torch.tensor(np.array(state['image']), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

                # Get the action to take
                # no_grad as we don't want the model learning here
                with torch.no_grad():
                    action, _, _, _, cfc_states, lstm_states, _ = agent.get_action_and_value(img, cfc_states, lstm_states, dones=dones, deterministic=True) # Deterministic set True as we don't want stochasticity to influence the model during eval
                
                action = action.item()
                state, reward, term, trunc, _ = env.step(action)
                done = np.logical_or(term, trunc)
                episode_reward += reward

            total_reward.append(episode_reward) # Append the episode reward for averaging later

        total_reward = np.asarray(total_reward, dtype=np.float32)

        mean = float(total_reward.mean())
        std  = float(total_reward.std(ddof=1))

        rewards.append(mean)
        stds.append(std)
        all_returns.append(total_reward)

    if return_all:
        return np.array(rewards), np.array(stds), np.array(all_returns)
    else:
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
    # Final performance subtracted by performance after learning task initially. How much it forgot
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
    plt.style.use('fivethirtyeight')
    sns.heatmap(perf_matrix, annot=True, fmt='.2f', cmap='GnBu', xticklabels=labels, yticklabels=labels)
    plt.title('Performance Matrix')
    plt.xlabel('Eval Task')
    plt.ylabel('Train Task')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metrics(fwt, bwt, mean_reward, save_path='./metrics'):
    plt.figure()
    plt.style.use('fivethirtyeight')
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

def extract_task_names(sequence):
    return [tasks.split('-')[1] for tasks in sequence]

# Plot the reward at the end of training
def plot_reward(path, sequence, save_path='./rewards.svg'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    sequence = extract_task_names(sequence)
    # Load the models associated 'episodes.csv' file
    df = pd.read_csv(f'{path}/episodes.csv')
    
    plt.style.use('fivethirtyeight')
    _, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', "#25a325", "#d61e1e", '#9467bd']
    
    # Extract the timestep and boundaries
    task_boundaries_steps = [list(df[df['task_index'] == i]['global_step'])[0] for i in range(len(sequence))]
    
    # Extract and plot x & y coordinates with styling
    x, y = df['global_step'], df['episodic_return']
    ax.plot(x, y, linewidth=2.5, alpha=0.8, color=colors[0], zorder=3)
    
    # Add vertical lines
    for indx, task in enumerate(sequence):
        if indx == 0: continue
        ax.axvline(x=task_boundaries_steps[indx], color=colors[indx], linestyle='dashed', 
                  linewidth=2, alpha=0.7, zorder=2, label=f'{sequence[indx-1]}→{task}')
    
    # Improved styling
    ax.set_xlabel('Total Timesteps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Episodic Return', fontsize=14, fontweight='bold')
    ax.set_title('Training Reward - Best HPO Trial', fontsize=16, fontweight='bold', pad=20)
    
    # Better grid and background
    ax.grid(True, alpha=0.75, linestyle='dashed', linewidth=0.5)
    ax.set_facecolor('#fafafa')
    
    # Modern legend
    legend = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Set proper margins
    ax.margins(0)
    ax.set_ylim(0, 1)
    # Better tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

# Plot the loss at the end of training
def plot_loss_curve(path, sequence, save_path='./rewards.svg'):
    sequence = extract_task_names(sequence)
    # Load the models associated 'updates.csv' file
    df = pd.read_csv(f'{path}/updates.csv')
    
    plt.style.use('fivethirtyeight')
    colors = ['#1f77b4', '#ff7f0e', "#25a325", "#d61e1e", '#9467bd']
    _, ax = plt.subplots(figsize=(12, 8))

    
    # Extract the timestep and boundaries
    task_boundaries_steps = [list(df[df['task_index'] == i]['global_step'])[0] for i in range(len(sequence))]
    
    # Extract and plot x & y coordinates with better styling
    x, y = df['global_step'], df['policy_loss']
    ax.plot(x, y, linewidth=2.5, alpha=0.8, color=colors[0], zorder=3)
    
    for indx, task in enumerate(sequence):
        if indx == 0: continue
        ax.axvline(x=task_boundaries_steps[indx], color=colors[indx], linestyle='dashed', 
                  linewidth=2, alpha=0.7, zorder=2, label=f'{sequence[indx-1]}→{task}')
    
    # Improved styling
    ax.set_xlabel('Total Timesteps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Policy Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training Loss - Best HPO Trial', fontsize=16, fontweight='bold', pad=20)
    
    # Better grid and background
    ax.grid(True, alpha=0.75, linestyle='dashed', linewidth=0.5)
    ax.set_facecolor('#fafafa')
    
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()