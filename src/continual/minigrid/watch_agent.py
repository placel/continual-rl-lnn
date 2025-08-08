import torch 
import gymnasium as gym
import torch.nn as nn
import torch.functional as F
import minigrid
import numpy as np
from torch.distributions import Categorical
import sys
import os
# Needed to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Custom class imports
from utils import evaluation
from utils import models # Custom class containing definitions for the main model implementations

# Setup device usage
device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
print(f'Using {device}')

# Create the CartPole environment
# env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='human')
env = gym.make('MiniGrid-Unlock-v0', render_mode='human')

n_actions = env.action_space.n
state, _ = env.reset()

config = {
    'action_space': n_actions,
    'hidden_dim': 64,
    'hidden_state_dim': 64,
    'actor_cfc': False,
    'critic_cfc': False
}
# Create the Agent
agent = models.Agent(config)
agent.load_state_dict(torch.load('./src/continual/minigrid/models/a=False_c=False_1754328328/final_model.pt'))
agent.to(device)

# Initialize the states for LNN prediction
batch_size = 1
# Make sure it's the same as the hidden_state size of the model trained on
# Ran into an error where hidden_dim was used instead, but it was a mismatch, leading to massive errors

h = torch.zeros(batch_size, config['hidden_state_dim']).to(device)
c = torch.zeros(batch_size, config['hidden_state_dim']).to(device)
actor_states = [h, c]

actor_states = None

done = False
import datetime
import time
start_time = time.time()
# print(f"Mission: {state['mission']}")
while not done:
    env.render()

    # time.sleep(0.5)
    # Permute the image to be (channels, height, width), then unsqueeze(0) to add batch of 1 to get (batch_size, channels, height, width)
    state_tensor = (torch.tensor(state['image'], dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0).unsqueeze(0)
    
    # Any pre-processing techniques need to be applied befor ewe use it
    with torch.no_grad():
        # INference the model and get the action
        # stochastic = False for argmax performance
        action, _, _, _, _, _ = agent.get_action_and_value(state_tensor, actor_states, deterministic=True)

    action = action.cpu().numpy()[0]
    state, reward, terminated, truncated, _ = env.step(action)

    done = np.logical_or(terminated, truncated)
    if done:
        print(f'Reward: {round(reward, 2)}')

print(f'Final training time: {datetime.timedelta(seconds=time.time() - start_time)}')
env.close() 