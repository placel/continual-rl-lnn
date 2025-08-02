import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
from utils import models
from utils import evaluation
import gymnasium as gym
import torch
import minigrid

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
print(f'Using {device}')

env_id = ['MiniGrid-Empty-8x8-v0']
env = gym.make(env_id[0], render_mode='human')

n_actions = env.action_space.n
state, _ = env.reset()

config = {
    'action_space': n_actions,
    'hidden_dim': 128,
    'actor_cfc': True,
    'critic_cfc': False
}
# Create the Agent
agent = models.Agent(config)
agent.load_state_dict(torch.load(f'{os.path.dirname(__file__)}/models/final_model.pt'))
agent.to(device)

rewards = evaluation.mean_reward(agent, env_id, render_mode='human')
print(f'Rewards: {rewards}')