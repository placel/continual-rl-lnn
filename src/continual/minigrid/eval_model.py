import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils import models
from utils import evaluation
import gymnasium as gym
import torch
import minigrid
import json

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
print(f'Using {device}')

# env_id = ['MiniGrid-Empty-5x5-v0']
# env_id = ['MiniGrid-DoorKey-5x5-v0']
env_id = ['MiniGrid-DoorKey-6x6-v0']
# env_id = ['MiniGrid-Unlock-v0']
env = gym.make(env_id[0], render_mode='human')

n_actions = env.action_space.n
state, _ = env.reset()

model_path = f'{os.path.dirname(__file__)}/experiments/ex-1754690870.244632/models/a=False_c=False_1754693888'
with open(f'{model_path}/config.json', 'rb') as f:
    config = json.load(f)

# Need to load the config file, much easier
# config = {'action_space': n_actions, 'hidden_dim': 64, 'hidden_state_dim': 128, 'actor_cfc': True, 'critic_cfc': True, 'use_lstm': False }
# config = {'action_space': n_actions, 'hidden_dim': 256, 'hidden_state_dim': 128, 'actor_cfc': False, 'critic_cfc': False, 'use_lstm': True }
# Create the Agent
agent = models.Agent(config['model_config'])
# agent = models.Agent(config)
# Set Strict to False, as we EWC buffers will cause error, but we don't need them
# agent.load_state_dict(torch.load(f'{os.path.dirname(__file__)}/experiments/ex-1754508521.3341138/models/a=True_c=True_ewc_1754513708/MiniGrid-Unlock-v0-es_False.pt'), strict=False)
### USE THIS LNN ONE
# agent.load_state_dict(torch.load(f'{os.path.dirname(__file__)}/experiments/ex-1754499199.99062/models/a=True_c=True_1754504139/MiniGrid-Unlock-v0-es_False.pt'), strict=False)
### USE THIS LSTM 
# agent.load_state_dict(torch.load(f'{os.path.dirname(__file__)}/experiments/continual/models/a=False_c=False_1754603688/MiniGrid-Unlock-v0-es_False.pt'), strict=False)
agent.load_state_dict(torch.load(f'{model_path}/{env_id[0]}.pt'), strict=False)
agent.to(device)

rewards = evaluation.mean_reward(agent, env_id, render_mode='human')
print(f'Rewards: {rewards}')