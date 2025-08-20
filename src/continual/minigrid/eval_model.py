import sys
import os
import gymnasium as gym
import torch
import minigrid
import json

# custom utils import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils import models
from utils import evaluation

device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
print(f'Using {device}')

# EXPERIMENT MODELS
# LNN
model_path = f'{os.path.dirname(__file__)}/experiments/phase_one/models/CfC_Critic_1755589055'
# model_path = f'{os.path.dirname(__file__)}/experiments/continual/models/a=True_c=True_1755281745'
# LSTM
# model_path = f'{os.path.dirname(__file__)}/experiments/ex-1754861588.2359602/models/a=False_c=False_1754864793'
# model_path = f'{os.path.dirname(__file__)}/experiments/continual/models/a=True_c=True_1754943727'

# HPO MODELS
# model_path = f'{os.path.dirname(__file__)}/HPO/mlp_hpo/models/hpo_trial_0'
# model_path = f'{os.path.dirname(__file__)}/HPO/shared_cfc_hpo_4/models/hpo_trial_18'

env_id = ['MiniGrid-Empty-5x5-v0']
# env_id = ['MiniGrid-DoorKey-5x5-v0']
# env_id = ['MiniGrid-DoorKey-6x6-v0']
# env_id = ['MiniGrid-Unlock-v0']
# env_id = ['MiniGrid-LavaGapS5-v0']
env = gym.make(env_id[0], render_mode='human')

n_actions = env.action_space.n
state, _ = env.reset()

# Load the config.json file within the evaluation model to create the corrosponding agent correctly
with open(f'{model_path}/config.json', 'rb') as f:
    config = json.load(f)

agent = models.Agent(config['model_config'])
# Set Strict to False, as we EWC buffers will cause error, but we don't need them
agent.load_state_dict(torch.load(f'{model_path}/MiniGrid-Unlock-v0.pt', weights_only=True), strict=False)
# agent.load_state_dict(torch.load(f'{model_path}/MiniGrid-DoorKey-5x5-v0.pt', weights_only=True), strict=False)
agent.to(device)

# Evaluate the model and enable human viewing
rewards, _, _ = evaluation.mean_reward(agent, env_id, render_mode='human')
print(f'Rewards: {rewards}')