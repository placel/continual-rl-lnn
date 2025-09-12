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

# Load model path
model_path = f'{os.path.dirname(__file__)}/experiments/phase_three/models/CfC_Critic_1756987356'

# Select the environment to test
env_id = ['MiniGrid-Empty-5x5-v0']
# env_id = ['MiniGrid-DoorKey-5x5-v0']
# env_id = ['MiniGrid-DoorKey-6x6-v0']
# env_id = ['MiniGrid-Unlock-v0']
# env_id = ['MiniGrid-UnlockPickup-v0']
# env_id = ['MiniGrid-FourRooms-v0']
# env_id = ['MiniGrid-LavaGapS5-v0']
# env_id = ['MiniGrid-LavaGapS6-v0']
env = gym.make(env_id[0], render_mode='human')

n_actions = env.action_space.n
state, _ = env.reset()

# Load the config.json file within the evaluation model to create the corrosponding agent correctly
with open(f'{model_path}/config.json', 'rb') as f:
    config = json.load(f)

agent = models.Agent(config['model_config'])
# Load the final_model.pt for final performance. Alter for specific environment testing
# Set Strict to False, as we EWC buffers will cause error, but we don't need them
agent.load_state_dict(torch.load(f'{model_path}/final_model.pt', weights_only=True), strict=False)
agent.to(device)

# Evaluate the model and enable human viewing
# Seed will ensure the same layouts every run
rewards, _, _ = evaluation.mean_reward(agent, env_id, render_mode='human', seed=1010)
print(f'Rewards: {rewards}')