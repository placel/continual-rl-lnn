import torch 
import gymnasium as gym
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # Only one output as this is the critic model, and is generating an advantage value for PPO
        )

        self.actor = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape)
        )

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None, stochastic=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)

        if stochastic:
            action = probs.sample()
        else:
            action = torch.argmax(probs.logits, dim=-1)

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

# Setup device usage
device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
print(f'Using {device}')

# Create the CartPole environment
env = gym.make('CartPole-v1', render_mode='human')

n_actions = env.action_space.n
state, _ = env.reset()
n_obs = len(state)

# Create the Agent
agent = Agent(n_obs, n_actions)
agent.load_state_dict(torch.load('src/ppo/baseline/cartpole/models/agent.pt'))


done = False
import datetime
import time
start_time = time.time()
while not done:
    env.render()

    # time.sleep(2)
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Any pre-processing techniques need to be applied befor ewe use it
    with torch.no_grad():
        # INference the model and get the action
        # stochastic = False for argmax performance
        action, _, _, _ = agent.get_action_and_value(state_tensor, stochastic=False)

    action = action.cpu().numpy()[0]
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

print(f'Final training time: {datetime.timedelta(seconds=time.time() - start_time)}')
env.close() 