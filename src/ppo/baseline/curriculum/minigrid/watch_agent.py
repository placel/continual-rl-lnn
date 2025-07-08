import torch 
import gymnasium as gym
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import CfCVariations as models

class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_size=64):
        super().__init__()

        self.hidden_size = hidden_size
        self.action_shape = action_shape
        self.critic = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # Only one output as this is the critic model, and is generating an advantage value for PPO
        )

        self.actor = models.ActorCfC(obs_shape, action_shape)

    def get_value(self, x):
        return self.critic(x)
    
    # If we're just watching the model, we don't need to store the states of the model
    def get_action_and_value(self, x, actor_states, action=None, stochastic=False):
        logits, new_actor_states = self.actor(x, actor_states)
        probs = Categorical(logits=logits)

        if stochastic:
            action = probs.sample()
        else:
            action = torch.argmax(probs.logits, dim=-1)

        return action, probs.log_prob(action), probs.entropy(), self.critic(x), new_actor_states

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
agent.load_state_dict(torch.load('./src/ppo/lnn/cartpole/models/lnn_agent.pt'))
agent.to(device)

# Initialize the states for LNN prediction
h = torch.zeros(agent.action_shape, agent.hidden_size).to(device)
c = torch.zeros(agent.action_shape, agent.hidden_size).to(device)
actor_states = [h, c]

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
        action, _, _, _, actor_states = agent.get_action_and_value(state_tensor, actor_states, stochastic=False)

    action = action.cpu().numpy()[0]
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
print(f'Final training time: {datetime.timedelta(seconds=time.time() - start_time)}')
env.close() 