import torch 
import gymnasium as gym
import torch.nn as nn
import minigrid
import numpy as np
from torch.distributions import Categorical
import CfCVariations as models

class Agent(nn.Module):
    def __init__(self, action_shape, hidden_size=128):
        super().__init__()

        self.hidden_size = hidden_size
        self.action_shape = action_shape
        # Can add MaxPooling later
        self.critic = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            # layer_init(nn.LazyLinear(64)), # Automatically infer the size of flattened tensor
            nn.LazyLinear(64), # Automatically infer the size of flattened tensor
            nn.ReLU(),
            nn.Linear(64, 1) # Output of 1 for as it's a value 
        )

        # Create critic model, only pass obs_space as action_space isn't needed in a critic model
        # self.critic = lnn_models.CriticCfC(obs_space)
        
        # Create the actor model. Pass in the envs which will adapt depending on the tasks at hand
        self.actor = models.ActorCfC(self.action_shape, hidden_dim=self.hidden_size)

    def get_value(self, x):
        return self.critic(x)
    
    # If we're just watching the model, we don't need to store the states of the model
    def get_action_and_value(self, x, actor_states, action=None, stochastic=False):
        logits, new_actor_states = self.actor(x, actor_states)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

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
env = gym.make('MiniGrid-LavaGapS5-v0', render_mode='human')

n_actions = env.action_space.n
state, _ = env.reset()

# Create the Agent
agent = Agent(n_actions, hidden_size=128)
agent.load_state_dict(torch.load('./src/ppo/lnn/minigrid/models/lava-gap.pt'))
agent.to(device)

# Initialize the states for LNN prediction
batch_size = 1
# Make sure it's the same as the hidden_state size of the model trained on
# Ran into an error where hidden_dim was used instead, but it was a mismatch, leading to massive errors
hidden_state_size = 64 
h = torch.zeros(batch_size, hidden_state_size).to(device)
c = torch.zeros(batch_size, hidden_state_size).to(device)
actor_states = [h, c]

done = False
import datetime
import time
start_time = time.time()
while not done:
    env.render()

    # time.sleep(2)
    # Permute the image to be (channels, height, width), then unsqueeze(0) to add batch of 1 to get (batch_size, channels, height, width)
    state_tensor = torch.tensor(state['image'], dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    
    # Any pre-processing techniques need to be applied befor ewe use it
    with torch.no_grad():
        # INference the model and get the action
        # stochastic = False for argmax performance
        action, _, _, _, actor_states = agent.get_action_and_value(state_tensor, actor_states, stochastic=False)
        print(f'Action: {action}')

    action = action.cpu().numpy()[0]
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
print(f'Final training time: {datetime.timedelta(seconds=time.time() - start_time)}')
env.close() 