import torch 
import gymnasium as gym
import torch.nn as nn
import torch.functional as F
import minigrid
import numpy as np
from torch.distributions import Categorical
import ModelVariations as models

# Need to store model arguments so I can load the model without needing to copy over settings
class Agent(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        action_space = config['action_space']
        hidden_dim = config['hidden_dim']
        actor_cfc = config['actor_cfc']
        critic_cfc = config['critic_cfc']

        self.shared_embedding = models.SharedEmbedding(hidden_dim)
        
        # Compute the output size dynamically
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, 7, 7)  # Example MiniGrid obs (C=3,H=7,W=7)
            shared_out = self.shared_embedding(dummy_image)
            embedding_dim = shared_out.shape[1]
        
        # Create the critic model with the same variables as the CfC
        self.critic = models.CriticHead(self.shared_embedding, embedding_dim, hidden_dim, critic_cfc)

        # Extract the action space of the environment
        # action_space = np.array(envs.single_action_space).prod()

        # Create the actor model. Pass in the envs which will adapt depending on the tasks at hand
        self.actor = models.ActorHead(self.shared_embedding, embedding_dim, action_space, hidden_dim, actor_cfc)

    def get_value(self, image, states=None):
        return self.critic(image, states)
    
    def get_action_and_value(self, image, states=None, action=None):
        # Pass the image, states, and mission to the model to get an action
        logits = self.actor(image, states)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(image, states)
# Setup device usage
device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
print(f'Using {device}')

# Create the CartPole environment
env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='human')

n_actions = env.action_space.n
state, _ = env.reset()

config = {
    'action_space': n_actions,
    'hidden_dim': 128,
    'actor_cfc': False,
    'critic_cfc': False
}
# Create the Agent
agent = models.Agent(config)
agent.load_state_dict(torch.load('./src/ppo/baseline/continual/minigrid/models/final_model.pt'))
agent.to(device)

# Initialize the states for LNN prediction
batch_size = 1
# Make sure it's the same as the hidden_state size of the model trained on
# Ran into an error where hidden_dim was used instead, but it was a mismatch, leading to massive errors
hidden_state_size = 128
h = torch.zeros(batch_size, hidden_state_size).to(device)
c = torch.zeros(batch_size, hidden_state_size).to(device)
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
        action, _, _, _ = agent.get_action_and_value(state_tensor, actor_states, determinstic=True)

    action = action.cpu().numpy()[0]
    state, reward, terminated, truncated, _ = env.step(action)

    done = np.logical_or(terminated, truncated)
    if done:
        print(f'Reward: {round(reward, 2)}')

print(f'Final training time: {datetime.timedelta(seconds=time.time() - start_time)}')
env.close() 