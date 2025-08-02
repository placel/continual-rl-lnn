import torch 
import gymnasium as gym
import torch.nn as nn
import minigrid
import numpy as np
from torch.distributions import Categorical
import ModelVariations as models
import torch.functional as F

# Need to store model arguments so I can load the model without needing to copy over settings
class Agent(nn.Module):
    
    def __init__(self, action_space, vocab_size, hidden_dim, word_embedding_dim, text_embedding_dim, actor_cfc=False, critic_cfc=False):
        super().__init__()

        self.shared_embedding = models.SharedEmbedding(vocab_size, hidden_dim, word_embedding_dim, text_embedding_dim)
        
        # Compute the output size dynamically
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, 7, 7)  # Example MiniGrid obs (C=3,H=7,W=7)
            dummy_text = torch.zeros(1, 10, dtype=torch.long)  # Example mission length = 10
            shared_out = self.shared_embedding(dummy_image, dummy_text)
            embedding_dim = shared_out.shape[1]
        
        # Create the critic model with the same variables as the CfC
        self.critic = models.CriticHead(self.shared_embedding, embedding_dim, hidden_dim, critic_cfc)

        # Extract the action space of the environment
        # action_space = np.array(envs.single_action_space).prod()

        # Create the actor model. Pass in the envs which will adapt depending on the tasks at hand
        self.actor = models.ActorHead(self.shared_embedding, embedding_dim, action_space, hidden_dim, actor_cfc)

    def get_value(self, image, mission, states=None):
        return self.critic(image, mission, states)
    
    def get_action_and_value(self, image, mission, states=None, action=None):
        # Pass the image, states, and mission to the model to get an action
        logits = self.actor(image, mission, states)
        probs = Categorical(logits=F.log_softmax(logits, dim=1))
        
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(image, mission, states)
# Setup device usage
device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
print(f'Using {device}')

# Create the CartPole environment
# env = gym.make('MiniGrid-Fetch-5x5-N2-v0', render_mode='human')
env = gym.make('MiniGrid-Empty-Random-5x5-v0', render_mode='human')

n_actions = env.action_space.n
state, _ = env.reset()

unique_words = [
    'get', 'a', 'go', 'fetch', 'you', 'must', 'to', 'the', 'matching',
    'object', 'at', 'end', 'of', 'hallway', 'traverse', 'rooms', 'goal',
    'put', 'near', 'and', 'open', 'red', 'door', 'then', 'blue', 'pick',
    'up', 'green', 'grey', 'purple', 'yellow', 'box', 'key', 'ball', 'square',
    'use', 'it', 'next', 'first', 'second', 'third'
]

PAD_IDX = 0
UNK_IDX = 1
# Start vocab indexing from 2 to leave 0 for PAD and 1 for UNK
word_dict = {word: idx + 2 for idx, word in enumerate(unique_words)}

def process_mission(missions, max_mission_length=12, device='cpu'):  # Increased from 9
    """Fixed mission processing with better debugging"""
    batch_tokens = []
    
    # print(missions)
    # Debug: Print missions to see what we're getting
    if isinstance(missions, (list, tuple)) and len(missions) > 0:
        # print(f"Sample mission: '{missions[0]}'")
        pass


    words = missions.lower().split()
    
    # Debug: Check for unknown words
    unknown_words = [w for w in words if w not in word_dict]
    if unknown_words:
        print(f"Unknown words in mission '{missions}': {unknown_words}")
    
    # Truncate if too long
    if len(words) > max_mission_length:
        words = words[:max_mission_length]
        print(f"Warning: Mission truncated from {len(missions.split())} to {max_mission_length} words")

    # Tokenize with better UNK handling
    tokens = [word_dict.get(word, UNK_IDX) for word in words]

    # Apply padding
    if len(tokens) < max_mission_length:
        tokens += [PAD_IDX] * (max_mission_length - len(tokens))

        batch_tokens.append(tokens)

    return torch.tensor(batch_tokens, dtype=torch.long).to(device=device)


# Create the Agent
agent = models.Agent(n_actions, vocab_size=len(word_dict) +2, hidden_dim=64, word_embedding_dim=32, text_embedding_dim=128, actor_cfc=False, critic_cfc=False)
agent.load_state_dict(torch.load('./src/ppo/baseline/curriculum/minigrid/models/MiniGrid-Fetch-5x5-N2-v0_R-0_94.pt'))
agent.to(device)

# Initialize the states for LNN prediction
batch_size = 1
# Make sure it's the same as the hidden_state size of the model trained on
# Ran into an error where hidden_dim was used instead, but it was a mismatch, leading to massive errors
hidden_state_size = 128
h = torch.zeros(batch_size, hidden_state_size).to(device)
c = torch.zeros(batch_size, hidden_state_size).to(device)
# actor_states = [h, c]
actor_states = None

done = False
import datetime
import time
start_time = time.time()
# print(f"Mission: {state['mission']}")
while not done:
    env.render()
    time.sleep(10)
    # time.sleep(1)
    # Permute the image to be (channels, height, width), then unsqueeze(0) to add batch of 1 to get (batch_size, channels, height, width)
    state_tensor = torch.tensor(np.array(state['image']), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    state_mission = process_mission(state['mission'], device=device)
    # Any pre-processing techniques need to be applied befor ewe use it
    with torch.no_grad():
        # INference the model and get the action
        # stochastic = False for argmax performance
        action, _, _, _ = agent.get_action_and_value(state_tensor, state_mission, actor_states)

    action = action.cpu().numpy()[0]
    state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated
    if done:
        print(f'Reward: {round(reward, 2)}')

print(f'Final training time: {datetime.timedelta(seconds=time.time() - start_time)}')
env.close() 