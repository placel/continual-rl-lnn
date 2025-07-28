import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SharedEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
    
        # Process the image with regularization techinques like BatchNormalization (maybe dropout later)
        self.conv1 = layer_init(nn.Conv2d(3, 16, kernel_size=(3, 3), stride=2, padding=1), std=1.0)    
        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1), std=1.0)
        self.conv3 = layer_init(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1), std=1.0)

        # flattene image for prediction
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear2 = layer_init(nn.Linear(hidden_dim, hidden_dim))

        # TODO Store model params in json or something
    
    def forward(self, image):
        
        # Process the image through CNN
        features = F.relu(self.conv1(image))
        features = F.relu(self.conv2(features))
        features = F.relu(self.conv3(features))

        # Flatten the image 
        features = self.flatten(features)
        features = F.relu(self.linear1(features))

        return features
        
class CriticHead(nn.Module):

    def __init__(self, shared_embedding, embedding_dim, hidden_dim=128, use_cfc=False):
        super().__init__()
        
        self.shared_embedding = shared_embedding

        # If we're using a CfC layer for the Actor Model, apply CfC
        # Otherwise use simple MLP
        if use_cfc:
            self.cfc1 = CfC(embedding_dim, hidden_dim, batch_first=True) # batch_first=True is required with the ncps library
        else:
            self.linear1 = layer_init(nn.Linear(embedding_dim, hidden_dim))

        # Final output space of 1 as we want a value, not an action
        self.output = layer_init(nn.Linear(hidden_dim, 1))
    
    def forward(self, image, cfc_states=None):

        x = self.shared_embedding(image)

        # If the model uses a CfC Layer, states need to be managed
        # This results in a different forward structure
        if cfc_states:
            # Reshaping to add the time dimension in the 1st index so we have (batch_size, time_dim, features)
            # Features is all feature maps sequentially combined  
            x = x.unsqueeze(1)

            # Process through CfC layer
            x, new_s1 = self.cfc1(x, cfc_states)
            
            # Remove extra time dimension
            x = x.squeeze(1)

            # TODO Finish implementing 
            return self.output(x)
        
        # Otherwise, process as normal
        x = torch.tanh(self.linear1(x))
        return self.output(x)

class ActorHead(nn.Module):

    def __init__(self, shared_embedding, embedding_dim, action_space, hidden_dim=128, use_cfc=False):
        super().__init__()

        self.shared_embedding = shared_embedding

        # If we're using a CfC layer for the Actor Model, apply CfC
        # Otherwise use simple MLP
        if use_cfc:
            self.cfc1 = CfC(embedding_dim, hidden_dim, batch_first=True) # batch_first=True is required with the ncps library
        else:
            self.linear1 = layer_init(nn.Linear(embedding_dim, hidden_dim), std=1.0)

        # Final output space of action_space to choose an action
        self.output = layer_init(nn.Linear(hidden_dim, action_space))
    
    def forward(self, image, cfc_states=None):

        x = self.shared_embedding(image)

        # If the model uses a CfC Layer, states need to be managed
        # This results in a different forward structure
        if cfc_states:
            # Reshaping to add the time dimension in the 1st index so we have (batch_size, time_dim, features)
            # Features is all feature maps sequentially combined  
            x = x.unsqueeze(1)

            # Process through CfC layer
            x, new_s1 = self.cfc1(x, cfc_states)
            
            # Remove extra time dimension
            x = x.squeeze(1)

            # TODO Finish implementing 
            return self.output(x)
        
        # Otherwise, process as normal
        x = torch.tanh(self.linear1(x))
        return self.output(x)

# Combine the Actor & Critic into a single model 
class Agent(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        # Extract params
        action_space = config['action_space']
        hidden_dim = config['hidden_dim']
        actor_cfc = config['actor_cfc']
        critic_cfc = config['critic_cfc']
        
        self.shared_embedding = SharedEmbedding(hidden_dim)
        
        # Compute the output size dynamically
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, 7, 7)  # Example MiniGrid obs (C=3,H=7,W=7)
            shared_out = self.shared_embedding(dummy_image)
            embedding_dim = shared_out.shape[1]
        
        # Create the critic model with the same variables as the CfC
        self.critic = CriticHead(self.shared_embedding, embedding_dim, hidden_dim, critic_cfc)

        # Extract the action space of the environment
        # action_space = np.array(envs.single_action_space).prod()

        # Create the actor model. Pass in the envs which will adapt depending on the tasks at hand
        self.actor = ActorHead(self.shared_embedding, embedding_dim, action_space, hidden_dim, actor_cfc)

    def get_value(self, image, states=None):
        return self.critic(image, states)
    
    def get_action_and_value(self, image, states=None, action=None, determinstic=False):
        # Pass the image, states, and mission to the model to get an action
        logits = self.actor(image, states)
        # probs = Categorical(logits=F.log_softmax(logits, dim=1))
        probs = Categorical(logits=logits)
        
        if determinstic:
            action = torch.argmax(logits, dim=1)
        elif action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(image, states)