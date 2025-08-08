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
    def __init__(self, hidden_dim=64):
        super().__init__()
    
        # Process the image with regularization techinques like BatchNormalization (maybe dropout later)
        # self.conv1 = layer_init(nn.Conv2d(3, 16, kernel_size=(3, 3), stride=1, padding=1), std=1.0)    
        # self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1), std=1.0)
        # self.conv3 = layer_init(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1), std=1.0)
        self.conv1 = layer_init(nn.Conv2d(3, 16, kernel_size=(2, 2)))    
        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2)))
        self.conv3 = layer_init(nn.Conv2d(32, 64, kernel_size=(2, 2)))
        self.max_pooling = nn.MaxPool2d((2,2))
        # self.conv3 = layer_init(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1), std=1.0)

        # flattene image for prediction
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear2 = layer_init(nn.Linear(hidden_dim, hidden_dim))

        # TODO Store model params in json or something
    
    def forward(self, image):
        
        # Process the image through CNN
        features = F.relu(self.conv1(image))
        features = self.max_pooling(features)
        features = F.relu(self.conv2(features))
        features = F.relu(self.conv3(features))

        # Flatten the image 
        features = self.flatten(features)
        features = F.relu(self.linear1(features))

        return features


class SharedNetwork(nn.Module):

    def __init__(self, shared_embedding, embedding_dim, hidden_dim=64, hidden_state_dim=128, use_cfc=False, use_lstm=False):
        super().__init__()
        
        self.shared_embedding = shared_embedding

        # If we're using a CfC layer for the Actor Model, apply CfC
        # Otherwise use simple MLP
        if use_cfc:
            self.cfc1 = CfC(embedding_dim, hidden_state_dim, batch_first=True)
        if use_lstm:
            self.lstm1 = nn.LSTM(embedding_dim, hidden_state_dim, batch_first=True)
        # Use 2 final layers for some 'fairness'
        else:
            self.linear1 = layer_init(nn.Linear(embedding_dim, hidden_dim))
            self.linear2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
            
    def forward(self, image, cfc_states=None, lstm_states=(None, None)):

        x = self.shared_embedding(image)

        # If the model uses a CfC Layer, states need to be managed
        # This results in a different forward structure
        # If the model (self) has the attribute 'cfc1', it was initialized to have a CfC layer, and this will execute
        if hasattr(self, 'cfc1'):
            # Reshaping to add the time dimension in the 1st index so we have (batch_size, time_dim, features)
            # Features is all feature maps sequentially combined  
            x = x.unsqueeze(1)

            # Process through CfC layer, extracting model states
            x, cfc_states = self.cfc1(x, cfc_states)
            # Remove extra time dimension
            x = x.squeeze(1)
        # Same case for LSTM, states need proper management
        elif hasattr(self, 'lstm1'):
            # Reshaping to add the time dimension in the 1st index so we have (batch_size, time_dim, features)
            # Features is all feature maps sequentially combined  
            x = x.unsqueeze(1)

            # print('STATES')
            # print(lstm_states)
            # Process through CfC layer, extracting model states
            x, lstm_states = self.lstm1(x, lstm_states)
            # Remove extra time dimension
            x = x.squeeze(1)
        # Process as normal
        else:
            # Otherwise, process as normal
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))

        return x, cfc_states, lstm_states

class CriticHead(nn.Module):

    def __init__(self, shared_embedding, embedding_dim, hidden_dim=128, hidden_state_dim=128, use_cfc=False):
        super().__init__()
        self.shared_embedding = shared_embedding

        # If we're using a CfC layer for the Actor Model, apply CfC
        # Otherwise use simple MLP
        if use_cfc:
            self.cfc1 = CfC(embedding_dim, hidden_state_dim, batch_first=True) # batch_first=True is required with the ncps library
            self.output = layer_init(nn.Linear(hidden_state_dim, 1), std=1.0)
        else:
            self.linear1 = layer_init(nn.Linear(embedding_dim, hidden_dim))
            self.output = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        # Final output space of 1 as we want a value, not an action
    
    def forward(self, image, cfc_states=None):
        x = self.shared_embedding(image)

        # If the model uses a CfC Layer, states need to be managed
        # This results in a different forward structure
        # If the model (self) has the attribute 'cfc1', it was initialized to have a CfC layer, and this will execute
        if hasattr(self, 'cfc1'):
            # Reshaping to add the time dimension in the 1st index so we have (batch_size, time_dim, features)
            # Features is all feature maps sequentially combined  
            x = x.unsqueeze(1)

            # Process through CfC layer, extracting model states
            x, cfc_states = self.cfc1(x, cfc_states)

            # Remove extra time dimension
            x = x.squeeze(1)

        # Process as normal
        else:
            # Otherwise, process as normal
            x = F.relu(self.linear1(x))

        return self.output(x), cfc_states

class ActorHead(nn.Module):

    def __init__(self, shared_embedding, embedding_dim, action_space, hidden_dim=128, hidden_state_dim=128, use_cfc=False):
        super().__init__()

        self.shared_embedding = shared_embedding

        # If we're using a CfC layer for the Actor Model, apply CfC
        # Otherwise use simple MLP
        if use_cfc:
            self.cfc1 = CfC(embedding_dim, hidden_state_dim, batch_first=True) # batch_first=True is required with the ncps library
            self.output = layer_init(nn.Linear(hidden_state_dim, action_space), std=0.01)
        else:
            self.linear1 = layer_init(nn.Linear(embedding_dim, hidden_dim))
            self.output = layer_init(nn.Linear(hidden_dim, action_space), std=0.01)

        # Final output space of action_space to choose an action
    
    def forward(self, image, cfc_states=None):

        x = self.shared_embedding(image)

        # If the model uses a CfC Layer, states need to be managed
        # This results in a different forward structure
        if hasattr(self, 'cfc1'):
            # Reshaping to add the time dimension in the 1st index so we have (batch_size, time_dim, features)
            # Features is all feature maps sequentially combined  
            x = x.unsqueeze(1)

            # Process through CfC layer and get states
            x, cfc_states = self.cfc1(x, cfc_states)

            # Remove extra time dimension
            x = x.squeeze(1)

        # Process the data as normal
        else:
            x = F.relu(self.linear1(x))

        # Otherwise, process as normal
        return self.output(x), cfc_states

# Combine the Actor & Critic into a single model 
class Agent(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        # Extract params
        self.action_space = config['action_space']
        self.hidden_dim = config['hidden_dim']
        self.hidden_state_dim = config['hidden_state_dim']
        self.actor_cfc = config['actor_cfc']
        self.critic_cfc = config['critic_cfc']
        self.use_lstm = config['use_lstm']
        
        self.shared_embedding = SharedEmbedding(self.hidden_dim) # Shared image embedding
        self.shared_cfc = None # Shared CfC layer
        self.shared_baseline = None # Shared MLP layer

        # Compute the output size dynamically; needed as input for critic and actor
        with torch.no_grad():
            dummy_image = torch.zeros(1, 3, 7, 7)  # Example MiniGrid obs (C=3,H=7,W=7)
            shared_out = self.shared_embedding(dummy_image)
            embedding_dim = shared_out.shape[1]
        
        # If both Actor and Critic are using CfC, share the network to ensure consistency
        if self.actor_cfc and self.critic_cfc:
            print('Actor & Critic CfC')
            # Pass the embedding to the CfC and return 
            # Output will be hidden_size as no reduction occurs. 
            self.shared_cfc = SharedNetwork(self.shared_embedding, embedding_dim, self.hidden_dim, self.hidden_state_dim, use_cfc=True)

            # Create the final head of actor and critic with hidden_state_dim (output of shared_cfc) as input
            self.critic = layer_init(nn.Linear(self.hidden_state_dim, 1), std=1.0)
            self.actor = layer_init(nn.Linear(self.hidden_state_dim, self.action_space), std=0.01)
        
        # If Actor & Critic heads are not both CfC, create them spearately (only used if enough time is left for extra experiments)
        elif self.actor_cfc or self.critic_cfc:
            print('CfC Single Head')
            # Create the critic model with the same variables as the CfC
            self.critic = CriticHead(self.shared_embedding, embedding_dim, self.hidden_dim, self.hidden_state_dim, self.critic_cfc)

            # Create the actor model. Pass in the envs which will adapt depending on the tasks at hand
            self.actor = ActorHead(self.shared_embedding, embedding_dim, self.action_space, self.hidden_dim, self.hidden_state_dim, self.actor_cfc)

        # Create the SharedNetwork with LSTM layer 
        elif self.use_lstm:
            print('LSTM Model')
            self.shared_baseline = SharedNetwork(self.shared_embedding, embedding_dim, self.hidden_dim, hidden_state_dim=self.hidden_state_dim, use_lstm=True)

            # Create the final head of actor and critic with hidden_state_dim (output of shared_baseline) as input
            self.critic = layer_init(nn.Linear(self.hidden_state_dim, 1), std=1.0)
            self.actor = layer_init(nn.Linear(self.hidden_state_dim, self.action_space), std=0.01)

        # Create the baseline model without CfC
        else:
            print('Baseline Model')
            self.shared_baseline = SharedNetwork(self.shared_embedding, embedding_dim, self.hidden_dim, use_cfc=False)

            self.critic = layer_init(nn.Linear(self.hidden_dim, 1), std=1.0)
            self.actor = layer_init(nn.Linear(self.hidden_dim, self.action_space), std=0.01)

    def get_value(self, image, cfc_states=None, lstm_states=(None, None)):
        # If sharing network, process image and states through the shared network
        if self.shared_cfc:
            output, _, _ = self.shared_cfc(image, cfc_states)
            value = self.critic(output)

        # If using baseline process through shared baseline network then to critic
        elif self.shared_baseline: 
            output, _, _ = self.shared_baseline(image, lstm_states=lstm_states) # lstm_states is None by default. If LSTM isn't being used, this won't be an issue
            value = self.critic(output)

        # If not using a shared network, pass states into critic separately (LNN_Actor XOR LNN_Critic)
        else:
            value, _, _ = self.critic(image, cfc_states)
            
        return value
    
    def get_action_and_value(self, image, cfc_states=None, lstm_states=(None, None), action=None, deterministic=False):
        # If using a shared network, process accordingly passing image and states to shared_network
        if self.shared_cfc:
            output, cfc_states, _ = self.shared_cfc(image, cfc_states)

            logits = self.actor(output)
            value = self.critic(output)
        
        # If only using only actor_cfc or critic_cfc, process input differently passing states to respective head
        elif self.actor_cfc:
            # Pass the image, states, and mission to the model to get an action
            logits, cfc_states = self.actor(image, cfc_states)
            value, _ = self.critic(image)

        elif self.critic_cfc:
            # Pass the image, states, and mission to the model to get an action
            logits, _ = self.actor(image)
            value, cfc_states = self.critic(image, cfc_states)
        
        # Lastly, if no CfC is being used (baseline), process normally
        else:
            output, _, lstm_states = self.shared_baseline(image, lstm_states=lstm_states)
            logits = self.actor(output)
            value = self.critic(output)

        probs = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        elif action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value, cfc_states, lstm_states, logits