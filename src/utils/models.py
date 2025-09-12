import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP
from torch.distributions.categorical import Categorical

# Remove randomness from initialized weights
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SharedEmbedding(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
    
        # Process the image with regularization techinques like BatchNormalization (maybe dropout) later
        self.conv1 = layer_init(nn.Conv2d(3, 16, kernel_size=(2, 2)))    
        self.conv2 = layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2)))
        self.conv3 = layer_init(nn.Conv2d(32, 64, kernel_size=(2, 2)))
        self.max_pooling = nn.MaxPool2d((2,2))

        # flattene image for prediction
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(hidden_dim)
    
    def forward(self, image):
        
        # Process the image through CNN
        features = F.relu(self.conv1(image))
        features = self.max_pooling(features)
        features = F.relu(self.conv2(features))
        features = F.relu(self.conv3(features))

        # # Flatten the image 
        features = self.flatten(features)
        features = F.relu(self.linear1(features))

        return features

class SharedNetwork(nn.Module):

    def __init__(self, shared_embedding=None, embedding_dim=None, hidden_dim=64, hidden_state_dim=128, use_cfc=False, use_lstm=False):
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
            
    def forward(self, x, cfc_states=None, lstm_states=(None, None)):
        
        if cfc_states is None and lstm_states == (None, None):
            x = self.shared_embedding(x)
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

    def __init__(self, embedding_dim, hidden_dim=128, hidden_state_dim=128, use_cfc=False):
        super().__init__()

        # If we're using a CfC layer for the Actor Model, apply CfC
        # Otherwise use simple MLP
        if use_cfc:
            # self.cfc1 = CfC(embedding_dim, hidden_state_dim, batch_first=True) # batch_first=True is required with the ncps library
            self.cfc1 = CfC(embedding_dim, hidden_state_dim, batch_first=True) # batch_first=True is required with the ncps library
            # self.output = layer_init(nn.Linear(hidden_state_dim, 1), std=1.0)
        else:
            self.linear1 = layer_init(nn.Linear(embedding_dim, hidden_dim))
            # self.output = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        # Final output space of 1 as we want a value, not an action
    
    # DEPRECATED
    def forward(self, x, cfc_states=None):
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

        return x, cfc_states

class ActorHead(nn.Module):

    def __init__(self, embedding_dim, hidden_dim=128, hidden_state_dim=128, use_cfc=False):
        super().__init__()

        # If we're using a CfC layer for the Actor Model, apply CfC
        # Otherwise use simple MLP
        if use_cfc:
            self.cfc1 = CfC(embedding_dim, hidden_state_dim, batch_first=True) # batch_first=True is required with the ncps library
            # self.output = layer_init(nn.Linear(hidden_state_dim, action_space), std=0.01)
        else:
            self.linear1 = layer_init(nn.Linear(embedding_dim, hidden_dim))
            # self.output = layer_init(nn.Linear(hidden_dim, action_space), std=0.01)

        # Final output space of action_space to choose an action
    
    # DEPRECATED
    def forward(self, x, cfc_states=None):
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
        return x, cfc_states

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
            self.shared_cfc = SharedNetwork(None, embedding_dim, self.hidden_dim, self.hidden_state_dim, use_cfc=True)

            # Create the final head of actor and critic with hidden_state_dim (output of shared_cfc) as input
            self.critic_head = layer_init(nn.Linear(self.hidden_state_dim, 1), std=1.0)
            self.actor_head = layer_init(nn.Linear(self.hidden_state_dim, self.action_space), std=0.01)
        
        # If Actor & Critic heads are not both CfC, create them spearately (only used if enough time is left for extra experiments)
        elif self.actor_cfc and not self.critic_cfc:
            print('CfC Actor Head')
            # Add an extra Linear layer before critic output to match depth of CfC network (more fair)
            self.critic_head = nn.Sequential( 
                layer_init(nn.Linear(embedding_dim, self.hidden_dim), std=1.0),
                layer_init(nn.Linear(self.hidden_dim, 1), std=1.0)
            )

            # Create the actor model. Pass in the envs which will adapt depending on the tasks at hand
            self.actor_core = ActorHead(embedding_dim, self.hidden_dim, self.hidden_state_dim, use_cfc=True)
            self.actor_head = layer_init(nn.Linear(self.hidden_state_dim, self.action_space), std=1.0)
        
        elif self.critic_cfc and not self.actor_cfc:
            print('CfC Critic Head')
            self.critic_core = CriticHead(embedding_dim, self.hidden_dim, self.hidden_state_dim, use_cfc=True)
            self.critic_head = layer_init(nn.Linear(self.hidden_state_dim, 1), std=1.0)

            # Create the actor model. Pass in the envs which will adapt depending on the tasks at hand
            self.actor_head = nn.Sequential( 
                layer_init(nn.Linear(embedding_dim, self.hidden_dim), std=1.0),
                layer_init(nn.Linear(self.hidden_dim, self.action_space), std=1.0)
            )
        # Create the SharedNetwork with LSTM layer 
        elif self.use_lstm:
            print('LSTM Model')
            self.shared_baseline = SharedNetwork(None, embedding_dim, self.hidden_dim, hidden_state_dim=self.hidden_state_dim, use_lstm=True)

            # Init LSTM params. See https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py
            for name, param in self.shared_baseline.lstm1.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)

            # Create the final head of actor and critic with hidden_state_dim (output of shared_baseline) as input
            self.critic_head = layer_init(nn.Linear(self.hidden_state_dim, 1), std=1.0)
            self.actor_head = layer_init(nn.Linear(self.hidden_state_dim, self.action_space), std=0.01)

        # Create the baseline model without CfC
        else:
            print('Baseline Model')
            self.shared_baseline = SharedNetwork(self.shared_embedding, embedding_dim, self.hidden_dim, use_cfc=False)
            self.critic_head = layer_init(nn.Linear(self.hidden_dim, 1), std=1.0)
            self.actor_head = layer_init(nn.Linear(self.hidden_dim, self.action_space), std=0.01)

    # See https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py
    def _lstm_roll(self, core, hidden, states, done):
        batch_size = states[0].shape[1]
        hidden = hidden.reshape(-1, batch_size, core.input_size)
        done = done.reshape(-1, batch_size)
        new_hidden = []
        for h, d in zip(hidden, done):
            h, states = core(
                h.unsqueeze(1),
                (
                    (1.0 - d).view(1, -1, 1) * states[0], # Reset hidden states to 0 if the environment is temrinated. Reseting ensures states don't cross into the next episode
                    (1.0 - d).view(1, -1, 1) * states[1]
                )
            )
            new_hidden += [h.squeeze(1)]
        new_hidden = torch.cat(new_hidden) 
        return new_hidden, states

    # Derived from _lstm_roll found above
    def _cfc_roll(self, cell, hidden, state, done):
        batch_size = state.shape[0]
        hidden = hidden.reshape(-1, batch_size, cell.input_size)
        done = done.reshape(-1, batch_size)
        new_hidden = []
        for h, d in zip(hidden, done):
            mask = (1.0 - d).view(batch_size, 1) # Apply time dimension
            h, state = cell(h.unsqueeze(1), state * mask) # Only one state dimension for CfC
            new_hidden += [h.squeeze(1)] # Remove time dimension
        new_hidden = torch.cat(new_hidden)
        return new_hidden, state

    def get_cfc_states(self, x, cfc_state=None, cfc_type='critic', dones=None):
        embedding = self.shared_embedding(x)

        # Assign the cfc module
        if cfc_type == 'shared':
            cell = self.shared_cfc.cfc1
        elif cfc_type == 'actor':
            cell = self.actor_core.cfc1
        elif cfc_type == 'critic':
            cell = self.critic_core.cfc1

        out, cfc_state = self._cfc_roll(cell, embedding, cfc_state, dones)
        return out, cfc_state

    # Get the LSTM states and update accordingly based on done
    def get_lstm_states(self, x, lstm_state=None, dones=None):
        embedding = self.shared_embedding(x)                            
        out, lstm_state = self._lstm_roll(self.shared_baseline.lstm1, embedding, lstm_state, dones)
        return out, lstm_state

    def get_value(self, image, cfc_states=None, lstm_states=(None, None), dones=None):
        # If sharing network, process image and states through the shared network
        if self.shared_cfc:
            output, _ = self.get_cfc_states(image, cfc_states, 'shared', dones) 
        # If not using a shared network, pass states into critic separately (LNN_Actor XOR LNN_Critic)
        elif self.actor_cfc and not self.critic_cfc:
            output, _ = self.get_cfc_states(image, cfc_states, 'actor', dones)
        elif self.critic_cfc and not self.actor_cfc:
            output, _ = self.get_cfc_states(image, cfc_states, 'critic', dones)
        # If using baseline process through shared baseline network then to critic
        elif self.use_lstm: 
            output, _ = self.get_lstm_states(image, lstm_states, dones)
        elif self.shared_baseline: 
            output, _, _ = self.shared_baseline(image) # lstm_states is None by default. If LSTM isn't being used, this won't be an issue
            
        return self.critic_head(output)
    
    def get_action_and_value(self, image, cfc_states=None, lstm_states=(None, None), action=None, dones=None, deterministic=False):
        if self.shared_cfc:
            output, cfc_states = self.get_cfc_states(image, cfc_states, 'shared', dones)
        elif self.actor_cfc and not self.critic_cfc:
            output, cfc_states = self.get_cfc_states(image, cfc_states, 'actor', dones)
        elif self.critic_cfc and not self.actor_cfc:
            output, cfc_states = self.get_cfc_states(image, cfc_states, 'critic', dones)
        elif self.use_lstm:
            output, lstm_states = self.get_lstm_states(image, lstm_states, dones)
        # Lastly, if no CfC or LSTM is being used (MLP), process normally
        else:
            output, _, _ = self.shared_baseline(image)

        logits = self.actor_head(output)
        value = self.critic_head(output)
        probs = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        elif action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value, cfc_states, lstm_states, logits
    
    # Needed for CLEAR to get the on-policy values for v-trace calculation
    def evaluate_actions(self, image, actions, cfc_states=None, lstm_states=(None, None), dones=None):
        _, logp, ent, value, cfc_states, lstm_states, logits = self.get_action_and_value(image, cfc_states=cfc_states, lstm_states=lstm_states, action=actions, dones=dones)

        return logp, value, cfc_states, lstm_states