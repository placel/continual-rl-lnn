import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CriticCfC(nn.Module):

    def __init__(self, obs_space):
        super().__init__()

        # Create the CfC layer used for the actor, but keep the Dense layer in the end as normal passing it through layer_init()
        self.cfc1 = CfC(obs_space, 64)
        self.cfc2 = CfC(64, 64)
        self.output = layer_init(nn.Linear(64, 1), std=0.01)

    def forward(self, x, states):
        # We only want the outputs for now; ignore hidden state
        s1, s2 = states
        x, s1 = self.cfc1(x, s1)
        x, s2 = self.cfc2(x, s2)
        return self.output(x), [s1, s2]

class ActorCfC(nn.Module):

    def __init__(self, action_space, hidden_dim):
        super().__init__()
        # Process the image with regularization techinques like BatchNormalization (maybe dropout later)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3))
        self.bn2 = nn.BatchNorm2d(32)

        # flattene image for prediction
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(hidden_dim)
        # self.dropout = nn.Dropout(0.2)

        # Create the CfC layer used for the actor, but keep the Dense layer in the end as normal passing it through layer_init()
        self.cfc1 = CfC(hidden_dim, 64, batch_first=True)
        self.cfc2 = CfC(64, 64, batch_first=True)
        self.output = layer_init(nn.Linear(64, action_space), std=1.0)

    def forward(self, x, states):
        # State storage
        s1, _ = states

        # Process the obs image first
        x = F.relu(self.conv1(x))
        
        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv2(x))
        
        # Flatten the image here
        x = self.flatten(x)
        x = F.relu(self.linear1(x))

        # Reshaping to add the time dimension in the 1st index so we have (batch_size, time_dim, features)
        # Features is all feature maps sequentially combined  
        x = x.unsqueeze(1)

        # Pass features to model along with state_1
        x, new_s1 = self.cfc1(x, s1)
        # Pass updated features along with new_states from self.cfc1
        # and store new_s2
        x, new_s2 = self.cfc2(x, new_s1)

        # Remove the time dimension we added earlier as it's not needed anymore
        x = x.squeeze(1)

        # Pass batch into output layer (we don't need the temporal states for a fully connected)
        # return states as well for update epoch training
        return self.output(x), [new_s1, new_s2]
