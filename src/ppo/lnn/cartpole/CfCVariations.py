import torch
import numpy as np
import torch.nn as nn
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

    def __init__(self, obs_space, action_space):
        super().__init__()

        # Create the CfC layer used for the actor, but keep the Dense layer in the end as normal passing it through layer_init()
        self.cfc1 = CfC(obs_space, 64)
        self.cfc2 = CfC(64, 64)
        self.output = layer_init(nn.Linear(64, action_space), std=1.0)

    def forward(self, x, states):
        # We only want the outputs for now; ignore hidden state
        s1, s2 = states
        outputs, new_s1, new_s2 = [], [], []

        # x is a batch of (4,4) envs, obs_space
        # We want to pass this through the model along with a hidden state for 
        # each env like (4, 64) (64 neurons in each hideen state).
        # Unlike the traditional MLP, CfC doesn't allow for this parralel processing
        # in the current version, and instead expects single experience processed at a time
        # So we need to unroll each indiviudal experience from different parallel envs
        # and pass them through with it's corrosponding hidden state.
        # However, we need to add a time dimension to each individual experience
        # for CfC processing, hence all the .unsqueeze(1) adding the extra time dim

        # So we are passing though each experience from parallel environments 
        # along with their respective hidden state, and the added time dimension

        for i in range(x.size(0)):
            # We also need to add a time dimension for the CfC 
            xi = x[i].unsqueeze(0)
            s1i = s1[i]
            s2i = s2[i]

            # Pass inputs to the model
            xi, new_s1i = self.cfc1(xi, s1i)
            xi, new_s2i = self.cfc2(xi, s2i)

            # Finally squeeze it back to orginal size
            outputs.append(xi.squeeze(0))
            new_s1.append(new_s1i)
            new_s2.append(new_s2i)
        
        # Restructure into original batches before returning
        x_out = torch.stack(outputs)
        new_s1 = torch.stack(new_s1)
        new_s2 = torch.stack(new_s2)

        # Pass batch into output layer (we don't need the temporal states for a fully connected)
        return self.output(x_out), [new_s1, new_s2]

class ActorCfC(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        # Enable batch_first for efficient batched processing
        self.cfc1 = CfC(obs_space, 64, batch_first=True)
        self.cfc2 = CfC(64, 64, batch_first=True)
        self.output = layer_init(nn.Linear(64, action_space), std=1.0)

    def forward(self, x, states):
        s1, s2 = states

        # Add time dimension: (batch_size, features) → (batch_size, 1, features)
        x = x.unsqueeze(1)

        # Run through CfC layers with batch support
        x, new_s1 = self.cfc1(x, s1)
        x, new_s2 = self.cfc2(x, new_s1)

        # Remove time dimension before output: (batch_size, 1, features) → (batch_size, features)
        x = x.squeeze(1)

        return self.output(x), [new_s1, new_s2]