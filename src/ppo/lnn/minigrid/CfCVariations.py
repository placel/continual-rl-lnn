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

    def forward(self, x):
        # We only want the outputs for now; ignore hidden state
        x, _ = self.cfc1(x)
        x, _ = self.cfc2(x)
        return self.output(x)

class ActorCfC(nn.Module):

    def __init__(self, obs_space, action_space):
        super().__init__()

        # Create the CfC layer used for the actor, but keep the Dense layer in the end as normal passing it through layer_init()
        self.cfc1 = CfC(obs_space, 64)
        self.cfc2 = CfC(64, 64)
        self.output = layer_init(nn.Linear(64, action_space), std=0.01)

    def forward(self, x):
        # We only want the outputs for now; ignore hidden state
        x, _ = self.cfc1(x)
        x, _ = self.cfc2(x)
        return self.output(x)
