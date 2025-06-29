import torch
import torch.nn as nn
import torch.nn.functional as F

class CartPoleModel(nn.Module):

    # n_observation = number of input variables (state variables)
    # n_actions = number of actions possible; number of output nodes
    def __init__(self, n_observations, n_actions, n_nodes):
        super(CartPoleModel, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_nodes)
        self.layer2 = nn.Linear(n_nodes, n_nodes)
        self.layer3 = nn.Linear(n_nodes, n_actions)

    # Pass the inputs through the network
    # Return the results
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)