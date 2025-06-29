import torch.nn as nn
import torch.nn.functional as F

class Curriculum(nn.Module):

    def __init__(self, n_actions, hidden_dim=256):
        super(Curriculum, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3))
        self.bn2 = nn.BatchNorm2d(32)

        # Compute the size of the flattened image for prediction
        self.flat_size = 32 * 3 * 3

        self.linear1 = nn.Linear(self.flat_size, hidden_dim)
        self.dropout = nn.Dropout(0.2)

        self.linear2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv1(x))

        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv2(x))
        
        # Actually flatten the image here
        x = x.view(x.size(0), -1)
        
        # x = self.dropout(F.relu(self.linear1(x)))
        x = F.relu(self.linear1(x))
        return self.linear2(x)