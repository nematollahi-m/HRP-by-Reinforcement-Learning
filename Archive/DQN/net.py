import torch.nn as nn
import torch
import torch.nn.functional as F


NUM_ACTIONS = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.fn1 = nn.Linear(3,32)
        self.fn2 = nn.Linear(32,64)
        self.fn3 = nn.Linear(64,128)
        self.fn4 = nn.Linear(128,NUM_ACTIONS)


    def forward(self, x):
        x = F.relu(self.fn1(x))
        x = F.relu(self.fn2(x))
        x = F.relu(self.fn3(x))
        x = F.sigmoid(self.fn4(x))
        return x