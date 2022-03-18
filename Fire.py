import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

MAX_ALLOWED_WORKER = 4
BUDGET = 6500
PLANTS = 45000
WAGE = 17
PRODUCTIVITY = 3
HIRE_COST = 30
FIRE_COST = 40
PRUNE_LENGTH = 31
REWARD_PRUNE = 8

class FarmEnv(Env):
    def __init__(self):
        super().__init__()

        self.state = (0, BUDGET, PLANTS)
        self.current_employee = 0
        self.hired = 0
        self.fired = 0
        self.cost = 0
        self.plants = 0
        self.action_space = Discrete(np.prod((MAX_ALLOWED_WORKER,MAX_ALLOWED_WORKER)))
        self.observation_space = Discrete(np.prod((MAX_ALLOWED_WORKER,BUDGET,PLANTS)))
        self.current_step = 0
        self.profit = 0
        self.done = False

    def step(self, action):
        reward = 0
        self.current_step += 1
        mapping = tuple(np.ndindex((MAX_ALLOWED_WORKER, MAX_ALLOWED_WORKER)))
        new_action = mapping[action]
        self.hired = new_action[0]
        self.fired = new_action[1]
        # firedd to add
        self.current_employee += self.hired
        # to be checked

        if self.current_employee < self.fired:
            reward -= 1000
            # continue
            self.done = True
            #self.current_employee = 0

        # to be checked
        if self.current_employee > MAX_ALLOWED_WORKER:
            num_hire = MAX_ALLOWED_WORKER - self.current_employee
            self.hired = MAX_ALLOWED_WORKER

        self.cost += (self.hired * HIRE_COST)
        self.cost += (self.current_employee * WAGE)
        self.cost += (self.fired * FIRE_COST)

        if self.cost > BUDGET:
            reward -= 1000
            self.done = True

        # change for each worker (TO DO- BOTH)
        self.plants += self.current_employee * round(np.random.normal(PRODUCTIVITY, 1))
        self.profit = self.plants * REWARD_PRUNE

        if self.plants >= PLANTS:
            reward += 1000
            self.done = True

        reward += (self.profit - self.cost)
        info = {}

        if self.current_step >= PRUNE_LENGTH:
            self.done = True

        self.state = (self.current_employee, self.plants, BUDGET - self.cost)

        return self.state, reward, self.done, info

    def render(self, mode="human"):
        pass

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        self.profit = 0
        self.state = (0, BUDGET, PLANTS)
        self.cost = 0
        self.plants = 0
        self.hired = 0
        self.fired = 0
        self.current_step = 0
        self.done = False
        return self.state

env = FarmEnv()
NUM_STATES = env.observation_space.n
print(NUM_STATES)
NUM_ACTIONS = env.action_space.n
print(NUM_ACTIONS)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))