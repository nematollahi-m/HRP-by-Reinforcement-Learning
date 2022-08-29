import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from typing import Optional, Union


MAX_ALLOWED_WORKER = 4
BUDGET = 650
PLANTS = 4500
WAGE = 17
PRODUCTIVITY = 3
HIRE_COST = 20
PRUNE_LENGTH = 31
REWARD_PRUNE = 20
action_size = MAX_ALLOWED_WORKER


class FarmEnv(Env):
    def __init__(self):
        super().__init__()
        # state = (number of workers, cost, plants)
        self.state = np.asarray([0.0, 0.0, 0.0])
        self.hired = 0.0
        self.cost = 0.0
        self.plants = 0.0
        self.action_space = Discrete(MAX_ALLOWED_WORKER + 1)
        self.observation_space = Discrete(np.prod((MAX_ALLOWED_WORKER + 1,BUDGET,PLANTS)))
        self.current_step = 0
        self.profit = 0.0
        self.action_env = 0
        self.done = False
        self.full = False

    def step(self, action):

        self.action_env = action
        reward = 0.0
        self.current_step += 1
        num_hire = action

        if self.hired + num_hire > MAX_ALLOWED_WORKER:
            num_hire = MAX_ALLOWED_WORKER - self.hired
            self.hired = MAX_ALLOWED_WORKER
            self.full = True
            self.action_env = num_hire
        else:
            self.hired += num_hire

        self.cost += (num_hire * HIRE_COST)
        self.cost += (self.hired * WAGE)

        if self.cost > BUDGET:
            reward -= 1000.0
            self.done = True

        self.plants += self.hired * round(np.random.normal(PRODUCTIVITY, 1))
        self.profit = self.plants * REWARD_PRUNE

        if self.plants >= PLANTS:
            self.profit += 1000.0
            self.done = True

        reward += (self.profit - self.cost)
        info = {}

        if self.current_step > PRUNE_LENGTH:
            self.done = True

        self.state = np.asarray([self.hired, self.cost, self.plants])

        return self.state, reward, self.done, info, self.full, self.action_env

    def render(self, mode="human"):
        pass

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        self.profit = 0.0
        self.state = np.asarray([0.0, 0.0, 0.0])
        self.cost = 0.0
        self.plants = 0.0
        self.hired = 0.0
        self.done = False
        self.full = False
        self.action_env = 0
        return self.state
