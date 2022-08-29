import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from typing import Optional, Union

MAX_ALLOWED_WORKER = 4
BUDGET = 6500
PLANTS = 45000
WAGE = 17
PRODUCTIVITY = 3
HIRE_COST = 30
FIRE_COST = 40
PRUNE_LENGTH = 31
REWARD_PRUNE = 8
action_size = np.prod((MAX_ALLOWED_WORKER,MAX_ALLOWED_WORKER))


class FarmEnv(Env):
    def __init__(self):

        super().__init__()
        self.state = np.asarray([0.0, 0.0, 0.0])
        self.current_employee = 0
        self.hired = 0.0
        self.fired = 0.0
        self.cost = 0.0
        self.plants = 0.0
        self.action_space = Discrete(np.prod((MAX_ALLOWED_WORKER,MAX_ALLOWED_WORKER)))
        self.observation_space = Discrete(np.prod((MAX_ALLOWED_WORKER,BUDGET,PLANTS)))
        self.current_step = 0
        self.profit = 0.0
        self.done = False

    def step(self, action):
        reward = 0
        self.current_step += 1
        mapping = tuple(np.ndindex((MAX_ALLOWED_WORKER, MAX_ALLOWED_WORKER)))
        new_action = mapping[action]
        self.hired = new_action[0]
        self.fired = new_action[1]

        self.current_employee += self.hired
        if self.current_employee < self.fired:
            reward -= 1000.0
            self.done = True

        # fired to add
        self.current_employee -= self.fired

        if self.current_employee > MAX_ALLOWED_WORKER:
            reward -= 1000.0
            self.done = True

        self.cost += (self.hired * HIRE_COST)
        self.cost += (self.current_employee * WAGE)
        self.cost += (self.fired * FIRE_COST)

        if self.cost > BUDGET:
            reward -= 1000.0
            self.done = True

        # change for each worker (TO DO- BOTH)
        self.plants += self.current_employee * round(np.random.normal(PRODUCTIVITY, 1))
        self.profit = self.plants * REWARD_PRUNE

        if self.plants >= PLANTS:
            reward += 1000.0
            self.done = True

        reward += (self.profit - self.cost)
        info = {}

        if self.current_step >= PRUNE_LENGTH:
            self.done = True

        self.state = np.asarray([self.current_employee, self.cost, self.plants])

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
        self.profit = 0.0
        self.state = np.asarray([0.0, 0.0, 0.0])
        self.cost = 0.0
        self.plants = 0.0
        self.hired = 0.0
        self.fired = 0.0
        self.current_step = 0
        self.done = False
        return self.state




