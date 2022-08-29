from typing import Optional, Union

import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import matplotlib.pyplot as plt

MAX_ALLOWED_WORKER = 5
BUDGET = 5000
PLANTS = 10000
WAGE = 20
PRODUCTIVITY = 700
HIRE_COST = 80
PRUNE = 31
REWARD_PRUNE = 8
PENALTY_UNPRUNE = 1

class FarmEnv(Env):
    def __init__(self):

        super().__init__()

        self.state = 0
        self.hired = 0
        self.cost = 0
        self.plants = 0
        self.action_space = Discrete(MAX_ALLOWED_WORKER + 1)
        self.observation_space = Discrete(PRUNE)
        self.current_step = 0
        self.profit = 0
        self.done = False

    def step(self, action):

        self.current_step += 1
        self.state = self.state + 1
        num_hire = action
        self.hired += num_hire
        if self.hired > MAX_ALLOWED_WORKER:
            num_hire = MAX_ALLOWED_WORKER - self.hired
            self.hired = MAX_ALLOWED_WORKER

        self.cost += (num_hire * HIRE_COST)
        self.cost += (self.hired * WAGE)
        if self.cost > BUDGET:
            self.cost += 1000
            self.done = True

        self.plants += self.hired * round(np.random.normal(PRODUCTIVITY, 1))
        self.profit = self.plants * REWARD_PRUNE

        if self.plants >= PLANTS:
            self.profit += 1000
            self.done = True

        reward = self.profit - self.cost

        info = {}

        if self.current_step >= PRUNE:
            self.done = True

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
        self.state = 0
        self.cost = 0
        self.plants = 0
        self.hired = 0
        self.current_step = 0
        self.done = False
        return self.state


env = FarmEnv()

q_table = np.zeros([env.observation_space.n, env.action_space.n])
print(q_table.shape)
env.action_space.sample()

alpha = 0.1
gamma = 0.6
epsilon = 0.2

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100):

    r = 0
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done1 = False

    while not done1:

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state, :])  # Exploit learned values

        next_state, reward, done1, info = env.step(action)

        if next_state == PRUNE:
            q_table[state, action] = 0
        else:
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

        r += reward

        state = next_state

    all_penalties.append(r/100)
    if i % 100 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")
plt.plot(all_penalties)
plt.show()

print(q_table)