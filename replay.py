# import numpy as np
#
#
# # Experience Replay (or Replay memory)
# class ReplayMemory():
#     def __init__(self, memory_size, batch_size):
#
#         # max number of samples to store
#         self.memory_size = memory_size
#
#         # batch size
#         self.minibatch_size = batch_size
#
#         self.experience = [None] * self.memory_size
#         self.current_index = 0
#         self.size = 0
#
#     # we need to store the experience
#     def store(self, observation, action, reward, new_observation, isterminal):
#
#         # store the experience as a tuple (current state, action, reward, next state, is it a terminal state)
#         self.experience[self.current_index] = (observation, action, reward, new_observation, isterminal)
#         self.current_index += 1
#
#         self.size = min(self.size + 1, self.memory_size)
#
#         # if the index is greater than  memory then we flush the index by subtracting it with memory size
#
#         if self.current_index >= self.memory_size:
#             self.current_index -= self.memory_size
#
#     # sample function is used to sample data.
#     def sample(self):
#
#         if self.size < self.minibatch_size:
#             return []
#
#         # randomly sample some indices
#         samples_index = np.floor(np.random.random((self.minibatch_size,))*self.size)
#
#         # select the experience from the sampled index
#         samples = [self.experience[int(i)] for i in samples_index]
#
#         return samples
#
from collections import namedtuple, deque
import random
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)