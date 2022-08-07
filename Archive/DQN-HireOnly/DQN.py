import main
import torch
import torch.nn as nn
import replay
import torch.optim as optim
import random
import math
from collections import namedtuple
from itertools import count
import NET
import matplotlib.pyplot as plt

REWARD_PRUNE = 20


def select_action(current_state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            current_state = torch.tensor(current_state,dtype=torch.float32)
            tmp = torch.argmax(q_net(current_state.to(device)))
            return torch.tensor([tmp])
    else:
        return torch.tensor([random.randrange(NUM_ACTIONS)], device=device, dtype=torch.long)


def optimize_model():
    # Not enough data to optimize the model
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = q_net(state_batch).gather(1, action_batch.reshape(action_batch.size(dim=0),1))
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in q_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# all the parameters
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

env = main.FarmEnv()
NUM_STATES = env.observation_space.n
print("Total number of states: ", NUM_STATES)
NUM_ACTIONS = env.action_space.n
print("Number of possible actions: ", NUM_ACTIONS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("available device is:", device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

memory_size = 1000
batch_size = 32


q_net = NET.DQN().to(device)
target_net = NET.DQN().to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(q_net.parameters())
memory = replay.ReplayMemory(10000)

steps_done = 0
episode_durations = []
num_episodes = 100

# A = [0.0,0.0,0.0]
# # # A = torch.zeros(3)
# B = torch.tensor(A)
# print(q_net(B))

total_rewards = []
tmp_reward = []

for i_episode in range(num_episodes):

    env.reset()
    track_sum_reward = 0
    full = False
    action_env = 0

    print(i_episode)
    for t in count():

        state = env.state

        if full is True:
            action = 0
        else:
            action = select_action(state)
            action = action.cpu().detach().numpy()
            action = action[0]

        next_state, reward, done, _, full, action_env = env.step(action)
        reward = torch.tensor([reward], device=device, dtype=torch.float32)
        torch_state = torch.tensor([state], dtype=torch.float32)
        torch_next_state = torch.tensor([next_state],dtype=torch.float32)
        torch_action = torch.tensor([action_env],dtype=torch.int64)
        memory.push(torch_state, torch_action, torch_next_state, reward)
        state = next_state
        # optimizing the model
        optimize_model()

        if done:
            episode_durations.append(t + 1)
            cost = state[1]
            plants = state[2]
            profit = plants * REWARD_PRUNE - cost
            tmp_reward.append(profit)
            break




plt.plot(tmp_reward)
plt.show()












