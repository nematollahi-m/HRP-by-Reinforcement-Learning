import Fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import replay
import torch.optim as optim
import random
import math
from collections import namedtuple, deque
from itertools import count


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.stack(list(tuple_of_tensors), dim=0)


env = Fire.FarmEnv()
NUM_STATES = env.observation_space.n
print("Total number of states: ", NUM_STATES)
NUM_ACTIONS = env.action_space.n
print("Number of possible actions: ", NUM_ACTIONS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("available device is:", device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

memory_size = 10000
batch_size = 32


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.fn1 = nn.Linear(3,32)
        self.fn2 = nn.Linear(32,64)
        self.fn3 = nn.Linear(64,128)
        self.fn4 = nn.Linear(128,NUM_ACTIONS)

    def forward(self,x):
        x = x.to(device)
        x = F.relu(self.fn1(x))
        x = F.relu(self.fn2(x))
        x = F.relu(self.fn3(x))
        x = F.sigmoid(self.fn4(x))
        return x


# training the network
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

q_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(q_net.parameters())
#memory = replay.ReplayMemory(memory_size,batch_size)
memory = replay.ReplayMemory(10000)

steps_done = 0


def select_action(current_state):

    current_state = torch.tensor(current_state)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return q_net(current_state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(NUM_ACTIONS)]], device=device, dtype=torch.long)


episode_durations = []


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

    state_action_values = q_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in q_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
for i_episode in range(num_episodes):
    env.reset()
    for t in count():

        state = env.state
        action = select_action(state)
        action = action.cpu().detach().numpy()
        # converting the tensor to int
        action = action[0]
        action = action[0]
        next_state, reward, done, _ = env.step(action)

        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward)
        state = next_state
        # optimizing the model
        optimize_model()

        # if done:
        #     episode_durations.append(t + 1)
        #     break













