import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
from CostObjective import FarmEnvCost
import os
from stable_baselines3.common.monitor import Monitor
from SaveResults import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN
from parameters import *
import torch as th

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

if th.cuda.is_available():
    print('GPU is available')
    device = 'cuda'
else:
    print('No GPU')
    device = 'cpu'

Cost_ENV = FarmEnvCost()

log_dir = "tmp/cost/"
os.makedirs(log_dir, exist_ok=True)
env_cost = Monitor(Cost_ENV, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

cost_model = DQN('MlpPolicy', env_cost, exploration_fraction=0.8, exploration_final_eps=0.02,
                 exploration_initial_eps=1.0, verbose=1, device=device)
cost_model.learn(total_timesteps=time_steps, callback=callback)

# plotting results
plot_results(log_dir)


# Testing the model
obs_cost = env_cost.reset()
done_cost = False

n_steps = PRUNE_LENGTH
print("########################## Testing the model ############################")
for step in range(n_steps):

    mapping = tuple(np.ndindex((MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1)))
    action_cost, _ = cost_model.predict(obs_cost, deterministic=True)
    action_cost_unroll = mapping[action_cost]
    obs_cost, reward_cost, done_cost, info = env_cost.step(action_cost)
    print(" *** Best Action *** ", action_cost_unroll)
    print('State = ', obs_cost, 'reward = ', reward_cost, 'done = ', done_cost)
    if done_cost is True:
        break

print('Total cost of pruning: ', BUDGET - obs_cost[1], ', Initial Budget: ', BUDGET)
print('Total Pruned Plants: ', PLANTS - obs_cost[2], ', Initial Number of Plants: ', PLANTS)

