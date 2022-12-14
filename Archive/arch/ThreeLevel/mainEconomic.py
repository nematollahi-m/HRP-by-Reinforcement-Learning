import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
from Economic import EconomicEnv
import os
from stable_baselines3.common.monitor import Monitor
from SaveResults import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN
from Parameters import *
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

economic_env = EconomicEnv()

log_dir = "tmp/Economic3level/"
os.makedirs(log_dir, exist_ok=True)
economic_env = Monitor(economic_env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

economic_model = DQN('MlpPolicy', economic_env, exploration_fraction=0.5, exploration_final_eps=0.02,
                     exploration_initial_eps=1.0, verbose=1, device=device)
economic_model.learn(total_timesteps=time_steps, callback=callback)

# plotting results
plot_results(log_dir)

# Testing the model
obs_economic = economic_env.reset()
done_economic = False
n_steps = PRUNE_LENGTH

print("########################## Testing the model ############################")
for step in range(n_steps):

    mapping = tuple(np.ndindex((MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1,
                                MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1)))
    action_economic, _ = economic_model.predict(obs_economic, deterministic=True)
    action_economic_unroll = mapping[action_economic]
    obs_economic, reward_economic, done_economic, info = economic_env.step(action_economic)
    print(" *** Best Action *** ", action_economic_unroll)
    print('State = ', obs_economic, 'reward = ', reward_economic, 'done = ', done_economic)
    if done_economic is True:
        break
