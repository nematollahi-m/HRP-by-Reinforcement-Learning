import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
from Environment import EnvironmentEnv
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

Env_EnV = EnvironmentEnv()

log_dir = "tmp/env3level/"
os.makedirs(log_dir, exist_ok=True)
env_env = Monitor(Env_EnV, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)


env_model =DQN('MlpPolicy', env_env, exploration_fraction=0.5, exploration_final_eps=0.02,
               exploration_initial_eps=1.0, verbose=1,device=device)
env_model.learn(total_timesteps=time_steps)

# plotting results
plot_results(log_dir)

# Testing the model
obs_env = env_env.reset()
done_env = False
n_steps = PRUNE_LENGTH

print("########################## Testing the model ############################")
for step in range(n_steps):

    mapping = tuple(np.ndindex((MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1,
                                MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1)))
    action_env, _ = env_model.predict(obs_env, deterministic=True)
    action_env_unroll = mapping[action_env]
    obs_env, reward_env, done_env, info = env_env.step(action_env)
    print(" *** Best Action *** ", action_env_unroll)
    print('State = ', obs_env, 'reward = ', reward_env, 'done = ', done_env)
    if done_env is True:
        break
