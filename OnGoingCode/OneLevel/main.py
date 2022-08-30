import numpy as np
from CostObjective import FarmEnvCost
from EnvironmentlENV import FarmEnvEnv
from SocialENV import FarmEnvSocial
from SaveResults import SaveOnBestTrainingRewardCallback
import os
from stable_baselines3 import DQN
from parameters import *
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import obs_as_tensor
import torch as th
import math


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]
    return x, y


def calculate_constants(rho1, x_best, x_worst):
    m_x_1 = (math.e ** (-rho1 * x_worst))
    m_x_2 = (math.e ** (-rho1 * x_worst)) - (math.e ** (-rho1 * x_best))
    m_x = float(m_x_1 / m_x_2)
    n_x = float(1 / m_x_2)
    return m_x, n_x


def calculate_utility(rho1, m_x, n_x, q_value):
    u_a = float(m_x - (n_x * (math.e ** (-q_value * rho1))))
    return u_a


def choose_action(obs_cost, cost_model, obs_env, env_model, obs_social, social_model):

    observation_cost = obs_cost.reshape((-1,) + cost_model.observation_space.shape)
    observation_cost = obs_as_tensor(observation_cost, device="cpu")

    observation_env = obs_env.reshape((-1,) + env_model.observation_space.shape)
    observation_env = obs_as_tensor(observation_env, device="cpu")

    observation_social = obs_social.reshape((-1,) + social_model.observation_space.shape)
    observation_social = obs_as_tensor(observation_social, device="cpu")

    with th.no_grad():
        q_value_cost = cost_model.q_net(observation_cost)
        q_value_env = env_model.q_net(observation_env)
        q_value_social = social_model.q_net(observation_social)


    q_value_cost_np = q_value_cost.numpy()
    best_cost = np.max(q_value_cost_np)
    worst_cost = np.min(q_value_cost_np)

    q_value_env_np = q_value_env.numpy()
    best_env = np.max(q_value_env_np)
    worst_env = np.min(q_value_env_np)

    q_value_social_np = q_value_social.numpy()
    best_social = np.max(q_value_social_np)
    worst_social = np.min(q_value_social_np)

    con_cost_m, con_cost_n = calculate_constants(rho, best_cost, worst_cost)
    con_env_m, con_env_n = calculate_constants(rho, best_env, worst_env)
    con_social_m, con_social_n = calculate_constants(rho, best_social, worst_social)

    utilities_cost = np.zeros(q_value_cost_np.shape[1])
    utilities_env = np.zeros(q_value_env_np.shape[1])
    utilities_social = np.zeros(q_value_social_np.shape[1])

    for i in range(0, q_value_cost_np.shape[1]):
        utilities_cost[i] = calculate_utility(rho, con_cost_m, con_cost_n, q_value_cost_np[0, i])
        utilities_env[i] = calculate_utility(rho, con_env_m, con_env_n, q_value_env_np[0, i])
        utilities_social[i] = calculate_utility(rho, con_social_m, con_social_n, q_value_social_np[0, i])

    total_utility = (lambda_cost * utilities_cost) + (lambda_env * utilities_env) + (lambda_social * utilities_social)
    new_action_id = np.argmax(total_utility)
    return new_action_id


if th.cuda.is_available():
    print('GPU is available')
    device = 'cuda'
else:
    print('No GPU')
    device = 'cpu'

# making the three environments
Cost_ENV = FarmEnvCost()
Env_EnV = FarmEnvEnv()
Social_Env = FarmEnvSocial()

log_dir_cost = "tmp/cost/"
log_dir_env = "tmp/env/"
log_dir_social = "tmp/social/"

os.makedirs(log_dir_cost, exist_ok=True)
os.makedirs(log_dir_env, exist_ok=True)
os.makedirs(log_dir_social, exist_ok=True)

env_cost = Monitor(Cost_ENV, log_dir_cost)
env_env = Monitor(Env_EnV, log_dir_env)
env_social = Monitor(Social_Env, log_dir_social)

callback_cost = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir_cost)
callback_env = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir_env)
callback_social = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir_social)


cost_model = DQN('MlpPolicy', env_cost, exploration_fraction=0.8, exploration_final_eps=0.02,
                 exploration_initial_eps=1.0, verbose=1, device=device)
cost_model.learn(total_timesteps=time_steps, callback=callback_cost)

print('############### Finished Training the first objective (Economical) ########################')

env_model = DQN('MlpPolicy', env_env, exploration_fraction=0.8, exploration_final_eps=0.02, exploration_initial_eps=1.0,
               verbose=1, device=device)
env_model.learn(total_timesteps=time_steps, callback=callback_env)
print('################# Finished Training the second objective (Environmental) ######################')


social_model = DQN('MlpPolicy', env_social, exploration_fraction=0.8, exploration_final_eps=0.02,
                  exploration_initial_eps=1.0, verbose=1, device=device)
social_model.learn(total_timesteps=time_steps, callback=callback_social)
print('#################### Finished Training the third objective (Social) ############################')


# Utility function
obs_cost = env_cost.reset()
obs_env = env_env.reset()
obs_social = env_social.reset()
n_steps = PRUNE_LENGTH

observation_cost = obs_cost.reshape((-1,) + cost_model.observation_space.shape)
observation_cost = obs_as_tensor(observation_cost, device="cpu")

observation_env = obs_env.reshape((-1,) + env_model.observation_space.shape)
observation_env = obs_as_tensor(observation_env, device="cpu")

observation_social = obs_social.reshape((-1,) + social_model.observation_space.shape)
observation_social = obs_as_tensor(observation_social, device="cpu")


with th.no_grad():
    q_value_cost = cost_model.q_net(observation_cost)
    q_value_env = env_model.q_net(observation_env)
    q_value_social = social_model.q_net(observation_social)

q_value_cost_np = q_value_cost.numpy()
best_cost = np.max(q_value_cost_np)
worst_cost = np.min(q_value_cost_np)

q_value_env_np = q_value_env.numpy()
best_env = np.max(q_value_env_np)
worst_env = np.min(q_value_env_np)

q_value_social_np = q_value_social.numpy()
best_social = np.max(q_value_social_np)
worst_social = np.min(q_value_social_np)

con_cost_m, con_cost_n = calculate_constants(rho, best_cost, worst_cost)
con_env_m, con_env_n = calculate_constants(rho, best_env, worst_env)
con_social_m, con_social_n = calculate_constants(rho, best_social, worst_social)

utilities_cost = np.zeros(q_value_cost_np.shape[1])
utilities_env = np.zeros(q_value_env_np.shape[1])
utilities_social = np.zeros(q_value_social_np.shape[1])

for i in range(0, q_value_cost_np.shape[1]):
    utilities_cost[i] = calculate_utility(rho, con_cost_m, con_cost_n, q_value_cost_np[0,i])
    utilities_env[i] = calculate_utility(rho, con_env_m, con_env_n, q_value_env_np[0, i])
    utilities_social[i] = calculate_utility(rho, con_social_m, con_social_n, q_value_social_np[0, i])


total_utility = (lambda_cost * utilities_cost) + (lambda_env * utilities_env) + (lambda_social * utilities_social)
print('******************************************************')
new_action_id = np.argmax(total_utility)
print(new_action_id)

if isinstance(new_action_id,np.ndarray):
    new_action_id = new_action_id[0]

for step in range(n_steps):

    worker_availability = round(np.random.normal(WORKER_AVAILABILITY, 1))
    action_id = choose_action(obs_cost, cost_model, obs_env, env_model, obs_social, social_model)
    print("Step {}".format(step + 1))

    mapping = tuple(np.ndindex((MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1)))
    new_action = mapping[action_id]

    action_cost, _ = cost_model.predict(obs_cost, deterministic=True)
    action_cost_unroll = mapping[action_cost]
    action_env, _ = env_model.predict(obs_cost, deterministic=True)
    action_env_unroll = mapping[action_env]
    action_social, _ = social_model.predict(obs_social, deterministic=True)
    action_social_unroll = mapping[action_social]

    obs_cost, reward_cost, done_cost, info = env_cost.step(action_id)
    obs_env, reward_env, done_env, info = env_env.step(action_id)
    obs_social, reward_social, done_social, info = env_social.step(action_id)

    print("Best Overall action= ", new_action)
    print("Best action Based on Economic Objective= ", action_cost_unroll)
    print("Best action Based on Environmental Objective= ", action_env_unroll)
    print("Best action Based on Social Objective= ", action_social_unroll)

    print('Cost State= ', obs_cost, 'reward= ', reward_cost, 'done= ', done_cost)
    print('Environment State= ', obs_env, 'reward= ', reward_env, 'done= ', done_env)
    print('Social State= ', obs_social, 'reward= ', reward_social, 'done= ', done_social)

    if done_cost is True:
        break



# plotting the results
x_cost, y_cost = plot_results(log_dir_cost)
x_env, y_env = plot_results(log_dir_env)
x_social, y_social = plot_results(log_dir_social)

title = 'Learning Curve'
fig = plt.figure(title)
plt.plot(x_cost, y_cost, 'r')
plt.plot(x_env, y_env, 'b')
plt.plot(x_social, y_social, 'g')
plt.xlabel('Number of Timesteps')
plt.ylabel('Rewards')
plt.title(title + " Smoothed")
plt.show()
