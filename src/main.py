import numpy as np
from stable_baselines3 import DQN
from src.env.EconomicObj import EconomicEnv
from src.env.EnvironmentObj import EnvironmentEnv
from src.env.SocialObj import SocialEnv
from Parameters import *
from stable_baselines3.common.utils import obs_as_tensor
from saveFunctions import *
import math
import torch as th
import warnings
warnings.filterwarnings('ignore')


'''
    Before running this code make sure three environments are trained and saved in model folder. 
'''

def calculate_constants(rho1, x_best, x_worst):
    if rho1 == 0:
        m_x = 0
        n_x = 0
        return m_x, n_x
    m_x_1 = float(math.exp((-rho1 * x_worst)))
    m_x_2 = float(math.exp((-rho1 * x_worst))) - (math.exp((-rho1 * x_best)))
    m_x = float(m_x_1 / m_x_2)
    n_x = float(1 / m_x_2)
    return m_x, n_x

def calculate_utility(rho1, m_x, n_x, q_value):
    if rho1 == 0:
        u_a = q_value
        return u_a
    u_a = float(m_x - (n_x * (math.exp((-q_value * rho1)))))
    return u_a

def choose_action(obs_econ, econ_model, obs_env, env_model, obs_social, social_model,device):

    observation_econ = obs_econ.reshape((-1,) + econ_model.observation_space.shape)
    observation_econ = obs_as_tensor(observation_econ, device=device)

    observation_env = obs_env.reshape((-1,) + env_model.observation_space.shape)
    observation_env = obs_as_tensor(observation_env, device=device)

    observation_social = obs_social.reshape((-1,) + social_model.observation_space.shape)
    observation_social = obs_as_tensor(observation_social, device=device)

    with th.no_grad():
        q_value_econ = econ_model.q_net(observation_econ)
        q_value_env = env_model.q_net(observation_env)
        q_value_social = social_model.q_net(observation_social)

    q_value_econ_np = q_value_econ.numpy()
    best_econ = np.max(q_value_econ_np)
    worst_econ = np.min(q_value_econ_np)

    q_value_econ_np = (q_value_econ - worst_econ) / (best_econ - worst_econ)
    best_econ = np.max(q_value_econ_np.numpy())
    worst_econ = np.min(q_value_econ_np.numpy())

    q_value_env_np = q_value_env.numpy()
    best_env = np.max(q_value_env_np)
    worst_env = np.min(q_value_env_np)
    q_value_env_np = (q_value_env_np - worst_env) / (best_env - worst_env)
    best_env = np.max(q_value_env_np)
    worst_env = np.min(q_value_env_np)

    q_value_social_np = q_value_social.numpy()
    best_social = np.max(q_value_social_np)
    worst_social = np.min(q_value_social_np)
    q_value_social_np = (q_value_social_np - worst_social) / (best_social - worst_social)
    best_social = np.max(q_value_social_np)
    worst_social = np.min(q_value_social_np)

    con_econ_m, con_econ_n = calculate_constants(rho, best_econ, worst_econ)
    con_env_m, con_env_n = calculate_constants(rho, best_env, worst_env)
    con_social_m, con_social_n = calculate_constants(rho, best_social, worst_social)

    utilities_econ = np.zeros(q_value_econ_np.shape[1])
    utilities_env = np.zeros(q_value_env_np.shape[1])
    utilities_social = np.zeros(q_value_social_np.shape[1])

    for i in range(0, q_value_econ_np.shape[1]):
        utilities_econ[i] = calculate_utility(rho, con_econ_m, con_econ_n, q_value_econ_np[0, i])
        utilities_env[i] = calculate_utility(rho, con_env_m, con_env_n, q_value_env_np[0, i])
        utilities_social[i] = calculate_utility(rho, con_social_m, con_social_n, q_value_social_np[0, i])

    total_utility = (lambda_econ * utilities_econ) + (lambda_env * utilities_env) + (lambda_social * utilities_social)
    new_action_id = np.argmax(total_utility)

    return new_action_id


def main():

    if th.cuda.is_available():
        print('GPU is available')
        device = 'cuda'
    else:
        print('No GPU')
        device = 'cpu'

    # loading trained models
    econ_model = DQN.load("../model/economic_model")
    env_model = DQN.load("../model/environmental_model")
    social_model = DQN.load("../model/social_model")

    econ_env = EconomicEnv()
    env_env = EnvironmentEnv()
    social_env = SocialEnv()

    obs_econ = econ_env.reset()
    obs_env = env_env.reset()
    obs_social = social_env.reset()
    n_steps = PRUNE_LENGTH

    print('******************************************************')

    track_reward_econ = []
    track_reward_env = []
    track_reward_social = []

    for step in range(n_steps):

        action_id = choose_action(obs_econ, econ_model, obs_env, env_model, obs_social, social_model, device)
        print("Step {}".format(step + 1))

        mapping = tuple \
            (np.ndindex(((2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1)))
        new_action = mapping[action_id]
        print(new_action)

        action_econ, _ = econ_model.predict(obs_econ, deterministic=True)
        action_econ_unroll = mapping[action_econ]
        print("Stable baseline prediction econ:", action_econ_unroll)
        action_env, _ = env_model.predict(obs_econ, deterministic=True)
        action_env_unroll = mapping[action_env]
        print("Stable baseline prediction env:", action_env_unroll)
        action_social, _ = social_model.predict(obs_social, deterministic=True)
        action_social_unroll = mapping[action_social]
        print("Stable baseline prediction social:", action_social_unroll)

        np.random.seed(step + 110)
        obs_econ, reward_econ, done_econ, info = econ_env.step(action_id)
        track_reward_econ = np.append(track_reward_econ, reward_econ)
        np.random.seed(step + 110)
        obs_env, reward_env, done_env, info = env_env.step(action_id)
        track_reward_env = np.append(track_reward_env, reward_env)
        np.random.seed(step + 110)
        obs_social, reward_social, done_social, info = social_env.step(action_id)
        track_reward_social = np.append(track_reward_social, reward_social)

        print('State= ', obs_econ)
        print('Economic Reward= ', reward_econ)
        print('Environmental Reward= ', reward_env)
        print('Social Reward= ', reward_social)
        print('******************************************************')

        if done_econ is True:
            remining_b1 = round(social_env.budget, 3)
            print('Remaining Budget = ', str(remining_b1) + '/' + str(BUDGET), 'Un-pruned Plants = ',
                  str(social_env.plants) + '/' + str(PLANTS))
            break

    print('done')

if __name__ == '__main__':
    main()

