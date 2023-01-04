from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
import torch as th
from Parameters import *
import math
import time

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

def choose_action(obs_econ, econ_model, obs_env, env_model, obs_social, social_model, device, k):

    st = time.time()

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

    ################################ test
    q_value_econ_np = (q_value_econ - worst_econ) / (best_econ - worst_econ)
    best_econ = np.max(q_value_econ_np.numpy())
    worst_econ = np.min(q_value_econ_np.numpy())

    q_value_env_np = q_value_env.numpy()
    best_env = np.max(q_value_env_np)
    worst_env = np.min(q_value_env_np)
    ################################ test
    q_value_env_np = (q_value_env_np - worst_env) / (best_env - worst_env)
    best_env = np.max(q_value_env_np)
    worst_env = np.min(q_value_env_np)
    #print(q_value_env_np)

    q_value_social_np = q_value_social.numpy()
    best_social = np.max(q_value_social_np)
    worst_social = np.min(q_value_social_np)
    ################################ test
    q_value_social_np = (q_value_social_np - worst_social) / (best_social - worst_social)
    best_social = np.max(q_value_social_np)
    worst_social = np.min(q_value_social_np)
    #print(q_value_social_np)

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
    sorted_id = np.argsort(-total_utility)
    sorted_value = -np.sort(-total_utility)

    new_actions_id = sorted_id[0:k]
    new_actions_value = sorted_value[0:k]

    et = time.time()

    print('MAUT time', (et-st))
    # return [new_actions_id, new_actions_value]
    return [sorted_id, sorted_value]
