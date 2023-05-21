import numpy as np
from src.env.EconomicObj import EconomicEnv
from src.env.EnvironmentObj import EnvironmentEnv
from src.env.SocialObj import SocialEnv
from Parameters import *
from saveFunctions import *
from tqdm import tqdm
import math

import warnings

warnings.filterwarnings('ignore')


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


def best_action(econ_env, env_env, social_env, state, remain_plants, remain_budget):
    # case 1: rho = -3, Econ: 0.5, Env: 0.25, Social: 0.25

    reward_econ = np.zeros([econ_env.action_space.n, 1])
    reward_env = np.zeros([env_env.action_space.n, 1])
    reward_social = np.zeros([social_env.action_space.n, 1])

    # getting all the rewards from Econ:
    current_state = state

    econ_env.budget = remain_budget
    econ_env.plants = remain_plants

    # for action in tqdm(range(econ_env.action_space.n)):
    for action in tqdm(range(100)):
        econ_env.state = current_state
        _, reward_econ[action], _, _ = econ_env.step(action)

    # getting all the rewards from Env:
    env_env.budget = remain_budget
    env_env.plants = remain_plants

    # for action in tqdm(range(env_env.action_space.n)):
    for action in tqdm(range(100)):
        env_env.state = current_state
        _, reward_env[action], _, _ = env_env.step(action)

    # getting all the rewards from Social:
    social_env.budget = remain_budget
    social_env.plants = remain_plants

    # for action in tqdm(range(social_env.action_space.n)):
    for action in tqdm(range(100)):
        social_env.state = current_state
        _, reward_social[action], _, _ = social_env.step(action)

    best_econ = np.max(reward_econ)
    worst_econ = np.min(reward_econ)

    reward_econ_norm = (reward_econ - worst_econ) / (best_econ - worst_econ)
    best_econ = np.max(reward_econ_norm)
    worst_econ = np.min(reward_econ_norm)

    best_env = np.max(reward_env)
    worst_env = np.min(reward_env)
    reward_env_norm = (reward_env - worst_env) / (best_env - worst_env)
    best_env = np.max(reward_env_norm)
    worst_env = np.min(reward_env_norm)

    best_social = np.max(reward_social)
    worst_social = np.min(reward_social)
    reward_social_norm = (reward_social - worst_social) / (best_social - worst_social)
    best_social = np.max(reward_social_norm)
    worst_social = np.min(reward_social_norm)

    # corrected
    rho = -3
    con_econ_m, con_econ_n = calculate_constants(rho, best_econ, worst_econ)
    con_env_m, con_env_n = calculate_constants(rho, best_env, worst_env)
    con_social_m, con_social_n = calculate_constants(rho, best_social, worst_social)

    utilities_econ = np.zeros(reward_econ_norm.shape[0])
    utilities_env = np.zeros(reward_env_norm.shape[0])
    utilities_social = np.zeros(reward_social_norm.shape[0])

    for i in range(0, utilities_econ.shape[0]):
        utilities_econ[i] = calculate_utility(rho, con_econ_m, con_econ_n, reward_econ_norm[i])
        utilities_env[i] = calculate_utility(rho, con_env_m, con_env_n, reward_env_norm[i])
        utilities_social[i] = calculate_utility(rho, con_social_m, con_social_n, reward_social_norm[i])

    total_utility = (lambda_econ * utilities_econ) + (lambda_env * utilities_env) + (lambda_social * utilities_social)
    new_action_id = np.argmax(total_utility)

    return new_action_id


def main():
    econ_env = EconomicEnv()
    env_env = EnvironmentEnv()
    social_env = SocialEnv()

    obs_econ = econ_env.reset()
    obs_env = env_env.reset()
    obs_social = social_env.reset()
    n_steps = PRUNE_LENGTH

    print('******************************************************')
    state = [0, 0, 0]
    remain_plants = PLANTS
    remain_budget = BUDGET

    availability_beg = round(np.random.normal(WORKER_AVAILABILITY_BEG, 3))
    availability_int = round(np.random.normal(WORKER_AVAILABILITY_INT, 2))
    availability_adv = round(np.random.normal(WORKER_AVAILABILITY_ADV, 1))
    productivity_beg = np.random.normal(PRODUCTIVITY_BEG, 0.03448)
    productivity_int = np.random.normal(PRODUCTIVITY_INT, 0.0258)
    productivity_adv = np.random.normal(PRODUCTIVITY_ADV, 0.01724)
    econ_env.availability_beg, env_env.availability_beg, social_env.availability_beg = availability_beg, availability_beg, availability_beg
    econ_env.availability_int, env_env.availability_int, social_env.availability_int = availability_int, availability_int, availability_int
    econ_env.availability_adv, env_env.availability_adv, social_env.availability_adv = availability_adv, availability_adv, availability_adv
    econ_env.productivity_beg, env_env.productivity_beg, social_env.productivity_beg = productivity_beg, productivity_beg, productivity_beg
    econ_env.productivity_int, env_env.productivity_int, social_env.productivity_int = productivity_int, productivity_int, productivity_int
    econ_env.productivity_adv, env_env.productivity_adv, social_env.productivity_adv = productivity_adv, productivity_adv, productivity_adv

    for step in range(n_steps):
        action_id = best_action(econ_env, env_env, social_env, state, remain_plants, remain_budget)
        print("Step {}".format(step + 1))
        obs, reward, done, _ = econ_env.step(action_id)
        remain_plants = econ_env.plants
        remain_budget = econ_env.budget
        if done:
            print("Done after {} steps".format(step + 1))
            break


if __name__ == '__main__':
    main()
