import math
from Parameters import *
from Social import SocialEnv
from Economic import EconomicEnv
from Enviroment import EnvironmentEnv
import numpy as np
from tqdm import tqdm
import time
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


def normalize(reward):
    mini = 1000
    maxi = 0
    for i in range(len(reward)):
        if reward[i] == -10:
            reward[i] = np.nan
        else:
            if reward[i] < mini:
                mini = reward[i]
            if reward[i] > maxi:
                maxi = reward[i]

    for i in range(len(reward)):
        if reward[i] != np.nan:
            reward[i] = (reward[i] - mini) / (maxi - mini)

    return reward, mini, maxi


def calculate_eliminate_utility(con_m, con_n, reward_norm):
    utilities = np.zeros(reward_norm.shape[0])
    for i in range(len(reward_norm)):
        if reward_norm[i] != np.nan:
            utilities[i] = calculate_utility(rho, con_m, con_n, reward_norm[i])
        else:
            utilities[i] = np.nan
    return utilities


def find_max_utility(utilities):
    maximum = 0
    maximum_id = -1
    for i in range(len(utilities)):
        if utilities[i] == np.nan:
            pass
        elif utilities[i] > maximum:
            maximum = utilities[i]
            maximum_id = i
    return maximum_id


def best_action(state, remain_plants, remain_budget, avail_beg, avail_int, avail_adv, prod_beg, prod_int, prod_adv):
    econ_env = EconomicEnv()
    env_env = EnvironmentEnv()
    social_env = SocialEnv()

    reward_econ = np.zeros([econ_env.action_space.n, 1])
    reward_env = np.zeros([env_env.action_space.n, 1])
    reward_social = np.zeros([social_env.action_space.n, 1])

    # getting all the rewards from Econ:
    print('Rho:', rho, 'ECON:', lambda_econ, 'ENV:', lambda_env, 'Soc:', lambda_social)
    for action in tqdm(range(econ_env.action_space.n)):
    #for action in tqdm(range(100)):
        econ_env.state = state
        econ_env.budget = remain_budget
        econ_env.plants = remain_plants
        econ_env.availability_beg = avail_beg
        econ_env.availability_int = avail_int
        econ_env.availability_adv = avail_adv
        econ_env.productivity_beg = prod_beg
        econ_env.productivity_int = prod_int
        econ_env.productivity_adv = prod_adv
        _, reward_econ[action], _, _ = econ_env.step(action)

    # getting all the rewards from Env
    for action in tqdm(range(env_env.action_space.n)):
    #for action in tqdm(range(100)):
        env_env.state = state
        env_env.budget = remain_budget
        env_env.plants = remain_plants
        env_env.availability_beg = avail_beg
        env_env.availability_int = avail_int
        env_env.availability_adv = avail_adv
        env_env.productivity_beg = prod_beg
        env_env.productivity_int = prod_int
        env_env.productivity_adv = prod_adv
        _, reward_env[action], _, _ = env_env.step(action)

    # getting all the rewards from Social:
    for action in tqdm(range(social_env.action_space.n)):
        # for action in tqdm(range(100)):
        social_env.state = state
        social_env.budget = remain_budget
        social_env.plants = remain_plants

        social_env.availability_beg = avail_beg
        social_env.availability_int = avail_int
        social_env.availability_adv = avail_adv

        social_env.productivity_beg = prod_beg
        social_env.productivity_int = prod_int
        social_env.productivity_adv = prod_adv
        _, reward_social[action], _, _ = social_env.step(action)

    # best_econ = np.max(reward_econ)
    # worst_econ = np.min(reward_econ)
    #
    # reward_econ_norm = (reward_econ - worst_econ) / (best_econ - worst_econ)
    # best_econ = np.max(reward_econ_norm)
    # worst_econ = np.min(reward_econ_norm)
    #
    # best_env = np.max(reward_env)
    # worst_env = np.min(reward_env)
    # reward_env_norm = (reward_env - worst_env) / (best_env - worst_env)
    # best_env = np.max(reward_env_norm)
    # worst_env = np.min(reward_env_norm)
    #
    # best_social = np.max(reward_social)
    # worst_social = np.min(reward_social)
    # reward_social_norm = (reward_social - worst_social) / (best_social - worst_social)
    # best_social = np.max(reward_social_norm)
    # worst_social = np.min(reward_social_norm)

    #################### Added remove this later
    reward_econ_norm, worst_econ, best_econ = normalize(reward_econ)
    reward_env_norm, worst_env, best_env = normalize(reward_env)
    reward_social_norm, worst_social, best_social = normalize(reward_social)
    #######################

    con_econ_m, con_econ_n = calculate_constants(rho, best_econ, worst_econ)
    con_env_m, con_env_n = calculate_constants(rho, best_env, worst_env)
    con_social_m, con_social_n = calculate_constants(rho, best_social, worst_social)

    utilities_econ = np.zeros(reward_econ_norm.shape[0])
    utilities_env = np.zeros(reward_env_norm.shape[0])
    utilities_social = np.zeros(reward_social_norm.shape[0])

    ################# Added remove this later
    utilities_econ = calculate_eliminate_utility(con_econ_m, con_econ_n, reward_econ_norm)
    utilities_env = calculate_eliminate_utility(con_env_m, con_env_n, reward_env_norm)
    utilities_social = calculate_eliminate_utility(con_social_m, con_social_n, reward_social_norm)
    #################

    # for i in range(0, utilities_econ.shape[0]):
    #     utilities_econ[i] = calculate_utility(rho, con_econ_m, con_econ_n, reward_econ_norm[i])
    #     utilities_env[i] = calculate_utility(rho, con_env_m, con_env_n, reward_env_norm[i])
    #     utilities_social[i] = calculate_utility(rho, con_social_m, con_social_n, reward_social_norm[i])


    total_utility = (lambda_econ * utilities_econ) + (lambda_env * utilities_env) + (lambda_social * utilities_social)


    #new_action_id = np.argmax(total_utility)

    ################# Added remove this later
    new_action_id = find_max_utility(total_utility)
    #################

    return new_action_id


def main():
    econ_env = EconomicEnv()
    env_env = EnvironmentEnv()
    social_env = SocialEnv()
    n_steps = PRUNE_LENGTH

    print('******************************************************')
    state = [0, 0, 0]
    remain_plants = PLANTS
    remain_budget = BUDGET

    total_econ = 0
    total_env = 0
    total_social = 0

    availability_beg = 20
    availability_int = 15
    availability_adv = 12
    productivity_beg = 0.24 * 1450
    productivity_int = 0.38 * 1450
    productivity_adv = 0.52 * 1450
    econ_env.availability_beg, env_env.availability_beg, social_env.availability_beg = availability_beg, availability_beg, availability_beg
    econ_env.availability_int, env_env.availability_int, social_env.availability_int = availability_int, availability_int, availability_int
    econ_env.availability_adv, env_env.availability_adv, social_env.availability_adv = availability_adv, availability_adv, availability_adv
    econ_env.productivity_beg, env_env.productivity_beg, social_env.productivity_beg = productivity_beg, productivity_beg, productivity_beg
    econ_env.productivity_int, env_env.productivity_int, social_env.productivity_int = productivity_int, productivity_int, productivity_int
    econ_env.productivity_adv, env_env.productivity_adv, social_env.productivity_adv = productivity_adv, productivity_adv, productivity_adv

    for step in range(n_steps):
        action_id = best_action(state, remain_plants, remain_budget, availability_beg, availability_int,
                                availability_adv,
                                productivity_beg, productivity_int, productivity_adv)
        mapping = tuple \
            (np.ndindex(((2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1)))
        print("Step {}".format(step + 1))
        print('Selected Action:', mapping[action_id])
        obs, reward_econ, done, _ = econ_env.step(action_id)
        _, reward_env, _, _ = env_env.step(action_id)
        _, reward_social, _, _ = social_env.step(action_id)
        remain_plants = econ_env.plants
        remain_budget = econ_env.budget

        if econ_env.plants != env_env.plants or econ_env.budget != env_env.budget:
            raise ValueError('Plants or Budget not equal')

        print('remaining budget:', remain_budget)
        print('remaining plants:', remain_plants)
        state = obs
        total_econ += reward_econ
        total_env += reward_env
        total_social += reward_social
        if done:
            print("Done after {} steps".format(step + 1))
            print('Total Economic Reward:', total_econ)
            print('Total Environmental Reward:', total_env)
            print('Total Social Reward:', total_social)
            break


if __name__ == '__main__':
    start_time = time.time()
    print('(1/9)Economic, Rho = -3')
    #Used for the Utility function
    rho = -3
    lambda_econ, lambda_env, lambda_social = 0.5, 0.25, 0.25
    main()
    end_time = time.time()
    print("running: ", end_time - start_time)
    print('********************************')

    start_time = time.time()
    print('(2/9)Economic, Rho = 0')
    #Used for the Utility function
    rho = 0
    lambda_econ, lambda_env, lambda_social = 0.5, 0.25, 0.25
    main()
    end_time = time.time()
    print("running: ", end_time - start_time)
    print('********************************')

    start_time = time.time()
    print('(3/9)Economic, Rho = 3')
    #Used for the Utility function
    rho = 3
    lambda_econ, lambda_env, lambda_social = 0.5, 0.25, 0.25
    main()
    end_time = time.time()
    print("running: ", end_time - start_time)
    print('********************************')

    start_time = time.time()
    print('(4/9)Env, Rho = -3')
    #Used for the Utility function
    rho = -3
    lambda_econ, lambda_env, lambda_social = 0.25, 0.5, 0.25
    main()
    end_time = time.time()
    print("running: ", end_time - start_time)
    print('********************************')

    start_time = time.time()
    print('(5/9)Env, Rho = 0')
    # Used for the Utility function
    rho = 0
    lambda_econ, lambda_env, lambda_social = 0.25, 0.5, 0.25
    main()
    end_time = time.time()
    print("running: ", end_time - start_time)
    print('********************************')

    start_time = time.time()
    print('(6/9)Env, Rho = 3')
    # Used for the Utility function
    rho = 3
    lambda_econ, lambda_env, lambda_social = 0.25, 0.5, 0.25
    main()
    end_time = time.time()
    print("running: ", end_time - start_time)
    print('********************************')

    start_time = time.time()
    print('(7/9)social, Rho = -3')
    # Used for the Utility function
    rho = -3
    lambda_econ, lambda_env, lambda_social = 0.25, 0.25, 0.5
    main()
    end_time = time.time()
    print("running: ", end_time - start_time)
    print('********************************')

    start_time = time.time()
    print('(8/9)social, Rho = 0')
    # Used for the Utility function
    rho = 0
    lambda_econ, lambda_env, lambda_social = 0.25, 0.25, 0.5
    main()
    end_time = time.time()
    print("running: ", end_time - start_time)
    print('********************************')

    start_time = time.time()
    print('(9/9)Social, Rho = 3')
    # Used for the Utility function
    rho = 3
    lambda_econ, lambda_env, lambda_social = 0.25, 0.25, 0.5
    main()
    end_time = time.time()
    print("running: ", end_time - start_time)
    print('********************************')