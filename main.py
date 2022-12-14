import numpy as np
from stable_baselines3 import DQN
from newEconomic import EconomicEnvNew
from EnvironmentNew import EnvironmentEnv
from SocialNew import SocialEnv
from Parameters import *
from stable_baselines3.common.utils import obs_as_tensor
from saveFunctions import *
import math


'''
    Before running this code make sure mainCost, mainEnvironment, and mainSocial has been executed and models are 
    trained.
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

def choose_action(obs_econ, econ_model, obs_env, env_model, obs_social, social_model):

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
    q_value_econ_np = q_value_econ_np.numpy()
    new_action_id = np.argmax(total_utility)
    return new_action_id


def main():
    # checking for cuda availability
    if th.cuda.is_available():
        print('GPU is available')
        device = 'cuda'
    else:
        print('No GPU')
        device = 'cpu'

    # plotting the results of training three objectives:
    log_dir_econ = "tmp/Economic/"
    log_dir_env = "tmp/Environmental/"
    log_dir_social = "tmp/Social/"
    x_econ, y_econ = plot_results_test(log_dir_econ)
    x_env, y_env = plot_results_test(log_dir_env)
    x_social, y_social = plot_results_test(log_dir_social)

    title = 'Learning Curve'
    fig = plt.figure(title)
    plt.plot(x_econ, y_econ, 'r')
    plt.plot(x_env, y_env, 'b')
    plt.plot(x_social, y_social, 'g')
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.legend(["Econ","Env","Soc"])
    plt.show()

    # Utility function
    econ_model = DQN.load("model/economic_model")
    env_model = DQN.load("model/environmental_model")
    social_model = DQN.load("model/social_model")

    Econ_ENV = EconomicEnvNew()
    Env_EnV = EnvironmentEnv()
    Social_Env = SocialEnv()

    obs_econ = Econ_ENV.reset()
    obs_env = Env_EnV.reset()
    obs_social = Social_Env.reset()
    n_steps = PRUNE_LENGTH
    print('******************************************************')
    track_reward_econ = []
    track_reward_env = []
    track_reward_socail = []

    for step in range(n_steps):

        action_id = choose_action(obs_econ, econ_model, obs_env, env_model, obs_social, social_model)
        print("Step {}".format(step + 1))

        mapping = tuple \
            (np.ndindex(((2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1)))
        new_action = mapping[action_id]
        print(new_action)

        action_econ, _ = econ_model.predict(obs_econ, deterministic=False)
        action_econ_unroll = mapping[action_econ]
        action_env, _ = env_model.predict(obs_econ, deterministic=False)
        action_env_unroll = mapping[action_env]
        action_social, _ = social_model.predict(obs_social, deterministic=False)
        action_social_unroll = mapping[action_social]

        np.random.seed(step + 110)
        obs_econ, reward_econ, done_econ, info = Econ_ENV.step(action_id)
        track_reward_econ = np.append(track_reward_econ, reward_econ)
        # print('econ')
        # print(obs_econ)
        np.random.seed(step + 110)
        obs_env, reward_env, done_env, info = Env_EnV.step(action_id)
        track_reward_env = np.append(track_reward_env, reward_env)
        # print('env')
        # print(obs_env)
        np.random.seed(step + 110)
        obs_social, reward_social, done_social, info = Social_Env.step(action_id)
        track_reward_socail = np.append(track_reward_socail, reward_social)
        # print('social')
        # print(obs_social)

        # print("Best Overall action= ", new_action)
        # print("Best action Based on Economic Objective= ", action_econ_unroll)
        # print("Best action Based on Environmental Objective= ", action_env_unroll)
        # print("Best action Based on Social Objective= ", action_social_unroll)

        print('Cost State= ', obs_econ, 'reward= ', reward_econ, 'done= ', done_econ)
        print('Environment State= ', obs_env, 'reward= ', reward_env, 'done= ', done_env)
        print('Social State= ', obs_social, 'reward= ', reward_social, 'done= ', done_social)

        if done_econ is True:
            remining_b1 = round(Social_Env.budget, 3)
            print('Remaining Budget = ', str(remining_b1) + '/' + str(BUDGET), 'Unpruned Plants = ',
                  str(Social_Env.plants) + '/' + str(PLANTS))
            break

    # print(track_reward)
    print('done')

if __name__ == '__main__':
    main()

