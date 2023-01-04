import numpy as np
from stable_baselines3 import DQN
from newEconomic import EconomicEnvNew
from EnvironmentNew import EnvironmentEnv
from SocialNew import SocialEnv
from Parameters import *
from stable_baselines3.common.utils import obs_as_tensor
from saveFunctions import *
import math
from action import choose_action
from treelib import Tree, Node
import time


def get_action_path(beam_tree, node_id):
    # getting the path
    x = beam_tree.get_node(node_id)
    list_action = [x.tag[0]]
    n_id = node_id

    while True:
        A = beam_tree.parent(int(n_id))
        B = A.tag
        if B == 0:
            break
        list_action = np.append(list_action, B[0])
        if A.identifier == 0:
            break
        else:
            n_id = A.identifier
    return list_action

def get_parent(new_tree, nd):
    A = new_tree.parent(int(nd))
    B = A.identifier
    return B


def get_current_state(action_path):
    # getting new status of the enviornment
    if len(action_path) == 0:
        tmp_env_econ = EconomicEnvNew()
        tmp_env_env = EnvironmentEnv()
        tmp_env_social = SocialEnv()
        obs_new = [0,0,0]
        done_new = False
        return tmp_env_econ,tmp_env_env,tmp_env_social, obs_new, done_new

    tmp_env_econ = EconomicEnvNew()
    tmp_env_env = EnvironmentEnv()
    tmp_env_social = SocialEnv()
    obs_new = tmp_env_econ.reset()
    for a in action_path:
        obs_new, reward_new, done_new, info = tmp_env_econ.step(a)

    return tmp_env_econ,tmp_env_env,tmp_env_social, obs_new, done_new


def check_eligibility(A, action_path):
    # check for action eligibility
    flag = True
    tmp_env, _, _, _, _ = get_current_state(action_path)
    obs_new, reward_new, done_new, info = tmp_env.step(A)
    mapping = tuple(
        np.ndindex(((2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1)))
    tmp_action = mapping[A]
    tmp_action = list(tmp_action)

    if reward_new <= -M:
        flag = False
    if tmp_env.plants < 0:
        flag = False

    if tmp_env.budget < 0:
        flag = False

    for i in range(3):
        if tmp_action[i] > MAX_ALLOWED_WORKER:
            tmp_action[i] = MAX_ALLOWED_WORKER - tmp_action[i]

    # add hire and fire
    if (tmp_action[0] +tmp_action[1] + tmp_action[2]) > MAX_ALLOWED_WORKER:
        flag = False


    if (tmp_action[0] + tmp_action[1] + tmp_action[2]) < 0:
        flag = False

    return flag

def expnad_nodes(new_c, k, new_tree,found_path,num_found, econ_model,env_model,social_model, c, parent_id):

    total_values = []
    total_indx = []
    track_identifier = []

    ll = 0
    #
    for i in range(k):
        action_list = get_action_path(new_tree, new_c[ll])
        if len(action_list) == 2:
            action_list = np.flip(action_list)
        env_econ, enc_env, env_social, obs, done = get_current_state(action_list)

        # if done == True:
        #     found_path.append([action_list])
        #     num_found += 1
        #     k = k - 1
        indx, valu = choose_action(obs, econ_model, obs, env_model, obs, social_model, device, k)


        added = 0
        start_id = 0
        while added < k:
            current_path = get_action_path(new_tree, new_c[ll])
            current_path = np.flip(current_path)
            chosen_action = indx[start_id]
            current_path = np.append(current_path, chosen_action)
            result = check_eligibility(chosen_action, current_path)
            if result:
                added += 1
                node_name = [indx[start_id], valu[start_id]]
                new_tree.create_node(node_name, c, parent=parent_id[i])
                total_values = np.append(total_values, valu[start_id])
                total_indx = np.append(total_indx, indx[start_id])
                track_identifier = np.append(track_identifier, c)
                c += 1
            start_id += 1
        ll += 1

        new_c = track_identifier

    return new_tree, new_c, c, total_values, total_indx, track_identifier

def remove_nodes(total_indx, total_values, track_identifier, k, new_tree):

    sorted_indx = total_indx[np.argsort(-total_values)]
    sorted_indx = sorted_indx.astype('int64')
    sorted_track = track_identifier[np.argsort(-total_values)]
    rm_identifiers = sorted_track[k:]
    remove_indxs = sorted_indx[k:]
    for j in range(remove_indxs.shape[0]):
        new_tree.remove_node(rm_identifiers[j])
    return new_tree, rm_identifiers

def test_chosen_path(Econ_ENV, Env_EnV, Social_Env, in_path):

    reward_econs = []
    reward_envs = []
    reward_socails= []
    for i in range(len(in_path)):

        a_i = in_path[i]
        np.random.seed(i + 110)
        obs_env, reward_env, done_env, info = Env_EnV.step(a_i)
        reward_envs = np.append(reward_envs, reward_env)

        np.random.seed(i + 110)
        obs_econ, reward_econ, done_econ, info = Econ_ENV.step(a_i)
        reward_econs = np.append(reward_econs,reward_econ)
        # print('econ')
        # print(obs_econ)
        # np.random.seed(i + 110)
        # obs_env, reward_env, done_env, info = Env_EnV.step(a_i)
        # reward_envs = np.append(reward_envs, reward_env)
        # print('env')
        # print(obs_env)
        np.random.seed(i + 110)
        obs_social, reward_social, done_social, info = Social_Env.step(a_i)
        reward_socails = np.append(reward_socails, reward_social)
        # print('social action:', )
        # print(obs_social)

        print('Current State= ', obs_econ)
        print('Action=', a_i)
        # print('Environment State= ', obs_env)
        # print('Social State= ', obs_social)
        if done_econ is True:
            remining_b1 = round(Social_Env.budget, 3)
            print('Remaining Budget = ', str(remining_b1) + '/' + str(BUDGET), 'Unpruned Plants = ',
                  str(Social_Env.plants) + '/' + str(PLANTS))
            break
    print('g')
    # return reward_econs, reward_envs, reward_socails


def main():
    found_path = []
    num_found = 0
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

    k = 10
    # first time step - creating first k
    action_ids, action_values = choose_action(obs_econ, econ_model, obs_env, env_model, obs_social, social_model, device, k)

    # creating the tree to keep track of paths
    st = time.time()
    new_tree = Tree()
    new_tree.create_node(0, 1)
    last_added_identifier = 1

    c = 2

    added = 0
    start_id = 0
    while added < k:
        current_path = []
        chosen_action = action_ids[start_id]
        result = check_eligibility(chosen_action, current_path)
        if result:
            added += 1
            node_name = [action_ids[start_id], action_values[start_id]]
            new_tree.create_node(node_name, c, parent=1)
            last_added_identifier += 1
            c += 1
        start_id += 1

    new_tree.show()

    start_point = 1
    parent_id = np.zeros(k)

    for i in range(k):
        parent_id[i] = start_point + 1
        start_point += 1


    # expanding the nodes
    new_c = np.zeros(k)

    for ll in range(k):
        new_c[ll] = 2 + ll

    for t in range(PRUNE_LENGTH):

        new_tree, new_c, c, total_values, total_indx, track_identifier = expnad_nodes(new_c, k, new_tree, found_path, num_found, econ_model, env_model, social_model, c, parent_id)
        new_tree.show()

        parent_id_tracker = []

        for o in range(k ** 2):
            parent_id_tracker = np.append(parent_id_tracker, get_parent(new_tree, new_c[o]))

        new_tree, rm_identifiers = remove_nodes(total_indx, total_values, track_identifier, k, new_tree)


        tmp_new_c = []
        tmp_parents = []
        for tt in range(len(new_c)):
            if new_c[tt] in rm_identifiers:
                continue
            else:
                tmp_new_c = np.append(tmp_new_c, new_c[tt])
                tmp_parents = np.append(tmp_parents, parent_id_tracker[tt])

        new_c = tmp_new_c
        parent_id = tmp_new_c


        new_tree.show()
    et = time.time()
    print('beam search time', et-st)
    for h in range(k):
        current_path = get_action_path(new_tree, new_c[h])

        current_path = np.flip(current_path)
        Econ_ENV = EconomicEnvNew()
        Env_EnV = EnvironmentEnv()
        Social_Env = SocialEnv()
        test_chosen_path(Econ_ENV,Env_EnV,Social_Env, current_path)
        print('***************************************')





if __name__ == '__main__':
    main()
