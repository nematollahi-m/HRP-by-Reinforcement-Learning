from stable_baselines3 import DQN
from src.env.EconomicObj import EconomicEnv
from src.env.EnvironmentObj import EnvironmentEnv
from src.env.SocialObj import SocialEnv
from Parameters import *
from saveFunctions import *
from action import choose_action
from treelib import Tree
import time
import torch as th
import warnings

warnings.filterwarnings('ignore')


def get_action_path(beam_tree, node_id):
    # getting the path
    x = beam_tree.get_node(node_id)
    list_action = [x.tag[0]]
    n_id = node_id

    while True:
        a = beam_tree.parent(int(n_id))
        b = a.tag
        if b == 0:
            break
        list_action = np.append(list_action, b[0])
        if a.identifier == 0:
            break
        else:
            n_id = a.identifier
    return list_action


def get_parent(new_tree, nd):
    a = new_tree.parent(int(nd))
    b = a.identifier
    return b


def get_current_state(action_path):
    # getting new status of the environment
    if len(action_path) == 0:
        tmp_env_econ = EconomicEnv()
        tmp_env_env = EnvironmentEnv()
        tmp_env_social = SocialEnv()
        obs_new = [0, 0, 0]
        done_new = False
        return tmp_env_econ, tmp_env_env, tmp_env_social, obs_new, done_new

    tmp_env_econ = EconomicEnv()
    tmp_env_env = EnvironmentEnv()
    tmp_env_social = SocialEnv()
    obs_new = tmp_env_econ.reset()
    for a in action_path:
        obs_new, reward_new, done_new, info = tmp_env_econ.step(a)

    return tmp_env_econ, tmp_env_env, tmp_env_social, obs_new, done_new


def check_eligibility(a, action_path):
    # check for action eligibility
    flag = True
    tmp_env, _, _, _, _ = get_current_state(action_path)
    obs_new, reward_new, done_new, info = tmp_env.step(a)
    mapping = tuple(
        np.ndindex(((2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1)))
    tmp_action = mapping[a]
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
    if (tmp_action[0] + tmp_action[1] + tmp_action[2]) > MAX_ALLOWED_WORKER:
        flag = False

    if (tmp_action[0] + tmp_action[1] + tmp_action[2]) < 0:
        flag = False

    return flag


def expand_nodes(new_c, k, new_tree, econ_model, env_model, social_model, c, parent_id, device):
    total_values = []
    total_index = []
    track_identifier = []
    ll = 0
    for i in range(k):
        action_list = get_action_path(new_tree, new_c[ll])
        if len(action_list) == 2:
            action_list = np.flip(action_list)
        env_econ, enc_env, env_social, obs, done = get_current_state(action_list)
        idx, value = choose_action(obs, econ_model, obs, env_model, obs, social_model, device, k)
        added = 0
        start_id = 0
        while added < k:
            current_path = get_action_path(new_tree, new_c[ll])
            current_path = np.flip(current_path)
            chosen_action = idx[start_id]
            current_path = np.append(current_path, chosen_action)
            result = check_eligibility(chosen_action, current_path)
            if result:
                added += 1
                node_name = [idx[start_id], value[start_id]]
                new_tree.create_node(node_name, c, parent=parent_id[i])
                total_values = np.append(total_values, value[start_id])
                total_index = np.append(total_index, idx[start_id])
                track_identifier = np.append(track_identifier, c)
                c += 1
            start_id += 1
        ll += 1
        new_c = track_identifier

    return new_tree, new_c, c, total_values, total_index, track_identifier


def remove_nodes(total_idx, total_values, track_identifier, k, new_tree):
    sorted_idx = total_idx[np.argsort(-total_values)]
    sorted_idx = sorted_idx.astype('int64')
    sorted_track = track_identifier[np.argsort(-total_values)]
    rm_identifiers = sorted_track[k:]
    remove_ids = sorted_idx[k:]
    for j in range(remove_ids.shape[0]):
        new_tree.remove_node(rm_identifiers[j])
    return new_tree, rm_identifiers


def test_chosen_path(econ_env, env_env, social_env, in_path):

    print('Testing chosen path')
    total_reward_econ = []
    total_reward_env = []
    total_reward_social = []

    for i in range(len(in_path)):

        a_i = in_path[i]

        np.random.seed(i + 110)
        obs_econ, reward_econ, done_econ, info = econ_env.step(a_i)
        total_reward_econ = np.append(total_reward_econ, reward_econ)

        obs_env, reward_env, done_env, info = env_env.step(a_i)
        total_reward_env = np.append(total_reward_env, reward_env)

        obs_social, reward_social, done_social, info = social_env.step(a_i)
        total_reward_social = np.append(total_reward_social, reward_social)

        print('Current State= ', obs_econ)
        print('Action=', a_i)
        if done_econ is True:
            remaining_b1 = round(social_env.budget, 3)
            print('Remaining Budget = ', str(remaining_b1) + '/' + str(BUDGET), 'Un-pruned Plants = ',
                  str(social_env.plants) + '/' + str(PLANTS))
            break


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
    print('******************************************************')

    k = beam_size

    # first time step - creating first k (k = beam-size) nodes of the tree
    action_ids, action_values = choose_action(obs_econ, econ_model, obs_env, env_model, obs_social, social_model,
                                              device, k)

    st = time.time()

    # creating the tree to keep track of best policies
    new_tree = Tree()
    new_tree.create_node(0, 1)
    last_added_identifier = 1
    new_tree.show()
    c = 2
    added = 0
    start_id = 0

    while added < k:
        current_path = []
        chosen_action = action_ids[start_id]
        # checking if the selected action violate models constraint
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

        new_tree, new_c, c, total_values, total_index, track_identifier = expand_nodes(new_c, k, new_tree, econ_model,
                                                                                       env_model, social_model, c,
                                                                                       parent_id, device)
        new_tree.show()
        parent_id_tracker = []

        for o in range(k ** 2):
            parent_id_tracker = np.append(parent_id_tracker, get_parent(new_tree, new_c[o]))

        new_tree, rm_identifiers = remove_nodes(total_index, total_values, track_identifier, k, new_tree)
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
    print('beam search time', et - st)
    for h in range(k):
        current_path = get_action_path(new_tree, new_c[h])
        current_path = np.flip(current_path)
        econ_env = EconomicEnv()
        env_env = EnvironmentEnv()
        social_env = SocialEnv()
        test_chosen_path(econ_env, env_env, social_env, current_path)
        print('***************************************')


if __name__ == '__main__':
    main()
