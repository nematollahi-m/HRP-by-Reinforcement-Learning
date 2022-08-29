# Final Model
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as pt
from itertools import product


def extract_possible_actions_list(state_key):
    '''
        Extracts possible number of actions given current state
        Args:
            state_key:          The key of the state
        Returns:
            List of possible actions for the next step.
    '''
    w = state_key[:-1]
    A = sum(w)
    remain = _total_allowed - A

    track = []
    for r in range(remain + 1):
        track.append(r)
    cc = product(track, repeat=_num_knowledge_level)
    G = list(cc)
    possible_action_list = []

    for i in G:
        if sum(i) > remain:
            continue
        else:
            possible_action_list.append(i)
    return possible_action_list


def select_action(state_key, next_action_list, q_df):
    '''
        Select action by Q(state, action).
        Concreat method.
        ε-greedy.
        Args:
            state_key:              The key of state.
            next_action_list:       The possible action in `self.t+1`.
                                    If the length of this list is 0, all action should be possible.
        Returns:
            The key of action.
    '''
    epsilon_greedy_flag = bool(np.random.binomial(n=1, p=_epsilon))

    if epsilon_greedy_flag is False:
        action_key = random.choice(next_action_list)

    else:
        action_key = predict_next_action(state_key, next_action_list, q_df)

    return action_key


def update_state(state_key, action_key):
    '''
        Updating the state according to the selectec action.
        Args:
            state_key:          The key of state
            action_key          The action at this state
        Returns:
            Updated state
    '''
    action = list(action_key)
    action.append(1)
    state_key = np.add(state_key, action)

    return tuple(state_key)


def predict_next_action(state_key, next_action_list, q_df):
    '''
        Predict next action by Q-Learning.
        Args:
            state_key:          The key of state in `self.t+1`.
            next_action_list:   The possible action in `self.t+1`.
        Returns:
            The key of action.
    '''

    if q_df is not None:
        next_action_q_df = q_df[q_df.state_key == state_key]
        next_action_q_df = next_action_q_df[next_action_q_df.action_key.isin(
            next_action_list)]
        if next_action_q_df.shape[0] == 0:
            return random.choice(next_action_list)
        else:
            if next_action_q_df.shape[0] == 1:
                max_q_action = next_action_q_df["action_key"].values[0]
            else:
                next_action_q_df = next_action_q_df.sort_values(
                    by=["q_value"], ascending=False)
                max_q_action = next_action_q_df.iloc[0, :]["action_key"]
            return max_q_action
    else:
        return random.choice(next_action_list)


def get_productivity():
    '''
        The productivity rate of workers from a normal distirbution
        Returns:
            Ratee of the productivity of each level worker.
    '''
    mu = _productivity
    sigma = 1
    p = []
    for i in range(len(mu)):
        temp = np.random.normal(mu[i], sigma, 1)
        p.append(temp[0])
    return p


def pruned(num_workers):
    '''
        Calculates the number of pruned plants at each state
        Args:
            num_workers:        Number of workers at each level
        Returns:
            The number of pruned plants based on the num workers and their productivity
    '''
    prune_rate = get_productivity()
    temp = np.round(np.dot(prune_rate, num_workers))
    return int(temp)


def observe_reward_value(state_key, action_key, time_step, remaining_plants):
    '''
        Calculating the reward of each state and action
        Args:
            state_key:              The key of the state
            action_key:             The key of the action
            time_step:              Current time step
            remaining_plants:       The remaining number of plants to prune
        Returns:
            A float number indicating the reward.
    '''

    w = state_key[:-1]

    if time_step == _time_step - 1:
        print('END POINT')
        # Cost
        hired = np.add(list(w), list(action_key))
        worker_cost = np.dot(list(action_key), _hiring_cost) + np.dot(
            hired, _wage)
        # Plants
        num_pruned = pruned(hired)
        remaining_plants += num_pruned
        penalty = (_alpha * worker_cost) + (_beta * remaining_plants *
                                            _penaltyRate * _extra_penalty_rate)
        return penalty, worker_cost, remaining_plants
    else:

        hired = np.add(w, action_key)
        worker_cost = np.dot(action_key, _hiring_cost) + np.dot(hired, _wage)
        num_pruned = pruned(hired)
        remaining_plants += num_pruned
        penalty = (_alpha * worker_cost) + (_beta * remaining_plants *
                                            _penaltyRate)
        return penalty, worker_cost, remaining_plants


def extract_q_df(state_key, action_key, q_df):
    '''
        Extract Q-Value from `self.q_df`.
        Args:
            state_key:      The key of state.
            action_key:     The key of action.
        Returns:
            Q-Value.
    '''
    q = 0.0

    if q_df is None:
        q_df_temp = save_q_df(state_key, action_key, q, q_df)
        return q, q_df_temp
    q_df_h = q_df[q_df.state_key == state_key]
    q_df_b = q_df_h[q_df_h.action_key == action_key]

    if q_df_b.shape[0]:
        q = float(q_df_b["q_value"])
    else:
        q_df = save_q_df(state_key, action_key, q, q_df)
    return q, q_df


def save_q_df(state_key, action_key, q_value, q_df):
    '''
        Insert or update Q-Value in `self.q_df`.
        Args:
            state_key:      State.
            action_key:     Action.
            q_value:        Q-Value.
        Exceptions:
            TypeError:      If the type of `q_value` is not float.
    '''
    # a1, a2 = action_key
    # action_key = (a1, a2)
    if isinstance(q_value, float) is False:
        raise TypeError("The type of q_value must be float.")

    new_q_df = pd.DataFrame([(state_key, action_key, q_value)],
                            columns=["state_key", "action_key", "q_value"])
    if q_df is not None:
        q_df = pd.concat([new_q_df, q_df])
        q_df = q_df.drop_duplicates(["state_key", "action_key"])
        return q_df
    else:
        q_df = new_q_df
        return q_df


def update_q(state_key, action_key, reward_value, next_max_q, q_df):
    '''
        Update Q-Value.
        Args:
            state_key:              The key of state.
            action_key:             The key of action.
            reward_value:           R-Value(Reward).
            next_max_q:             Maximum Q-Value.
    '''
    # Now Q-Value.
    q, q_df = extract_q_df(state_key, action_key, q_df)
    # Update Q-Value.
    new_q = q + _learning_rate * (reward_value + (_gamma * next_max_q) - q)
    # Save updated Q-Value.
    q_df = save_q_df(state_key, action_key, new_q, q_df)
    return q_df


def inference(q_table):
    route_list = []
    initial_state = (0, 0, 0)
    for i in range(_time_step):
        q_df = q_table[q_table.state_key == initial_state]
        if q_df.shape[0] > 1:
            q_df = q_df.sort_values(by=["q_value"], ascending=False)
            action_key = q_df.iloc[0, :]["action_key"]
            q_value = q_df.iloc[0, :]["q_value"]
        elif q_df.shape[0] == 1:
            action_key = q_df.action_key.values[0]
            q_value = q_df.q_value.values[0]
        else:
            action_key_list = extract_possible_actions_list(initial_state)
            #action_key_list = [v for v in action_key_list if v not in memory_list]
            q_value = 0.0
            if len(action_key_list):
                action_key = random.choice(action_key_list)
                _q_df = q_df[q_df.action_key == action_key]
                if _q_df.shape[0]:
                    q_value = _q_df.q_value.values[0]

        initial_state = update_state(state_key=initial_state,
                                     action_key=action_key)
        #        x, y, z = initial_state
        route_list.append((initial_state, q_value))
        print(i)
        #memory_list.append(state_key)

    return route_list


# initializing the variables

# Number of knowledge levels
_num_knowledge_level = 2
# penalty at the end of each iteration
_extra_penalty_rate = 1000
# total number of allowed workers
_total_allowed = 10
# number of plants to prune
_initial_num_plants = 45000
# Available budget
_budget = 600000
# Number of time steps (e.g. 5 weeks)
_time_step = 5
# number of iteration
_epochs = 3000
q_table = None
# Hyperparameters
_learning_rate = 0.1
_gamma = 0.1
_epsilon = 0.6
_alpha = 0.8
_beta = 0.2

num_hired = [0, 0]
# number of plants each level worker, prunes (hour)
_productivity = [-2.85, -15.2]
# the cost of hiring at each level
_hiring_cost = [-12.5, -70.5]
# Salary of each level
_wage = [-17, -26]
# Parameters of normal distibution
mu = 6
sigma = 1
# Defining the state in the format of state = (level1, level2, ... ,time step)
start_state = (0, 0, 0)

# for leaving the plants unpruned at each stage!
_penaltyRate = -10

# tracking the agents learning
track_res = np.zeros(_epochs)

# Learning Section
for i in range(_epochs):
    state = start_state
    _num_plants = _initial_num_plants
    _total_cost = 0
    for j in range(_time_step):
        # Extracting possible actions from current state
        possible_action_list = extract_possible_actions_list(state)
        # Choosing an action based on the ε-greedy method
        action = select_action(state, possible_action_list, q_table)
        # Calculating the reward
        reward, cost, _num_plants = observe_reward_value(
            state, action, j, _num_plants)
        if _num_plants <= 0:
            print('All the plants are pruned!')
            break
        _total_cost += cost
        if _total_cost > _budget:
            print('Out of budget!')
            break
        # finding next state based on the action
        new_state_key = update_state(state, action)
        # extracting new action list
        next_next_action_list = extract_possible_actions_list(new_state_key)
        # predicting the next possible action
        next_action_key = predict_next_action(new_state_key,
                                              next_next_action_list, q_table)
        # extracting the q-value of it
        next_max_q, q_table = extract_q_df(new_state_key, next_action_key,
                                           q_table)
        # updating the q-table according to q-value and reward
        q_table = update_q(state_key=state,
                           action_key=action,
                           reward_value=reward,
                           next_max_q=next_max_q,
                           q_df=q_table)
        # moving to the next state
        state = new_state_key

    track_res[i] = _total_cost
    print('Done with round:', i)

print(q_table)

A = inference(q_table)
print(A)

pt.figure()
pt.plot(track_res)
pt.show()
