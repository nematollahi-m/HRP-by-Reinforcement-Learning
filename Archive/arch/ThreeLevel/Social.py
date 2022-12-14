
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from Parameters import *


class SocialEnv(Env):
    def __init__(self):
        super().__init__()
        # 0: False 1: True
        # state = (number of hired workers_beg ,number of hired workers_med, number of hired workers_adv ,
        # Ran out of budget? , remaining plants)
        self.state = np.array([0, 0, 0, 0, PLANTS])
        self.action_space = Discrete(((MAX_ALLOWED_WORKER + 1) * (MAX_ALLOWED_WORKER + 1)) ** 3)
        self.observation_space = Box(low=np.array([0, 0, 0, 0, 0]), high=np.array(
            [MAX_ALLOWED_WORKER, MAX_ALLOWED_WORKER, MAX_ALLOWED_WORKER, 1, PLANTS]))
        self.prune_len = PRUNE_LENGTH
        self.budget = BUDGET

    def step(self, action):

        done = False
        info = {}

        availability_beg = round(np.random.normal(WORKER_AVAILABILITY_BEG, 1))
        availability_int = round(np.random.normal(WORKER_AVAILABILITY_INT, 1))
        availability_adv = round(np.random.normal(WORKER_AVAILABILITY_ADV, 1))

        mapping = tuple(np.ndindex((MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1,
                                    MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1)))
        new_action = mapping[action]

        m_b_t_1 = self.state[0]
        m_i_t_1 = self.state[1]
        m_a_t_1 = self.state[2]
        p_t = self.state[4]

        h_b_t = new_action[0]
        f_b_t = new_action[1]
        h_i_t = new_action[2]
        f_i_t = new_action[3]
        h_a_t = new_action[4]
        f_a_t = new_action[5]

        if (h_b_t > availability_beg) or (h_i_t > availability_int) or (h_a_t > availability_adv):
            r_t = -M
            return self.state, r_t, done, info

        if (m_b_t_1 + m_i_t_1 + m_a_t_1) + (h_b_t + h_i_t + h_a_t) > MAX_ALLOWED_WORKER:
            r_t = -M
            return self.state, r_t, done, info

        if (m_b_t_1 - f_b_t < 0) or (m_i_t_1 - f_i_t < 0) or (m_a_t_1 - f_a_t < 0):
            r_t = -M
            return self.state, r_t, done, info

        m_b_t = m_b_t_1 + h_b_t - f_b_t
        m_i_t = m_i_t_1 + h_i_t - f_i_t
        m_a_t = m_a_t_1 + h_a_t - f_a_t

        c_hire = HIRE_COST * (h_b_t + h_i_t + h_a_t)
        c_fire = FIRE_COST * (f_b_t + f_i_t + f_a_t)
        c_wage = (WAGE_BEG * m_b_t) + (WAGE_INT * m_i_t) + (WAGE_ADV * m_a_t)

        c_t = c_hire + c_fire + c_wage

        pl_b_t = m_b_t * round(np.random.normal(PRODUCTIVITY_BEG, 1))
        pl_i_t = m_i_t * round(np.random.normal(PRODUCTIVITY_INT, 1))
        pl_a_t = m_a_t * round(np.random.normal(PRODUCTIVITY_ADV, 1))
        pl_t = pl_b_t + pl_i_t + pl_a_t

        new_h = h_b_t + h_i_t + h_a_t
        new_f = f_b_t + f_i_t + f_a_t
        new_m = m_b_t + m_i_t + m_a_t + 1
        new_m1 = m_b_t_1 + m_i_t_1 + m_a_t_1 + 1

        if pl_t >= p_t:

            c_wage = ((WAGE_BEG * m_b_t) + (WAGE_INT * m_i_t) + (WAGE_ADV * m_a_t)) * (p_t / pl_t)
            c_t = c_hire + c_fire + c_wage

            if c_t > self.budget:
                print('FAIL: Pruned more than available but exceeded the budget')
                r_t = -M
                done = True
                return self.state, r_t, done, info
            else:
                self.state = np.asarray([m_b_t, m_i_t, m_a_t, 0, 0])
                self.budget = self.budget - c_t
                print('SUCCESS **********************************************************************************')
                r_t = ((new_h/new_m) - (new_f/new_m1) + 1) * M
                done = True
                return self.state, r_t, done, info

        if c_t > self.budget:
            r_t = -M
            done = True
            return self.state, r_t, done, info

        if self.prune_len <= 1:
            print('End of the season')
            r_t = -M
            self.state = np.asarray([m_b_t, m_i_t, m_a_t, 0, 0])
            self.budget = self.budget - c_t
            done = True
            return self.state, r_t, done, info

        r_t = ((new_h/new_m) - (new_f/new_m1) + 1) * M
        self.state = np.asarray([m_b_t, m_i_t, m_a_t, 0, 0])
        self.prune_len -= 1
        self.budget = self.budget - c_t

        return self.state, r_t, done, info

    def render(self):
        pass

    def reset(self):
        self.state = np.asarray([0, 0, 0, 0, PLANTS])
        self.prune_len = PRUNE_LENGTH
        self.budget = BUDGET
        return self.state
