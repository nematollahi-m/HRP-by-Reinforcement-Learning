from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from Parameters import *


class SocialEnv(Env):
    def __init__(self):
        super().__init__()
        self.state = np.array([0, 0, 0])
        self.action_space = Discrete((2 * MAX_ALLOWED_WORKER + 1) ** 3)
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array(
            [MAX_ALLOWED_WORKER, MAX_ALLOWED_WORKER, MAX_ALLOWED_WORKER]))
        self.prune_len = PRUNE_LENGTH
        self.budget = BUDGET
        self.plants = PLANTS

    def step(self, action):

        done = False
        info = {}

        mapping = tuple(
            np.ndindex(((2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1)))
        new_action = mapping[action]
        new_action = list(new_action)
        for o in range(0, len(new_action)):
            if new_action[o] > MAX_ALLOWED_WORKER:
                new_action[o] = MAX_ALLOWED_WORKER - new_action[o]
        new_action = tuple(new_action)

        m_b_t_1 = self.state[0]
        m_i_t_1 = self.state[1]
        m_a_t_1 = self.state[2]
        p_t = self.plants

        h_b_t = new_action[0]
        h_i_t = new_action[1]
        h_a_t = new_action[2]

        availability_beg = round(np.random.normal(WORKER_AVAILABILITY_BEG, 3))
        availability_int = round(np.random.normal(WORKER_AVAILABILITY_INT, 2))
        availability_adv = round(np.random.normal(WORKER_AVAILABILITY_ADV, 1))

        if (h_b_t > availability_beg) or (h_i_t > availability_int) or (h_a_t > availability_adv):
            r_t = -M
            return self.state, r_t, done, info

        if (m_b_t_1 + m_i_t_1 + m_a_t_1) + (h_b_t + h_i_t + h_a_t) > MAX_ALLOWED_WORKER:
            r_t = -M
            return self.state, r_t, done, info

        if (m_b_t_1 + h_b_t < 0) or (m_i_t_1 + h_i_t < 0) or (m_a_t_1 + h_a_t < 0):
            r_t = -M
            return self.state, r_t, done, info

        m_b_t = m_b_t_1 + h_b_t
        m_i_t = m_i_t_1 + h_i_t
        m_a_t = m_a_t_1 + h_a_t

        c_hire = HIRE_COST * (max(0, h_b_t) + max(0, h_i_t) + max(0, h_a_t))
        c_fire = FIRE_COST * (max(0, -h_b_t) + max(0, -h_i_t) + max(0, -h_a_t))
        c_wage = (WAGE_BEG * m_b_t) + (WAGE_INT * m_i_t) + (WAGE_ADV * m_a_t)

        c_t = c_hire + c_fire + c_wage

        pl_b_t = m_b_t * np.random.normal(PRODUCTIVITY_BEG, 0.03448)
        pl_i_t = m_i_t * np.random.normal(PRODUCTIVITY_INT, 0.0258)
        pl_a_t = m_a_t * np.random.normal(PRODUCTIVITY_ADV, 0.01724)

        pl_t = pl_b_t + pl_i_t + pl_a_t

        if pl_t >= p_t:

            c_hire = HIRE_COST * (max(0, h_b_t) + max(0, h_i_t) + max(0, h_a_t))
            c_fire = FIRE_COST * (max(0, -h_b_t) + max(0, -h_i_t) + max(0, -h_a_t))
            c_wage = ((WAGE_BEG * m_b_t) + (WAGE_INT * m_i_t) + (WAGE_ADV * m_a_t)) * (p_t / pl_t)
            c_t = c_hire + c_fire + c_wage

            if c_t > self.budget:
                r_t = -M
                done = True
                return self.state, r_t, done, info
            else:
                self.state = np.asarray([m_b_t, m_i_t, m_a_t])
                self.plants = 0
                self.budget = self.budget - c_t
                print('********************* Goal Achieved **************************')
                if c_t == 0:
                    r_t = -M
                else:
                    r_t = (m_b_t + m_i_t + m_a_t - max(0, -h_b_t) - max(0, -h_i_t) - max(0,
                                                                                         -h_a_t)) / MAX_ALLOWED_WORKER + 1

                done = True
                return self.state, r_t, done, info

        if c_t > self.budget:
            r_t = -M
            done = True
            return self.state, r_t, done, info

        if self.prune_len <= 1:
            r_t = (m_b_t + m_i_t + m_a_t - max(0, -h_b_t) - max(0, -h_i_t) - max(0, -h_a_t)) / MAX_ALLOWED_WORKER

            self.state = np.asarray([m_b_t, m_i_t, m_a_t])
            self.budget = self.budget - c_t
            self.plants = self.plants - pl_t
            done = True
            return self.state, r_t, done, info

        r_t = (m_b_t + m_i_t + m_a_t - max(0, -h_b_t) - max(0, -h_i_t) - max(0, -h_a_t)) / MAX_ALLOWED_WORKER

        self.state = np.asarray([m_b_t, m_i_t, m_a_t])
        self.prune_len -= 1
        self.budget = self.budget - c_t
        self.plants = self.plants - pl_t

        return self.state, r_t, done, info

    def render(self):
        pass

    def reset(self):
        self.state = np.asarray([0, 0, 0])
        self.prune_len = PRUNE_LENGTH
        self.budget = BUDGET
        self.plants = PLANTS
        return self.state


