from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from parameters import *


class FarmEnvEnv(Env):
    def __init__(self):
        super().__init__()
        # state = (number of hired workers, remaining budget , remaining plants)
        self.state = np.array([0, BUDGET, PLANTS])
        self.action_space = Discrete((MAX_ALLOWED_WORKER + 1) * (MAX_ALLOWED_WORKER + 1))
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([MAX_ALLOWED_WORKER, BUDGET, PLANTS]))
        self.prune_len = PRUNE_LENGTH

    def step(self, action):

        done = False
        info = {}

        available_workers = round(np.random.normal(WORKER_AVAILABILITY, 1))

        mapping = tuple(np.ndindex((MAX_ALLOWED_WORKER + 1, MAX_ALLOWED_WORKER + 1)))
        new_action = mapping[action]

        m_t_1 = self.state[0]
        b_t = self.state[1]
        p_t = self.state[2]

        h_t = new_action[0]
        f_t = new_action[1]

        if h_t > available_workers:
            r_t = -M
            return self.state, r_t, done, info

        if m_t_1 + h_t > MAX_ALLOWED_WORKER:
            r_t = -M
            return self.state, r_t, done, info

        if m_t_1 - f_t < 0:
            r_t = -M
            return self.state, r_t, done, info

        m_t = m_t_1 + h_t - f_t

        c_t = (HIRE_COST * h_t) + (FIRE_COST * f_t) + (WAGE * m_t)
        pl_t = m_t * np.random.normal(PRODUCTIVITY, 0)

        if pl_t >= p_t:
            c_t = (HIRE_COST * h_t) + (FIRE_COST * f_t) + ((p_t / pl_t) * WAGE * m_t)
            if c_t > b_t:
                print('Pruned more than available but exceeded the budget!')
                r_t = -M
                done = True
                return self.state, r_t, done, info
            else:
                self.state = np.asarray([m_t, b_t - int(c_t), 0])
                print('Goal achieved **************************************************************************')
                if c_t == 0:
                    r_t = -M
                else:
                    r_t = (QUALITY - (c_t / BUDGET)) * M
                done = True
                return self.state, r_t, done, info

        if c_t > b_t:
            print('Exceeded the budget!')
            r_t = -M
            done = True
            return self.state, r_t, done, info

        if self.prune_len <= 1:
            print('End of season')
            if c_t == 0:
                r_t = -M
            else:
                r_t = -M
            self.state = np.asarray([m_t, b_t - int(c_t), p_t - int(pl_t)])
            done = True
            return self.state, r_t, done, info

        if c_t == 0 or pl_t == 0:
            r_t = -M
        else:
            r_t = QUALITY * (pl_t / PLANTS) - (c_t / BUDGET)

        self.state = np.asarray([m_t, b_t - int(c_t), p_t - int(pl_t)])
        self.prune_len -= 1

        return self.state, r_t, done, info

    def render(self):
        pass

    def reset(self):
        self.state = np.asarray([0, BUDGET, PLANTS])
        self.prune_len = PRUNE_LENGTH
        return self.state
