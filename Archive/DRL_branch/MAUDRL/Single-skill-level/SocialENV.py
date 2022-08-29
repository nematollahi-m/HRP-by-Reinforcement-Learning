
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from parameters import *


class FarmEnvSocial(Env):
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
            # print('No enough worker in market to hire!')
            r_t = -M * beta
            return self.state, r_t, done, info

        if m_t_1 + h_t > MAX_ALLOWED_WORKER:
            # print('I cannot hire this many workers!')
            r_t = -M * beta
            return self.state, r_t, done, info

        if m_t_1 - f_t < 0:
            # print('I cannot fire this many workers!')
            r_t = -M * beta
            return self.state, r_t, done, info

        m_t = m_t_1 + h_t - f_t

        c_t = (HIRE_COST * h_t) + (FIRE_COST * f_t) + (WAGE * m_t)
        pl_t = m_t * round(np.random.normal(PRODUCTIVITY, 1))

        if pl_t > p_t:
            x = pl_t
            pl_t = p_t
            c_t = (HIRE_COST * h_t) + (FIRE_COST * f_t) + ((pl_t / x) * WAGE * m_t)
            if c_t > b_t:
                print('I pruned more than available but exceeded the budget!')
                r_t = -M * beta
                done = True
                return self.state, r_t, done, info
            else:
                self.state = [m_t, b_t - c_t, 0]
                print(
                    '********************************** I reached the goal ************************************************')
                if m_t == 0:
                    r_t = 0
                elif m_t_1 == 0:
                    r_t = 0
                else:
                    r_t = alpha * (h_t / m_t) - beta * (f_t / m_t_1) + delta * (p_t / PLANTS)

                if m_t == 0:
                    r_t = 0
                elif m_t_1 == 0:
                    r_t = 0
                else:
                    r_t = alpha * (h_t / m_t) - beta * (f_t / m_t_1) + delta * (p_t / PLANTS)
                done = True
                return self.state, r_t, done, info

        if c_t > b_t:
            print('I exceeded the budget!')
            r_t = -M * beta
            done = True
            return self.state, r_t, done, info

        if self.prune_len <= 0:
            print('I reached end of season! ')
            r_t = (alpha + (beta * QUALITY)) * (pl_t / PLANTS)
            self.state = [m_t, b_t - c_t, p_t - pl_t]
            done = True
            return self.state, r_t, done, info

        if m_t == 0:
            r_t = 0
        elif m_t_1 == 0:
            r_t = 0
        else:
            r_t = alpha * (h_t / m_t) - beta * (f_t / m_t_1) + delta * (p_t / PLANTS)
        # print('I am still making decision')
        self.state = [m_t, b_t - c_t, p_t - pl_t]
        self.prune_len -= 1

        return self.state, r_t, done, info

    def render(self):
        pass

    def reset(self):
        # self.state = np.asarray([random.randint(0, MAX_ALLOWED_WORKER), random.randint(0, BUDGET),
        #                           random.randint(0, PLANTS)])
        self.state = np.asarray([0, BUDGET, PLANTS])
        self.prune_len = PRUNE_LENGTH
        return self.state
