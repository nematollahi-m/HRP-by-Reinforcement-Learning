import numpy as np
from stable_baselines3 import DQN
from src.env.EconomicObj import EconomicEnv
from src.env.EnvironmentObj import EnvironmentEnv
from src.env.SocialObj import SocialEnv
from Parameters import *
from stable_baselines3.common.utils import obs_as_tensor
from saveFunctions import *
import math
import warnings
from action import *
import torch as th

warnings.filterwarnings('ignore')

'''
    Before running this code make sure three environments are trained and saved in model folder. 
'''


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



def main():
    if th.cuda.is_available():
        print('GPU is available')
        device = 'cuda'
    else:
        print('No GPU')
        device = 'cpu'


if __name__ == '__main__':
    main()
