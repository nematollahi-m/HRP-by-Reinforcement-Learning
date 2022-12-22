import numpy as np
import matplotlib.pyplot as plt
import torch as th
from stable_baselines3.common.results_plotter import load_results, ts2xy



def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title='Learning Curve'):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

def plot_results_test(log_folder):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    x = x[len(x) - len(y):]
    return x, y



if th.cuda.is_available():
    print('GPU is available')
    device = 'cuda'
else:
    print('No GPU')
    device = 'cpu'
