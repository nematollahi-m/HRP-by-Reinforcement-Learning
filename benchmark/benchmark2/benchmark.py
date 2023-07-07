import numpy as np
import os
import torch as th
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from Parameters import *
from AggregatedEnv import Aggregated
import warnings

warnings.filterwarnings('ignore')


class SaveOnBestTrainingRewardCallback(BaseCallback):
    # This class is based on the code: https://stable-baselines.readthedocs.io/en/master/guide/examples.html

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


if th.cuda.is_available():
    print('GPU is available')
    device = 'cuda'
else:
    print('No GPU')
    device = 'cpu'

"""
    Training Economic Agent - saving the model
"""
print('********* Training Economic *********')
agg_env = Aggregated()
log_dir = "../../tmp/Aggregated/"
os.makedirs(log_dir, exist_ok=True)

agg_env = Monitor(agg_env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
agg_model = DQN('MlpPolicy', agg_env, learning_rate=0.0001, exploration_fraction=0.8, exploration_final_eps=0.001,
                exploration_initial_eps=1.0, verbose=1, device=device)
agg_model.learn(total_timesteps=time_steps, callback=callback)

print('********* Done with training Aggregated *********')

# saving the model
print('********* Saving the aggregated model *********')

agg_model.save("../../model/agg_model")
loaded_agg_model = DQN.load("../../model/agg_model")
print(f"The loaded_model has {loaded_agg_model.replay_buffer.size()} transitions in its buffer")

agg_model.save_replay_buffer("../../model/agg_model_replay_buffer")
loaded_agg_model.load_replay_buffer("../../model/agg_model_replay_buffer")
print(f"The loaded_model has {loaded_agg_model.replay_buffer.size()} transitions in its buffer")

agg_policy = agg_model.policy
agg_policy.save("../../model/agg_policy")

# loading the model

path = "../../model/agg_model"
loaded_model = DQN.load(path)

agg_env = Aggregated()
obs_agg = agg_env.reset()
n_steps = PRUNE_LENGTH
print('******************************************************')
track_reward_agg = []
mapping = tuple \
    (np.ndindex(((2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1, (2 * MAX_ALLOWED_WORKER) + 1)))

for step in range(n_steps):

    action_agg, _ = loaded_model.predict(obs_agg, deterministic=True)
    action_agg_unroll = mapping[action_agg]
    print("Selected Action:", action_agg_unroll)
    zz = "Selected Action: " + str(action_agg_unroll) + '\n'

    obs_agg, reward_agg, done_agg, info = agg_env.step(action_agg)
    track_reward_agg = np.append(track_reward_agg, reward_agg)
    print('State', obs_agg, 'Reward= ', reward_agg)
    zz = 'State: ' + str(obs_agg) + 'Reward= ' + str(reward_agg) + '\n'

    print('******************************************************')

    if done_agg is True:
        remaining_b1 = round(agg_env.budget, 3)
        print('Remaining Budget = ', str(remaining_b1) + '/' + str(BUDGET), 'Un-pruned Plants = ',
              str(agg_env.plants) + '/' + str(PLANTS))
        zz = 'Remaining Budget = ' + str(remaining_b1) + '/' + str(BUDGET) + 'Un-pruned Plants = ' + str(
            agg_env.plants) + '/' + str(PLANTS) + '\n'
        break
