import numpy as np
import os
import torch as th
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
from src.Parameters import *
from src.env.EconomicObj import EconomicEnv
from src.env.EnvironmentObj import EnvironmentEnv
from src.env.SocialObj import SocialEnv
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
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

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
economic_env = EconomicEnv()
log_dir = "../../tmp/Economic/"
os.makedirs(log_dir, exist_ok=True)

economic_env = Monitor(economic_env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
economic_model = DQN('MlpPolicy', economic_env, learning_rate=0.0001, exploration_fraction=0.8, exploration_final_eps=0.001,
                     exploration_initial_eps=1.0, verbose=1, device=device)
economic_model.learn(total_timesteps=time_steps, callback=callback)

print('********* Done with training Economic *********')

# saving the model
print('********* Saving the economic model *********')

economic_model.save("../../model/economic_model")
loaded_economic_model = DQN.load("../../model/economic_model")
print(f"The loaded_model has {loaded_economic_model.replay_buffer.size()} transitions in its buffer")

economic_model.save_replay_buffer("../../model/economic_model_replay_buffer")
loaded_economic_model.load_replay_buffer("../../model/economic_model_replay_buffer")
print(f"The loaded_model has {loaded_economic_model.replay_buffer.size()} transitions in its buffer")

economic_policy = economic_model.policy
economic_policy.save("../../model/economic_policy")


"""
    Training Environment Agent - saving the model
"""
print('********* Training Environment *********')
environmnet_env = EnvironmentEnv()
log_dir = "../../tmp/Environmental/"
os.makedirs(log_dir, exist_ok=True)
environmnet_env = Monitor(environmnet_env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
environmnet_model = DQN('MlpPolicy', environmnet_env, learning_rate=0.0001, exploration_fraction=0.8, exploration_final_eps=0.001,
                     exploration_initial_eps=1.0, verbose=1, device=device, tensorboard_log="./dqn_environmnet/")
environmnet_model.learn(total_timesteps=time_steps, callback=callback)
print('********* Done with training Environment *********')

# saving the model
print('********* Saving the environment model *********')
environmnet_model.save("../../model/environmental_model")
loaded_environmnet_model = DQN.load("../../model/environmental_model")
print(f"The loaded_model has {loaded_environmnet_model.replay_buffer.size()} transitions in its buffer")

environmnet_model.save_replay_buffer("../../model/environmental_model_replay_buffer")
loaded_environmnet_model.load_replay_buffer("../../model/environmental_model_replay_buffer")
print(f"The loaded_model has {loaded_environmnet_model.replay_buffer.size()} transitions in its buffer")

environmnet_policy = environmnet_model.policy
environmnet_policy.save("../../model/environmental_policy")

"""
    Training Social Agent - saving the model
"""
social_env = SocialEnv()
log_dir = "../../tmp/Social/"
os.makedirs(log_dir, exist_ok=True)
social_env = Monitor(social_env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
social_model = DQN('MlpPolicy', social_env, learning_rate=0.0001, exploration_fraction=0.8, exploration_final_eps=0.001,
                     exploration_initial_eps=1.0, verbose=1, device=device, tensorboard_log="./dqn_social/")
social_model.learn(total_timesteps=time_steps, callback=callback)

social_model.save("../../model/social_model")
loaded_social_model = DQN.load("../../model/social_model")
print(f"The loaded_model has {loaded_social_model.replay_buffer.size()} transitions in its buffer")

social_model.save_replay_buffer("../../model/social_model_replay_buffer")
loaded_social_model.load_replay_buffer("../../model/social_model_replay_buffer")
print(f"The loaded_model has {loaded_social_model.replay_buffer.size()} transitions in its buffer")

social_policy = social_model.policy
social_policy.save("../../model/social_policy")



