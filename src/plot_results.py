import matplotlib.pyplot as plt
from saveFunctions import *

# plotting the results of training three objectives:
log_dir_econ = "../tmp/Economic/"
log_dir_env = "../tmp/Environmental/"
log_dir_social = "../tmp/Social/"
x_econ, y_econ = plot_results_test(log_dir_econ)
x_env, y_env = plot_results_test(log_dir_env)
x_social, y_social = plot_results_test(log_dir_social)

title = 'Learning Curve'
fig = plt.figure(title)
plt.plot(x_econ, y_econ, 'r')
plt.plot(x_env, y_env, 'b')
plt.plot(x_social, y_social, 'g')
plt.xlabel('Number of Timesteps')
plt.ylabel('Rewards')
plt.title(title)
plt.legend(["Econ","Env","Soc"])
plt.show()