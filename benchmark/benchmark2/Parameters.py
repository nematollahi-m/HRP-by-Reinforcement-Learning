""" model parameters """

MAX_ALLOWED_WORKER = 12
BUDGET = 100000
PLANTS = 30 * 1450

WAGE_BEG = 15.65 * 8 * 5
WAGE_INT = 20 * 8 * 5
WAGE_ADV = 30 * 8 * 5

PRODUCTIVITY_BEG = 700 / (2 * 1450)
PRODUCTIVITY_INT = 1100 / (2 * 1450)
PRODUCTIVITY_ADV = 1500 / (2 * 1450)

STD_PRODUCTIVITY_BEG = 100 / (2 * 1450)
STD_PRODUCTIVITY_INT = 75 / (2 * 1450)
STD_PRODUCTIVITY_ADV = 50 / (2 * 1450)

HIRE_COST = 750
FIRE_COST = 450
PRUNE_LENGTH = 8

PRUNE_PROFIT = 3
WORKER_AVAILABILITY_BEG = 20
WORKER_AVAILABILITY_INT = 15
WORKER_AVAILABILITY_ADV = 12

QUALITY_BEG = 506.88
QUALITY_INT = 790.76
QUALITY_ADV = 1074.10

action_size = ((MAX_ALLOWED_WORKER + 1) * (MAX_ALLOWED_WORKER + 1)) ** 3

# Reward constant
M = 10

#Used for the Utility function
rho = -3

# Economic, Environmental, and Social Coefficients
lambda_econ = 0.25
lambda_env = 0.25
lambda_social = 0.5

# number of training time steps
time_steps = 400000

beam_size = 7

state_to_save = 0


best_env = M + 1
best_soc = M + 1
best_econ = M + 1
# min(hired, current workers)
worst_env = -M
worst_soc = -M
worst_econ = -M