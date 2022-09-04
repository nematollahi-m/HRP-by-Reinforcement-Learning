
""" model parameters """

MAX_ALLOWED_WORKER = 3
BUDGET = 200
PLANTS = 50

WAGE_BEG = 15
WAGE_INT = 20
WAGE_ADV = 30

PRODUCTIVITY_BEG = 5
PRODUCTIVITY_INT = 7
PRODUCTIVITY_ADV = 10

HIRE_COST = 20
FIRE_COST = 25
PRUNE_LENGTH = 10

PRUNE_PROFIT = 3
WORKER_AVAILABILITY_BEG = 10
WORKER_AVAILABILITY_INT = 8
WORKER_AVAILABILITY_ADV = 7

QUALITY_BEG = 2
QUALITY_INT = 4
QUALITY_ADV = 5

action_size = ((MAX_ALLOWED_WORKER + 1) * (MAX_ALLOWED_WORKER + 1)) ** 3

alpha_economic = 0.5
beta_economic = 0.5

alpha_env = 0.5
beta_env = 0.5

alpha_social = 0.3
beta_social = 0.3
delta_social = 0.4

M = 5
rho = 0.1
lambda_econ = 0.2
lambda_env = 0.4
lambda_social = 0.4

time_steps = 100000


