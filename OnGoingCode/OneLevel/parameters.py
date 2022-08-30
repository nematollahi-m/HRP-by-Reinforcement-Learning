
MAX_ALLOWED_WORKER = 20
BUDGET = 200
PLANTS = 20
WAGE = 15.65 * 8 * 5 / 1000
PRODUCTIVITY = 550 / (2 * 1450)
HIRE_COST = 750 / 1000
FIRE_COST = 450 / 1000
PRUNE_LENGTH = 8
PRUNE_PROFIT = 3
WORKER_AVAILABILITY = 30
QUALITY = 10

rho = 0.1
time_steps = 1000000

action_size = (MAX_ALLOWED_WORKER + 1) * (MAX_ALLOWED_WORKER + 1)
input_shape = 3

alpha_cost = 0.5
beta_cost = 0.5

alpha_env = 0.1
beta_env = 0.9

alpha_social = 0.2
beta_social = 0.6
delta_social = 0.2

M = 100

lambda_cost = 0.4
lambda_env = 0.1
lambda_social = 0.5




