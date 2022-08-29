
MAX_ALLOWED_WORKER = 10
BUDGET = 200
PLANTS = 50
WAGE = 15
PRODUCTIVITY = 5
HIRE_COST = 20
FIRE_COST = 25
PRUNE_LENGTH = 10
PRUNE_PROFIT = 3
WORKER_AVAILABILITY = 2
QUALITY = 10
rho = 0.1
time_steps = 20000

action_size = (MAX_ALLOWED_WORKER + 1) * (MAX_ALLOWED_WORKER + 1)
input_shape = 3
alpha = 0.5
beta = 0.5
delta = 0.4
M = 5

lambda_cost = 0.4
lambda_env = 0.1
lambda_social = 0.5