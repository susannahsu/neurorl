skip_relocated = True # skip area RELOCATED
max_stacks = 2 # maximum number of stacks allowed
max_blocks = 7 # maximum number of blocks allowed in each stack
max_steps = 100 # maximum number of actions allowed in each episode
base_block_reward = 3 # base reward for getting 1 block correct, cumulates with more blocks correct
block_reward_decay_factor = 1.2 # discount factor for subsequently correct blocks, the larger the faster the decay
action_cost = 1e-3 # cost for performing any action

# the current difficulty level to focus on in curriculum training 
curriculum = 2 # use None for fixed curriculum, 2 for dynamic curriculum
episode_reward_threshold = 1.4 # current upper episode reward threshold to proceed to next curriculum level
threshold1 = 1.4 # upper episode reward threshold to proceed from curriculum 2 to 3
threshold2 = 2.0  # upper episode reward threshold to proceed from curriculum 3 to 4
threshold3 = 2.3  # upper episode reward threshold to proceed from curriculum 4 to above, and final stabalizing
episode_reward_lowerbound = -1.4 # reasonable lower bound for determining whether to switch or continue final stabalizing
episode_reward_lowerbound_factor = 1.04 # lower bound for episode reward scales with level
up_pressure = 0 # current pressure counter for determining when to increment curriculum
down_pressure = 0 # current pressure counter for determinng when to decrement curriculum
up_pressure_threshold = 1 # pressure threshold for proceeding to the next curriculum
down_pressure_threshold = 1 # pressure threshold for falling back to the previous curriculum