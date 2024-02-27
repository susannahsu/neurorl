SKIP_RELOCATED = True # skip area RELOCATED
MAX_STACKS = 1 # maximum number of stacks allowed
MAX_BLOCKS = 7 # maximum number of blocks allowed in each stack
MAX_STEPS = 50 # maximum number of actions allowed in each episode
BASE_BLOCK_REWARD = 3 # base reward for getting 1 block correct, cumulates with more blocks correct
BLOCK_REWARD_DECAY_FACTOR = 1.2 # discount factor for subsequently correct blocks, the larger the faster the decay
ACTION_COST = 1e-3 # cost for performing any action

# the current difficulty level to focus on in curriculum training 
CURRICULUM = 2 # use None for fixed curriculum, 2 for dynamic curriculum
EPISODE_REWARD_THRESHOLD = 1.4 # threshold to proceed to next curriculum level