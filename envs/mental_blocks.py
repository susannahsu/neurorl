'''
use venv: 
	source acme/bin/activate
learn to plan using parse, pop, remove, etc.
suppose we have 3 regions: current stacks, table stacks, goal stacks
'''
import random
import dm_env
from dm_env import test_utils
from absl.testing import absltest
from acme import types, specs
from typing import Any, Dict, Optional
import tree
import numpy as np

SKIP_RELOCATED = True # skip area RELOCATED
MAX_STACKS = 1 # maximum number of stacks allowed
MAX_BLOCKS = 7 # maximum number of blocks allowed in each stack
MAX_STEPS = 50 # maximum number of actions allowed in each episode
BASE_BLOCK_REWARD = 3 # base reward for getting 1 block correct, cumulates with more blocks correct
BLOCK_REWARD_DECAY_FACTOR = 1.2 # discount factor for subsequently correct blocks, the larger the faster the decay
ACTION_COST = 1 # cost for performing any action

class Simulator():
	def __init__(self, input_stacks, goal_stacks, 
			  max_stacks=MAX_STACKS, max_blocks=MAX_BLOCKS, max_steps=MAX_STEPS,
			  base_block_reward=BASE_BLOCK_REWARD, reward_decay_factor=BLOCK_REWARD_DECAY_FACTOR,
			  action_cost=ACTION_COST,
			  verbose=False):
		self.max_stacks = max_stacks
		self.max_blocks = max_blocks
		self.base_block_reward = base_block_reward # base score for getting 1 block correct, cumulates with more blocks correct
		self.reward_decay_factor = reward_decay_factor # discount factor for subsequently correct blocks, the larger the faster the decay
		self.action_cost = action_cost
		self.max_steps = max_steps # maximum number of actions allowed in an episode
		self.verbose = verbose
		self.current_time = 0 # current time step
		self.state = self.__create_state_representation() # current state
		self.state_size = self.state.size # total number of elements in flattened state matrix
		num_valid_blocks = [[1 for _ in goal_stack] for goal_stack in goal_stacks]
		self.num_valid_blocks = sum(sum(l) for l in num_valid_blocks)
		self.max_episode_reward = sum([self.base_block_reward / (self.reward_decay_factor**(intersection+1)) for istack in range(len(goal_stacks)) for jblock in range(len(goal_stacks[istack])) for intersection in range(jblock+1)] 
								+ [-5*action_cost for istack in range(len(goal_stacks)) for _ in range(len(goal_stacks[istack]))]
								+ [-1*action_cost for _ in range(len(goal_stacks)) ] + [-action_cost, -action_cost])
		self.action_dict = self.__create_action_dictionary() 
		self.n_actions = len(self.action_dict)
		# format the input and goal to same length
		self.goal = [[-1 for _ in range(max_blocks)] for _ in range(max_stacks)] # [max_stacks, max_blocks]
		for istack in range(len(goal_stacks)): # a stack [-1, -1, ..., top block, ..., bottom block]
			for jblock in range(1, len(goal_stacks[istack])+1): # traverse backwards
				self.goal[istack][-jblock] = goal_stacks[istack][-jblock] 
		self.input_stacks = [[-1 for _ in range(max_blocks)] for _ in range(max_stacks)] # [max_stacks, max_blocks]
		for istack in range(len(input_stacks)): # a stack [-1, -1, ..., top block, ..., bottom block]
			for jblock in range(1, len(input_stacks[istack])+1): # traverse backwards
				self.input_stacks[istack][-jblock] = input_stacks[istack][-jblock]
	

	def close(self):
		self.current_time = 0
		self.state = None
		return 
	

	def reset(self):
		self.state = self.__create_state_representation()
		self.current_time = 0
		return self.state, None
	
	
	def __parse_input(self):
		for istack in range(self.max_stacks):
			for jblock in range(self.max_blocks):
				self.state[istack*self.max_blocks+jblock, :] = 0 # initialize one-hot encoding
				if self.input_stacks[istack][jblock] != -1: # encode nonempty block
					self.state[istack*self.max_blocks+jblock, self.input_stacks[istack][jblock]] = 1
		self.state[self.max_blocks*(self.max_stacks*2+1):self.max_blocks*(self.max_stacks*2+1)+self.max_stacks, :] = 0 # clear intersection record
		self.state[self.max_blocks*self.max_stacks:self.max_blocks*self.max_stacks+self.max_blocks, :] = 0 # clear table stacks
		self.state[-4, :] = 0 # reset cur pointer 
		self.state[-4, 0] = 1 # point to the first cur stack
		self.state[-3, :] = 0 # reset table pointer
		self.state[-3, 0] = 1 # point to the first table stack
		self.state[-2, :] = 1 # input parsed
	

	def __parse_goal(self):
		for istack in range(self.max_stacks):
			for jblock in range(self.max_blocks):
				self.state[(istack+self.max_stacks+1)*self.max_blocks+jblock, :] = 0 # initialize one-hot encoding
				if self.goal[istack][jblock] != -1: # encode nonempty block
					self.state[(istack+self.max_stacks+1)*self.max_blocks+jblock, self.goal[istack][jblock]] = 1
		self.state[-1, :] = 1 # goal parsed


	def step(self, action_idx):
		action_name = self.action_dict[int(action_idx)] 
		cur_pointer = np.argmax(self.state[-4]) # pointer of current stack idx
		table_pointer = np.argmax(self.state[-3]) # pointer of table stack idx
		input_parsed = np.any(self.state[-2]==1) # True if any 1 exists (whole vector will be 1s if parsed already)
		goal_parsed = np.any(self.state[-1]==1) # True if any 1 exists (whole vector will be 1s if parsed already)
		reward = -self.action_cost # default cost for performing any action
		terminated = False
		truncated = False
		info = None
		if (not input_parsed) or (not goal_parsed):
			if (not input_parsed) and (not goal_parsed) and (action_name != "parse_input") and (action_name != "parse_goal"):
				reward -= self.action_cost*2
				print('\tboth input and goal not parsed yet') if self.verbose else 0
			elif (not input_parsed) and action_name == "parse_input":
				self.__parse_input()
				input_parsed = True
			elif (not goal_parsed) and action_name == "parse_goal":
				self.__parse_goal()
				goal_parsed = True
			else:
				reward -= self.action_cost*2
				print('\teither input or goal not parsed yet') if self.verbose else 0
		elif action_name == "next_stack":
			if cur_pointer + 1 >= self.max_stacks: # new cur pointer idx out of range
				reward -= self.action_cost
				print('\tcur pointer out of range') if self.verbose else 0
			else: # next stack
				self.state[-4, cur_pointer] = 0
				self.state[-4, cur_pointer+1] = 1
		elif action_name == "previous_stack":
			if cur_pointer == 0: # cur pointer is already minimum
				reward -= self.action_cost
				print('\tcur pointer out of range') if self.verbose else 0
			else: # previous stack
				self.state[-4, cur_pointer] = 0
				self.state[-4, cur_pointer-1] = 1
		elif action_name == "next_table":
			if table_pointer + 1 >= self.max_blocks: # new table pointer out of range
				reward -= self.action_cost
				print('\ttable pointer out of range') if self.verbose else 0
			else: # next table stack
				self.state[-3, table_pointer] = 0
				self.state[-3, table_pointer+1] = 1
		elif action_name == "previous_table":
			if table_pointer == 0: # table pointer is already minimum
				reward -= self.action_cost
				print('\ttable pointer out of range') if self.verbose else 0
			else: # previous table stack
				self.state[-3, table_pointer] = 0
				self.state[-3, table_pointer-1] = 1
		elif action_name == "remove":
			if np.all(self.state[cur_pointer*self.max_blocks + self.max_blocks-1]==0): # nothing to remove, cur stack is empty
				reward -= self.action_cost
				print('\tnothing to remove, cur stack empty') if self.verbose else 0
			else: # pop the top block from cur stack
				pop_idx = 0
				block_id = -1
				for i in range(cur_pointer*self.max_blocks, cur_pointer*self.max_blocks+self.max_blocks): # from top block to bottom
					if np.any(self.state[i]==1): # find the first nonempty block in stack 
						pop_idx = i
						block_id = np.argmax(self.state[i])
						break
				self.state[pop_idx, block_id] = 0 # remove the block
				for i in range(self.max_blocks * self.max_stacks, self.max_blocks * self.max_stacks + self.max_blocks): 
					if np.all(self.state[i]==0): # find the first empty table stack
						self.state[i, block_id] = 1 # put the removed block to the tabel
						break
				print('\tremove top block', block_id) if self.verbose else 0
				r, terminated = self.__readout_reward()
				reward += r
		elif action_name == "add":
			if np.all(self.state[self.max_blocks*self.max_stacks+table_pointer]==0): # nothing to add, table stack is empty
				reward -= self.action_cost
				print('\tnothing to add, table stack empty') if self.verbose else 0
			elif np.any(self.state[cur_pointer*self.max_blocks]==1): # intent to add to cur stack, but stack full
				print('\tintend to add to full stack, last block in stack is',self.state[cur_pointer+self.max_blocks-1]) if self.verbose else 0
				reward -= self.action_cost
			else: # add the block to cur stack
				new_block = np.argmax(self.state[self.max_blocks*self.max_stacks+table_pointer]) # the new blockid to be added
				self.state[self.max_blocks*self.max_stacks+table_pointer, new_block] = 0 # remove the block from table
				for i in range(cur_pointer*self.max_blocks+self.max_blocks-1, cur_pointer*self.max_blocks-1, -1): # from bottom to top
					if np.all(self.state[i]==0): # the first empty block on top
						self.state[i, new_block] = 1 # add the block to the top of cur stack
						break
				r, terminated = self.__readout_reward()
				reward += r 
		elif action_name == "parse_input": # parse input repetitively
			assert input_parsed
			self.__parse_input()
			reward -= self.action_cost
			print('\tinput parsed again, reset') if self.verbose else 0
		elif action_name == "parse_goal": # parse goal repetitively
			assert goal_parsed
			reward -= self.action_cost
			print('\tgoal parsed again') if self.verbose else 0
		self.current_time += 1
		if self.current_time >= self.max_steps:
			truncated = True
		return self.state.copy(), reward, terminated, truncated, info


	def __intersection(self, curstack, goalstack):
		'''
		return the number of blocks that match in curstack and goalstack (starting from the bottom)
			and the number of blocks that need to be removed from curstack (i.e. the non matching blocks)
		'''
		intersection, num_to_remove = 0, 0
		for i in range(self.max_blocks-1, -1, -1): # iterate from bottom to top
			if goalstack[i] != -1 and (curstack[i]==goalstack[i]):
				intersection += 1
			else: # first nonmatching block
				break
		for j in range(self.max_blocks-1, -1, -1):
			if curstack[j]==-1: # find the idx of first empty block
				break
		num_to_remove = i-j # number of blocks to be removed from curstack
		return intersection, num_to_remove 
	

	def __decode_curstack(self):
		cur_stacks = []
		for istack in range(self.max_stacks):
			stack = []
			for jblock in range(self.max_blocks):
				if np.any(self.state[istack*self.max_blocks+jblock]==1):
					stack.append(np.argmax(self.state[istack*self.max_blocks+jblock]).item())
				else:
					stack.append(-1)
			cur_stacks.append(stack)
		return cur_stacks


	def __readout_reward(self):
		cur_stacks = self.__decode_curstack()
		print('cur_stacks', cur_stacks) if self.verbose else 0
		intersection_record = self.state[self.max_blocks*(self.max_stacks*2+1):self.max_blocks*(self.max_stacks*2+1)+self.max_stacks, :].tolist()
		base_block_reward = self.base_block_reward
		reward_decay_factor = self.reward_decay_factor
		score = 0
		intersections = [0] * self.max_stacks # temporary intersection for the current readout
		for istack in range(self.max_stacks): # iterate each stack
			intersections[istack], _ = self.__intersection(cur_stacks[istack], self.goal[istack]) # number of blocks that match 
			print('intersections', intersections, 'intersection_record', intersection_record) if self.verbose else 0
			if intersections[istack]!=0 and (intersection_record[istack][intersections[istack]-1]==0): # new match
				score += sum([base_block_reward / (reward_decay_factor**(d+1)) for d in range(intersections[istack])])
				intersection_record[istack][intersections[istack]-1] = 1
			elif intersections[istack] != 0: # already matched, give smaller reward
				score += 0
		all_correct = sum(intersections) == self.num_valid_blocks # all blocks are matched
		self.state[self.max_blocks*(self.max_stacks*2+1):self.max_blocks*(self.max_stacks*2+1)+self.max_stacks,:] = np.array(intersection_record)
		return score, all_correct
	
		
	def __create_state_representation(self):
		'''
		Create initial state vector (2D)
			cur stack 1 top block (one-hot encoding)
			cur stack 1 second block
			...
			cur stack 1 bottom block
			...
			...
			cur max_stacks top block
			...
			cur max_stacks bottom block
			
			table stack 1 (one block only) top block (one-hot encoding)
			table stack 2 (one block only) top block (one-hot encoding)
			...
			table stack max_blocks (one block only) top block (one-hot encoding)
			
			goal stack 1 top block (one-hot encoding)
			goal stack 1 second block
			...
			goal stack 1 bottom block
			...
			...
			goal max_stacks top block
			...
			goal max_stacks bottom block
			
			intersection record for stack 1
			...
			intersection record for stack max_stacks
			
			cur stack pointer (one-hot encoding, 0 to max_stacks-1)
			table pointer (one-hot encoding, 0 to max_blocks-1)
			
			input parsed (all 0 if False, all 1 if parsed)
			goal parsed (all 0 if False, all 1 if parsed)
		'''
		self.state = np.zeros((self.max_blocks*self.max_stacks \
									+ self.max_blocks \
									+ self.max_blocks*self.max_stacks \
									+ self.max_stacks \
									+ 4, 
								self.max_blocks))
		self.state[-4, 0] = 1 # initialize the cur pointer to be the first stack (one-hot encoding)
		self.state[-3, 0] = 1 # initialize the table pointer to be the first table stack (one-hot encoding)
		return self.state


	def __create_action_dictionary(self):
		'''
		Create action dictionary: a dict that contains mapping of action index to action name
		'''
		idx = 0 # action idx
		dictionary = {} # action idx --> action name
		dictionary[idx] = "next_stack"
		idx += 1
		dictionary[idx] = "previous_stack"
		idx += 1
		dictionary[idx] = "next_table"
		idx += 1
		dictionary[idx] = "previous_table"
		idx += 1
		dictionary[idx] = "remove"
		idx += 1
		dictionary[idx] = "add"
		idx += 1
		dictionary[idx] = "parse_input"
		idx += 1
		dictionary[idx] = "parse_goal"
		idx += 1
		return dictionary


	def expert_demo(self):
		'''
		use the most naive algorithm to create expert demo: remove all blocks to table, and reassemble them in order
		'''
		final_actions = []
		# first need to parse input and goal
		action = "parse_input"
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = "parse_goal"
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		# remove everything from cur stacks
		table = [] # a cache storing blocks on the table
		cur_pointer = 0 # current pointer to cur stacks
		table_pointer = 0 # current pointer to table stacks
		for istack in range(self.max_stacks):
			for jblock in range(self.max_blocks): # iterating from top to bottom
				if self.input_stacks[istack][jblock]== -1: # have not reached valid block yet
					continue
				action = "remove"
				final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
				table.append(self.input_stacks[istack][jblock]) # record the blocks put on table
			if self.max_stacks>1 and istack!=self.max_stacks-1: # move to next stack if applicable
				action = "next_stack"
				final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
				cur_pointer += 1
		# add blocks according to goal
		for istack in range(self.max_stacks):
			for jblock in range(self.max_blocks-1, -1, -1): # iterating from bottom to top in each stack
				if self.goal[istack][jblock] == -1: # no more blocks in this stack
					break # go to next goal stack
				blocktoadd = self.goal[istack][jblock]
				loc = table.index(blocktoadd) # table stack idx containing this block
				print('blocktoadd:{}, table loc:{}'.format(blocktoadd, loc)) if self.verbose else 0
				if loc - table_pointer > 0: # should move to the right to reach the table stack
					for _ in range(loc - table_pointer):
						action = "next_table"
						final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
				elif loc - table_pointer < 0:
					for _ in range(table_pointer - loc):
						action = "previous_table"
						final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
				table_pointer = loc # table pointer is now moved to the table stack to be added
				if istack - cur_pointer > 0: # cur stack pointer need to move to the right
					for _ in range(istack - cur_pointer):
						action = "next_stack"
						final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
				elif istack - cur_pointer < 0: # cur stack pointer need to move to the left
					for _ in range(cur_pointer - istack):
						action = "previous_stack"
						final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
				cur_pointer = istack # cur pointer moved to the stack to be added
				action = "add" # add the block 
				final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		return final_actions # list of integers representing actions


	def test(self, verbose=False, use_expert=False):
		self.verbose=verbose
		self.state = self.__create_state_representation()
		print('initial state\n', self.state)
		total_reward = 0
		if not use_expert:
			for t in range(self.max_steps+10):
				action_idx = random.choice(list(range(0, self.n_actions)))
				# print('current state\n', self.state)
				self.state, reward, done, truncated, info = self.step(action_idx)
				total_reward += reward
				print('t={}, reward={}, action_idx={}, action={}, done={}, next state\n{}'.format(t, reward, action_idx, self.action_dict[action_idx], done, self.state))
		else:
			expert_actions = self.expert_demo()
			for t in range(len(expert_actions)):
				action_idx = expert_actions[t]
				self.state, reward, done, truncated, info = self.step(action_idx)
				# print('current state\n', self.state)
				total_reward += reward
				print('t={}, reward={}, action_idx={}, action={}, done={}, next state\n{}'.format(t, reward, action_idx, self.action_dict[action_idx], done, self.state))
		print('total_reward', total_reward)
			


class EnvWrapper(dm_env.Environment):
	'''
	Wraps a Simulator object to be compatible with dm_env.Environment
	Reference: 
		https://github.com/wcarvalho/human-sf/blob/da0c65d04be708199ffe48d5f5118b295bfd43a3/lib/dm_env_wrappers.py#L15
		https://github.com/google-deepmind/dm_env/
		https://github.com/google-deepmind/acme/
	'''
	def __init__(self, environment: Simulator):
		self._environment = environment
		self._reset_next_step = True
		self._last_info = None
		obs_space = self._environment.state
		act_space = self._environment.n_actions-1 # maximum action index
		self._observation_spec = _convert_to_spec(obs_space, name='observation')
		self._action_spec = _convert_to_spec(act_space, name='action')


	def reset(self) -> dm_env.TimeStep:
		self._reset_next_step = False
		observation, info = self._environment.reset()
		self._last_info = info
		return dm_env.restart(observation)
	

	def step(self, action: types.NestedArray) -> dm_env.TimeStep:
		if self._reset_next_step:
			return self.reset()
		observation, reward, done, truncated, info = self._environment.step(action)
		self._reset_next_step = done or truncated
		self._last_truncated = truncated
		self._last_info = info
		# Convert the type of the reward based on the spec, respecting the scalar or array property.
		reward = tree.map_structure(
			lambda x, t: (  # pylint: disable=g-long-lambda
				t.dtype.type(x)
				if np.isscalar(x) else np.asarray(x, dtype=t.dtype)),
			reward,
			self.reward_spec())
		if truncated:
			return dm_env.truncation(reward, observation)
		if done:
			return dm_env.termination(reward, observation)
		return dm_env.transition(reward, observation)
	

	def observation_spec(self) -> types.NestedSpec:
		return self._observation_spec


	def action_spec(self) -> types.NestedSpec:
		return self._action_spec


	def get_info(self) -> Optional[Dict[str, Any]]:
		return self._last_info
	

	@property
	def environment(self) -> Simulator:
		return self._environment
	

	def __getattr__(self, name: str):
		if name.startswith('__'):
			raise AttributeError('attempted to get missing private attribute {}'.format(name))
		return getattr(self._environment, name)
	

	def close(self):
		self._environment.close()
		




def _convert_to_spec(space: Any,
					name: Optional[str] = None) -> types.NestedSpec:
	"""
	Converts a Python data structure to a dm_env spec or nested structure of specs.
	The function supports scalars, numpy arrays, tuples, and dictionaries.
	Args:
		space: The data item to convert (can be scalar, numpy array, tuple, or dict).
		name: Optional name to apply to the return spec.
	Returns:
		A dm_env spec or nested structure of specs, corresponding to the input item.
	"""
	if isinstance(space, int): # scalar int for number of actions
		dtype = type(space)
		min_val = 0 # minimum action index
		max_val = space # maximum action index
		try:
			assert name=='action'
		except:
			raise ValueError('Converting integer to dm_env spec, but name is not action')
		return specs.BoundedArray(
			shape=(),
			dtype=dtype,
			minimum=min_val,
			maximum=max_val,
			name=name
		)
	elif isinstance(space, np.ndarray):
		min_val, max_val = space.min(), space.max()
		try:
			assert name=='observation'
		except:	
			raise ValueError("Converting np.ndarray to dm_env spec, but name is not 'observation'")
		return specs.BoundedArray(
			shape=space.shape,
			dtype=space.dtype,
			minimum=min_val,
			maximum=max_val,
			name=name
		)
	elif isinstance(space, tuple):
		return tuple(_convert_to_spec(s, name) for s in space)
	elif isinstance(space, dict):
		return {
			key: _convert_to_spec(value, key)
			for key, value in space.items()
		}
	else:
		raise ValueError('Unsupported data type for conversion to dm_env spec: {}'.format(space))
	


def create_random_problem(max_num_blocks=MAX_BLOCKS, max_num_stacks=MAX_STACKS, difficulty=None):
	while True:
		num_blocks = random.choice(list(range(2, max_num_blocks+1))) if difficulty==None else difficulty
		# input stacks
		available_blocks = list(range(num_blocks))
		input_stacks = []
		for _ in range(max_num_stacks):
			if len(available_blocks)==0:
				continue
			num_blocks_istack = random.choice(list(range(len(available_blocks)))) #if max_num_stacks>1 else len(available_blocks)
			curstack = []
			for _ in range(num_blocks_istack):
				curblock = random.choice(available_blocks)
				curstack.append(curblock)
				available_blocks.remove(curblock)
			if curstack!=[]:
				input_stacks.append(curstack)
		if len(available_blocks) != 0:
			random.shuffle(available_blocks)
			if len(input_stacks) == 0:
				input_stacks.append(available_blocks)
			else:
				for ab in available_blocks:
					input_stacks[-1].append(ab)
		# goal stacks
		available_blocks = list(range(num_blocks))
		goal_stacks = []
		for _ in range(max_num_stacks):
			if len(available_blocks)==0:
				continue
			num_blocks_istack = random.choice(list(range(len(available_blocks))))# if max_num_stacks>1 else len(available_blocks)
			curstack = []
			for _ in range(num_blocks_istack):
				curblock = random.choice(available_blocks)
				curstack.append(curblock)
				available_blocks.remove(curblock)
			if curstack!=[]:
				goal_stacks.append(curstack)
		if len(available_blocks) != 0:
			random.shuffle(available_blocks)
			if len(goal_stacks) == 0:
				goal_stacks.append(available_blocks)
			else:
				for ab in available_blocks:
					goal_stacks[-1].append(ab)		
		if input_stacks != goal_stacks: # found valid problem
			break
	return num_blocks, input_stacks, goal_stacks


class Test(test_utils.EnvironmentTestMixin, absltest.TestCase):
	def make_object_under_test(self):
		num_blocks, input_stacks, goal_stacks = create_random_problem(difficulty=7)
		# input_stacks = [[0,1,2]]
		# goal_stacks = [[2,0,1]]
		print('difficulty', num_blocks, '\ninput stacks', input_stacks, '\ngoal stacks', goal_stacks)
		environment = Simulator(input_stacks=input_stacks, goal_stacks=goal_stacks)
		return EnvWrapper(environment)
	def make_action_sequence(self):
		for _ in range(200):
			yield self.make_action()


if __name__ == '__main__':
	# random.seed(3)	
	
	
	# num_blocks, input_stacks, goal_stacks = create_random_problem(difficulty=7)
	# print('difficulty', num_blocks, '\ninput stacks', input_stacks, '\ngoal stacks', goal_stacks)
	# environment = Simulator(input_stacks=input_stacks, goal_stacks=goal_stacks)
	# environment.test(verbose=True, use_expert=True)
	# print('difficulty', num_blocks, '\ninput stacks', input_stacks, '\ngoal stacks', goal_stacks)
	# print('max_episode_reward', environment.max_episode_reward)
	

	absltest.main()