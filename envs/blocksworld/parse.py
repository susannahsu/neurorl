'''
learning parse 1 stack of blocks into the brain
'''
import numpy as np
import random

from envs.blocksworld.cfg import configurations
from envs.blocksworld import utils

import dm_env
from dm_env import test_utils
from absl.testing import absltest
from acme import types, specs
from typing import Any, Dict, Optional
import tree

cfg = configurations['parse']

class Simulator():
	# brain only represent 1 stack
	def __init__(self, 
				max_blocks = cfg['max_blocks'], # max number of blocks to parse
				max_steps = cfg['max_steps'],
				action_cost = cfg['action_cost'],
				reward_decay_factor = cfg['reward_decay_factor'],
				episode_max_reward = cfg['episode_max_reward'],
				skip_relocated = cfg['skip_relocated'],
				area_status = cfg['area_status'],
				verbose=False):
		self.all_areas, self.head, self.relocated_area, self.blocks_area = utils.init_simulator_areas()
		self.max_blocks = max_blocks
		self.max_steps = max_steps # max steps allowed in episode
		self.action_cost = action_cost
		self.reward_decay_factor = reward_decay_factor
		self.episode_max_reward = episode_max_reward
		self.skip_relocated = skip_relocated
		self.verbose = verbose
		self.area_status = area_status # area attributes to encode in state, default ['last_activated', 'num_block_assemblies', 'num_total_assemblies']
		self.action_dict = self.create_action_dictionary() 
		self.num_actions = len(self.action_dict)
		
		self.num_blocks, self.goal = self.create_episode() 

		self.correct_record = np.zeros_like(self.goal) # binary record for how many blocks are correct
		self.unit_reward = utils.calculate_unit_reward(self.reward_decay_factor, len(self.goal), self.episode_max_reward)
		self.state, self.action_to_statechange, self.area_to_stateidx, self.stateidx_to_fibername, self.assembly_dict, self.last_active_assembly = self.create_state_representation()
		self.num_fibers = len(self.stateidx_to_fibername.keys())
		self.num_areas = len(self.area_to_stateidx.keys())
		self.num_assemblies = self.max_blocks # total number of assemblies ever created
		self.just_projected = False # record if the previous action was project
		self.current_time = 0 # current step in the episode
		self.all_correct = False # if the most recent readout has everything correct
		
	def create_episode(self, shuffle=False, difficulty_mode=False, cur_curriculum_level=None):
		goal = [None] * self.max_blocks # dummy goal template, to be filled
		num_blocks = None # actual number of blocks in the stack, to be modified
		if difficulty_mode=='curriculum':
			assert cur_curriculum_level!=None, f"requested curriculum but current level is not given"
			num_blocks = self.sample_from_curriculum(cur_curriculum_level)
		elif difficulty_mode=='uniform' or (type(difficulty_mode)==int and difficulty_mode==0): # uniform mode
			num_blocks = random.randint(1, self.max_blocks)
		elif type(difficulty_mode)==bool and (difficulty_mode==False): # default max blocks to parse
			num_blocks = self.max_blocks
		elif type(difficulty_mode)==int:
			assert 1<=difficulty_mode<=self.max_blocks, \
				f"invalid difficulty mode: {difficulty_mode}, should be 'curriculum', or 0, or values in [1, {self.max_blocks}]"
			num_blocks = difficulty_mode
		else:
			raise ValueError(f"unrecognized difficulty mode {difficulty_mode} (type {type(difficulty_mode)})")
		assert num_blocks <= self.max_blocks, \
			f"number of actual blocks to parse {num_blocks} should be smaller than max_blocks {self.max_blocks}"
		stack = list(range(num_blocks)) # the actual blocks in the stack, to be filled
		if shuffle:
			random.shuffle(stack)
		goal[:num_blocks] = stack
		return num_blocks, goal

	def sample_from_curriculum(self, cur_curriculum_level):
		assert 1 <= curriculum <= self.max_blocks
		level = curriculum
		return level

	def reset(self, shuffle=False, difficulty_mode=False, cur_curriculum_level=None):
		'''
		Reset environment for new episode.
		Return:
			state: (numpy array with float32)
			info: (any=None)
		'''
		self.num_blocks, self.goal = self.create_episode(shuffle=shuffle, difficulty_mode=difficulty_mode, cur_curriculum_level=cur_curriculum_level)
		self.unit_reward = utils.calculate_unit_reward(self.reward_decay_factor, len(self.goal), self.episode_max_reward)
		self.state, self.action_to_statechange, self.area_to_stateidx, self.stateidx_to_fibername, self.assembly_dict, self.last_active_assembly = self.create_state_representation()
		self.all_correct = False
		self.correct_record = np.zeros_like(self.goal)
		self.just_projected = False
		self.num_assemblies = self.max_blocks
		self.current_time = 0
		info = None
		return self.state.copy(), info

	def close(self):
		'''
		Close and clear the environment.
		Return nothing.
		'''
		del self.state
		del self.action_to_statechange, self.area_to_stateidx, self.stateidx_to_fibername, self.assembly_dict, self.last_active_assembly
		del self.correct_record
		del self.all_correct, self.just_projected
		del self.current_time, self.num_assemblies
		return 



	def sample_expert_demo(self):
		'''
		Return:
			expert_demo: (list of int)
				list of expert actions to solve the puzzle.
		'''
		stack = self.goal[:self.num_blocks]
		actions = []
		action_module0 = [10,0] # optimal fiber actions for parsing first block
		action_module1 = [11,1,6,2] # ... second block
		action_module2 = [7,3,12,4] # ... third block
		action_module3 = [13,5,0,8] # ... fourth block
		action_module4 = [1,9,2,6] # ... fifth block
		inhibit_action_module0 = [11, 1] # close fibers to terminate episode after module0
		inhibit_action_module1 = [7, 3] 
		inhibit_action_module2 = [13, 5]
		inhibit_action_module3 = [1, 9]
		inhibit_action_module4 = [3, 7]
		project_star = [18]
		activate_next_block = [19]
		activate_prev_block = [20]
		activated_block = -1 # currently activated block id in BLOCKS area
		if len(stack)>0: # module 0
			tmp_actions = []
			tmp_actions += utils.go_activate_block(activated_block, stack[0], activate_next_block, activate_prev_block)
			tmp_actions += action_module0
			random.shuffle(tmp_actions)
			actions += tmp_actions
			actions += project_star
			activated_block = stack.pop(0)
		if len(stack)>0: # module 1
			tmp_actions = []
			tmp_actions += utils.go_activate_block(activated_block, stack[0], activate_next_block, activate_prev_block)
			tmp_actions += action_module1
			random.shuffle(tmp_actions)
			actions += tmp_actions
			actions += project_star
			activated_block = stack.pop(0)
		else: # no more blocks after module0
			tmp_actions = inhibit_action_module0
			random.shuffle(tmp_actions)
			actions += tmp_actions
			return actions
		if len(stack)==0: # no more blocks after module1
			tmp_actions = inhibit_action_module1
			random.shuffle(tmp_actions)
			actions += tmp_actions
			return actions
		imodule = 0
		while True: # loop through module 2,3,4
			tmp_actions = []
			tmp_actions += utils.go_activate_block(activated_block, stack[0], activate_next_block, activate_prev_block)
			if imodule%3 == 0:
				actions += action_module2
			elif imodule%3 == 1:
				actions += action_module3
			elif imodule%3 == 2:
				actions += action_module4
			random.shuffle(tmp_actions)
			actions += tmp_actions
			actions += project_star
			activated_block = stack.pop(0)
			if len(stack)==0:
				tmp_actions = [inhibit_action_module2, inhibit_action_module3, inhibit_action_module4][imodule%3]
				random.shuffle(tmp_actions)
				actions += tmp_actions
				return actions
			imodule += 1
		
		

	def step(self, action_idx):
		'''
		Return: 
			state: (numpy array with float32)
			reward: (float)
			terminated: (boolean)
			truncated: (boolean)
			info: (any=None)
		'''
		action_tuple = self.action_dict[int(action_idx)] # (action name, *kwargs)
		action_name = action_tuple[0]
		state_change_tuple = self.action_to_statechange[int(action_idx)] # (state index, new state value)
		stateidx_to_fibername = self.stateidx_to_fibername # {state vec idx: (area1, area2)} 
		area_to_stateidx = self.area_to_stateidx # {area_name: state vec starting idx}
		reward = -self.action_cost # default cost for performing any action
		terminated = False # whether the episode ended
		truncated = False # end due to max steps
		info = None
		if (action_name == "disinhibit_fiber") or (action_name == "inhibit_fiber"):
			area1, area2 = action_tuple[1], action_tuple[2]
			if self.state[state_change_tuple[0]] == state_change_tuple[1]: # BAD, fiber is already disinhibited/inhibited
				reward -= self.action_cost
			self.state[state_change_tuple[0]] = state_change_tuple[1] # update state
			self.just_projected = False
		elif action_name == "project_star": # state_change_tuple = ([],[]) 
			if [*self.last_active_assembly.values()].count(-1)==len(self.last_active_assembly): # BAD, no active assembly exists
				reward -= self.action_cost 
			elif self.just_projected: # BAD, consecutive project
				reward -= self.action_cost
			else: # GOOD, valid project
				self.assembly_dict, self.last_active_assembly, self.num_assemblies = utils.synthetic_project(self.state, self.assembly_dict, self.stateidx_to_fibername, self.last_active_assembly, self.num_assemblies, self.verbose, blocks_area=self.blocks_area)
				for area_name in self.last_active_assembly.keys():  # update state for each area
					if (self.skip_relocated and area_name==self.relocated_area) or (area_name==self.blocks_area):
						continue # only node and head areas need to update
					# update last active assembly in state 
					assert self.area_status[0] == 'last_activated', f"idx 0 in self.area_status {self.area_status} should be last_activated"
					self.state[area_to_stateidx[area_name][self.area_status[0]]] = self.last_active_assembly[area_name] 
					# update the number of self.blocks_area related assemblies in this area
					assert self.area_status[1] == 'num_block_assemblies', f"idx 1 in self.area_status {self.area_status} should be num_block_assemblies"
					count = 0 
					for assembly_info in self.assembly_dict[area_name]:
						connected_areas, connected_assemblies = assembly_info[0], assembly_info[1]
						if self.blocks_area in connected_areas:
							count += 1
					self.state[area_to_stateidx[area_name][self.area_status[1]]] = count
					# update the number of total assemblies in this area
					assert self.area_status[2] == 'num_total_assemblies', f"idx 2 in self.area_status {self.area_status} should be num_total_assemblies"
					self.state[area_to_stateidx[area_name][self.area_status[2]]] = len(self.assembly_dict[area_name])
				# readout stack	and compute reward
				readout = utils.synthetic_readout(self.assembly_dict, self.last_active_assembly, self.head, len(self.goal), self.blocks_area)
				r, self.all_correct, self.correct_record = utils.calculate_readout_reward(readout, self.goal, self.correct_record, self.unit_reward, self.reward_decay_factor)
				reward += r 
			self.just_projected = True
		elif action_name == "activate_block":
			bidx = int(self.state[state_change_tuple[0]]) # currently activated block id
			newbidx = int(bidx) + state_change_tuple[1] # the new block id to be activated (prev -1 or next +1)
			if newbidx < 0 or newbidx >= self.max_blocks: # BAD, new block id is out of range
				reward -= self.action_cost
			else: # GOOD, valid activate
				self.state[state_change_tuple[0]] = newbidx # update block id in state vec
				self.last_active_assembly[self.blocks_area] = newbidx # update the last active assembly
			self.just_projected = False
		else:
			raise ValueError(f"\tError: action_idx {action_idx} is not recognized!")
		self.current_time += 1 # increment step in the episode 
		if self.current_time >= self.max_steps:
			truncated = True
		terminated = self.all_correct and utils.all_fiber_closed(self.state, self.stateidx_to_fibername)
		return self.state.copy(), reward, terminated, truncated, info


		
	def create_state_representation(self):
		'''
		Initialize the episode state in the environmenmt. 
		Return:
			state: (numpy array with float32)
				state representation
				[goal stack,
				fiber inhibition status,
				last activated assembly idx in the area, 
				number of blocks-connected assemblies in each area,
				number of all assemblies in each area]
			action_to_statechange: (dict)
				map action index to change in state vector
				{action_idx: ([state vector indices needed to be modified], [new values in these state indices])}
			area_to_stateidx: (dict)
				map area name to indices in state vector
				{area: [corresponding indices in state vector]}
			stateidx_to_fibername: (dict)
				mapping state vector index to fiber between two areas
				{state idx: (area1, area2)}
			assembly_dict: (dict)
				dictionary storing assembly associations that currently exist in the brain
				{area: [assembly_idx0[source areas[A0, A1], source assembly_idx[a0, a1]], 
						assembly_idx1[[A3], [a3]], 
						assembly_idx2[[A4], [a4]], 
						...]}
				i.e. area has assembly_idx0, which is associated/projected from area A0 assembly a0, and area A1 assembly a1
			last_active_assembly: (dict)
				dictionary storing the latest activated assembly idx in each area
				{area: assembly_idx}
				assembly_idx = -1 means that no previously activated assembly exists
		'''
		state_vec = [] # state vector
		action_to_statechange = {} # action -> state change, {action_idx: ([state vector indices needed to be modified], [new values in these state indices])}
		action_idx = 0 # action index, the order should match that in self.action_dict
		state_vector_idx = 0 # initialize the idx in state vec to be changed
		area_to_stateidx = {} # dict of index of each area
		stateidx_to_fibername = {} # mapping of state vec index to fiber name
		assembly_dict = {} # {area: [assembly1 associations...]}
		last_active_assembly = {} # {area: assembly_idx}
		# encode goal stack
		for b in self.goal:
			if b==None:  # filler for empty block
				state_vec.append(-1)
			else:
				state_vec.append(b)
			state_vector_idx += 1 # increment state index
		# encode fiber inhibition status
		area_pairs = [] # record pairs of areas already visited
		for area1 in self.all_areas:
			if self.skip_relocated and area1==self.relocated_area:
				continue
			last_active_assembly[area1] = -1 # initialize area with no activated assembly
			assembly_dict[area1] = [] # will become {area: [a_id 0[source a_name[A1, A2], source a_id [a1, a2]], 1[[A3], [a3]], 2[(A4, a4)], ...]}
			for area2 in self.all_areas:
				if (self.skip_relocated and area2==self.relocated_area) or (area1==area2) or ([area1, area2] in area_pairs): 
					continue # skip relocated area, self connection, already encoded area pairs
				# if (("_H" in area1) and ("_N0" not in area2)) or (("_H" in area2) and ("_N0" not in area1)):
				if (("_H" in area1) and (area2==self.blocks_area)) or (("_H" in area2) and (area1==self.blocks_area)):
					continue # for HEADS fibers, only consider N0, N1, N2, do not connect to blocks
				# add fiber status to state vector
				state_vec.append(0) # fiber should be locked upon initialization
				stateidx_to_fibername[state_vector_idx] = (area1, area2) # record state idx -> fiber name
				# action -> state change: disinhibit fiber
				assert self.action_dict[action_idx][0]=="disinhibit_fiber" and self.action_dict[action_idx][1]==area1 and self.action_dict[action_idx][2]==area2, \
						f"action_index {action_idx} should have (disinhibit_fiber, {area1}, {area2}), but action_dict has {self.action_dict}"
				action_to_statechange[action_idx] = ([state_vector_idx], 1) # fiber open
				action_idx += 1
				# action -> state change: inhibit fiber
				assert self.action_dict[action_idx][0]=="inhibit_fiber" and self.action_dict[action_idx][1]==area1 and self.action_dict[action_idx][2]==area2, \
						f"action_index {action_idx} should have (inhibit_fiber, {area1}, {area2}), but action_dict has {self.action_dict}"
				action_to_statechange[action_idx] = ([state_vector_idx], 0) # fiber close
				action_idx += 1
				# increment state idx
				state_vector_idx += 1
				# update visited area_pairs
				area_pairs.append([area1, area2])
				area_pairs.append([area2, area1])
		# action -> state change: strong project (i.e. project star)
		assert self.action_dict[action_idx][0]=="project_star", \
			f"action_index {action_idx} should have (project_star, None), but action_dict has {self.action_dict}"
		action_to_statechange[action_idx] = ([],[]) # no pre-specified new state values, things will be updated after project
		action_idx += 1
		# encode area status, no need to encode area inhibition status since assuming action bundle
		for istatus, status_name in enumerate(self.area_status): 
			for area_name in self.all_areas: 
				if self.skip_relocated and area_name==self.relocated_area:
					continue
				if istatus==0: # encode last activated assembly index in this area
					area_to_stateidx[area_name] = {status_name: state_vector_idx} # area -> state idx
					state_vec.append(-1) # initialize most recent assembly as none 
				elif istatus==1: # encode number of blocks-connected assemblies in this area
					area_to_stateidx[area_name][status_name] = state_vector_idx # area -> state idx
					if area_name==self.blocks_area:
						state_vec.append(self.max_blocks)
					else:
						state_vec.append(0)
				elif istatus==2: # encode number of total assemblies in this area
					area_to_stateidx[area_name][status_name] = state_vector_idx # area -> state idx
					if area_name==self.blocks_area:
						state_vec.append(self.max_blocks)
					else:
						state_vec.append(0)
				else:
					raise ValueError(f"there are only {len(self.area_status)} status for each area in state, but requesting {istatus}")
				state_vector_idx += 1 # increment state vector idx
		# action -> state change: activate next block assembly
		assert self.action_dict[action_idx][0]=="activate_block" and self.action_dict[action_idx][1]=="next", \
			f"action_index {action_idx} should have (activate_block, next), but action_dict has {self.action_dict}"
		action_to_statechange[action_idx] = (area_to_stateidx[self.blocks_area][self.area_status[0]], +1) 
		action_idx += 1
		assert self.action_dict[action_idx][0]=="activate_block" and self.action_dict[action_idx][1]=="previous", \
			f"action_index {action_idx} should have (activate_block, previous), but action_dict has {self.action_dict}"
		# action -> state change: activate previous block assembly
		action_to_statechange[action_idx] = (area_to_stateidx[self.blocks_area][self.area_status[0]], -1) 
		# initialize assembly dict for blocks area, other areas will be updated during project
		assembly_dict[self.blocks_area] = [[[],[]] for _ in range(self.max_blocks)] 
		return np.array(state_vec, dtype=np.float32), \
				action_to_statechange, \
				area_to_stateidx, \
				stateidx_to_fibername, \
				assembly_dict, \
				last_active_assembly


	def create_action_dictionary(self):
		'''
		Create action dictionary: a dict that contains mapping of action index to action name
		Assuming action bundle: disinhibit_fiber entails disinhibiting the two areas and the fiber, opening the fibers in both directions
		Return:
			dictonary: (dict) 
				{action_idx : (action name, *kwargs)}
		'''
		idx = 0 # action idx
		dictionary = {} # action idx --> (action name, *kwargs)
		area_pairs = [] # store pairs of areas already visited
		# disinhibit and inhibit fibers
		for area1 in self.all_areas:
			if self.skip_relocated and area1==self.relocated_area:
				continue
			for area2 in self.all_areas:
				if (self.skip_relocated and area2==self.relocated_area) or (area1==area2) or ([area1, area2] in area_pairs):
					continue  # skip relocated area, self connection, already encoded area pairs
				if (("_H" in area1) and (area2==self.blocks_area)) or (("_H" in area2) and (area1==self.blocks_area)):
					continue # HEADS will connect with N0, N1, N2, but not BLOCKS
				dictionary[idx] = ("disinhibit_fiber", area1, area2)
				idx += 1
				dictionary[idx] = ("inhibit_fiber", area1, area2)
				idx += 1
				# update area_pairs
				area_pairs.append([area1, area2])
				area_pairs.append([area2, area1])
		# project star
		dictionary[idx] = ("project_star", None)
		idx += 1
		# activate block
		dictionary[idx] = ("activate_block", "next")
		idx += 1
		dictionary[idx] = ("activate_block", "previous")
		return dictionary



def test_simulator(max_blocks=7, expert=True, repeat=10, verbose=False):
	import time
	sim = Simulator(max_blocks=max_blocks, verbose=verbose)
	print(sim.action_dict)
	start_time = time.time()
	print(f'initial state for new simulator: {sim.state}')
	for difficulty in range(max_blocks+1):
		for _ in range(repeat):
			print(f'\n\n------------ repeat {repeat}, state after reset\t{sim.reset(shuffle=True, difficulty_mode=difficulty)[0]}')
			expert_demo = sim.sample_expert_demo() if expert else None
			rtotal = 0 # total reward of episode
			nsteps = sim.max_steps if (not expert) else len(expert_demo)
			for t in range(nsteps):
				action_idx = random.choice(list(range(sim.num_actions))) if (not expert) else expert_demo[t]
				next_state, reward, terminated, truncated, info = sim.step(action_idx)
				rtotal += reward
				print(f't={t},\tr={round(reward, 5)},\taction={action_idx}\t{sim.action_dict[action_idx]},\ttruncated={truncated},\tdone={terminated},\n\tjust_projected={sim.just_projected}, all_correct={sim.all_correct}, correct_record={sim.correct_record}')
				print(f'\tnext state {next_state}\t')
			readout = utils.synthetic_readout(sim.assembly_dict, sim.last_active_assembly, sim.head, len(sim.goal), sim.blocks_area)
			print(f'end of episode (difficulty={difficulty}), num_blocks={sim.num_blocks}, synthetic readout {readout}, goal {sim.goal}, total reward={rtotal}, time lapse={time.time()-start_time}')
			if expert:
				assert readout == sim.goal, f"readout {readout} and goal {sim.goal} should be the same"
				assert terminated, "episode should be done"
				assert np.isclose(rtotal, sim.episode_max_reward-sim.action_cost*nsteps), \
						f"rtotal {rtotal} and theoretical total {sim.episode_max_reward-sim.action_cost*nsteps} should be roughly the same"





class EnvWrapper(dm_env.Environment):
	'''
	Wraps a Simulator object to be compatible with dm_env.Environment
	Reference: 
		https://github.com/wcarvalho/human-sf/blob/da0c65d04be708199ffe48d5f5118b295bfd43a3/lib/dm_env_wrappers.py#L15
		https://github.com/google-deepmind/dm_env/
		https://github.com/google-deepmind/acme/
	'''
	def __init__(self, environment: Simulator, shuffle=True, difficulty_mode='uniform', cur_curriculum_level=None):
		self._environment = environment
		self._reset_next_step = True
		self._last_info = None
		obs_space = self._environment.state
		act_space = self._environment.num_actions-1 # maximum action index
		self._observation_spec = _convert_to_spec(obs_space, name='observation')
		self._action_spec = _convert_to_spec(act_space, name='action')
		self.shuffle = shuffle # whether to shuffle blocks for each episode
		self.difficulty_mode = difficulty_mode
		self.cur_curriculum_level = cur_curriculum_level

	def reset(self) -> dm_env.TimeStep:
		self._reset_next_step = False
		observation, info = self._environment.reset(shuffle=self.shuffle, difficulty_mode=self.difficulty_mode, cur_curriculum_level=self.cur_curriculum_level)
		self._last_info = info
		return dm_env.restart(observation)
	
	def step(self, action: types.NestedArray) -> dm_env.TimeStep:
		if self._reset_next_step:
			return self.reset()
		observation, reward, done, truncated, info = self._environment.step(action)
		self._reset_next_step = done or truncated
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
	if isinstance(space, int): # scalar int for max idx of an action
		dtype = type(space)
		min_val = 0 # minimum action index (inclusive)
		max_val = space # maximum action index (inclusive)
		try:
			assert name=='action'
		except:
			raise ValueError('Converting integer to dm_env spec, but name is not action')
		return specs.DiscreteArray(
			num_values=max_val+1,
			name=name
		)
	elif isinstance(space, np.ndarray): # observation
		min_val, max_val = space.min(), cfg['max_assemblies']
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
	

class Test(test_utils.EnvironmentTestMixin, absltest.TestCase):
	def make_object_under_test(self):
		rng = np.random.default_rng(1)
		sim = Simulator(max_blocks=7)
		return EnvWrapper(sim, rng)
	def make_action_sequence(self):
		for _ in range(200):
			yield self.make_action()

if __name__ == "__main__":

	# random.seed(1)
	test_simulator(max_blocks=7, expert=True, repeat=100, verbose=False)
	
	absltest.main()

