'''
learning parse 1 stack of blocks into the brain
'''
from envs.blocksworld.AC.bw_apps import *
from envs.blocksworld import parse_cfg
from envs.blocksworld import utils

import copy
import pprint
import sys
import itertools

import dm_env
from dm_env import test_utils
from absl.testing import absltest
from acme import types, specs
from typing import Any, Dict, Optional
import tree


class Simulator():
	def __init__(self, 
					max_blocks, # max number of blocks to parse
					max_stacks = parse_cfg.max_stacks, # default 1, brain should only take 1 stack
					max_steps = parse_cfg.max_steps,
					action_cost = parse_cfg.action_cost,
					reward_decay_factor = parse_cfg.reward_decay_factor,
					episode_max_reward = parse_cfg.episode_max_reward,
					verbose=False):
		prefix = "G"
		node_areas = []
		for j in range(max_stacks):
			node_areas_stack_j = []
			for k in range(MAX_NODES_AREAS):
				node_areas_stack_j.append(str(j)+"_N"+str(k))
			node_areas.append(node_areas_stack_j)
		node_areas = node_areas
		head_areas = []
		for j in range(max_stacks):
			head_areas.append(str(j)+"_H")
		regions = []
		for j in range(max_stacks):
			regions_stack_j = node_areas[j] + [head_areas[j]]
			regions.append(regions_stack_j)
		print(f"parse.py Simulator max_stacks={max_stacks}, regions={regions}")
		oa = add_prefix(regions=[item for sublist in regions for item in sublist], prefix=prefix)
		oa = oa + [RELOCATED]
		self.prefix = prefix
		self.other_areas = oa
		self.all_areas = [BLOCKS]
		self.all_areas.extend(self.other_areas) 
		self.head = [element for element in self.all_areas if '_H' in element][0]
		self.action_dict = self.__create_action_dictionary() 
		self.num_actions = len(self.action_dict)

		self.action_cost = action_cost
		self.verbose = verbose
		self.max_steps = max_steps # max steps allowed in episode
		self.max_blocks = max_blocks
		self.episode_max_reward = episode_max_reward
		self.reward_decay_factor = reward_decay_factor

		self.num_blocks = max_blocks # TODO: make it random
		self.goal = list(range(self.num_blocks))# TODO: randomly shuffle block id
		self.correct_record = np.zeros_like(self.goal) # binary record for how many blocks are correct
		self.unit_reward = self.episode_max_reward / sum([self.reward_decay_factor**(b+1) for b in range(len(self.goal))])
		self.state, self.state_change_dict, self.area_state_dict, self.fiber_state_dict, self.assembly_dict, self.last_active_assembly = self.__create_state_representation()
		self.num_fibers = len(self.fiber_state_dict.keys())
		self.num_areas = len(self.area_state_dict.keys())
		self.num_assemblies = 0 # total number of assemblies ever created
		self.just_projected = False # record if the previous action was project
		self.current_time = 0 # current step in the episode
		self.all_correct = False # if the brain has represented all blocks correctly
		
		
	def reset(self, num_blocks=None, random_number_generator=np.random):
		self.num_blocks = self.max_blocks if num_blocks==None else num_blocks # TODO: make it random
		self.goal = list(range(self.num_blocks)) # TODO: make it random
		self.unit_reward = self.episode_max_reward / sum([self.reward_decay_factor**(b+1) for b in range(len(self.goal))])
		self.state, self.state_change_dict, self.area_state_dict, self.fiber_state_dict, self.assembly_dict, self.last_active_assembly = self.__create_state_representation()
		self.all_correct = False # if the brain has represented all blocks correctly
		self.correct_record = np.zeros_like(self.goal)
		self.num_assemblies = 0
		self.just_projected = False
		self.current_time = 0
		return self.state.copy(), None

	def close(self):
		self.current_time = 0
		self.state = None
		self.state_change_dict = None 
		self.area_state_dict = None 
		self.fiber_state_dict = None 
		self.assembly_dict = None 
		self.last_active_assembly = None
		self.all_correct = False # if the brain has represented all blocks correctly
		self.correct_record = np.zeros_like(self.goal)
		self.num_assemblies = 0
		self.just_projected = False
		return 

	def sample_expert_demo(self, nblocks=None):
		# [15,10,0,  14,   11,1, 15,6,2,  14,   7,3, 15,12,4,  14,   13,5, 0,8,15,  14,   1,9, 2,6,15,  14,   3,7, 4,12,15,  14]
		actions_block0 = [15,10,0] # optimal action indices for parsing first block
		actions_block1 = [11,1,15,6,2] # ... for parsing the second block
		actions_block2 = [7,3,15,12,4] # ... for parsing the third block
		actions_block3 = [13,5,0,8,15]
		actions_block4 = [1,9,2,6,15]
		if nblocks==None:
			nblocks = self.num_blocks
		expert_demo = []
		if nblocks>0:
			random.shuffle(actions_block0)
			expert_demo += actions_block0
			expert_demo += [14]
			nblocks -= 1
		if nblocks>0:
			random.shuffle(actions_block1)
			expert_demo += actions_block1
			expert_demo += [14]
			nblocks -= 1
		imodule = 0
		while nblocks > 0:
			if imodule%3 == 0:
				random.shuffle(actions_block2)
				expert_demo += actions_block2
				expert_demo += [14]
			elif imodule%3 == 1:
				random.shuffle(actions_block3)
				expert_demo += actions_block3
				expert_demo += [14]
			if imodule%3 == 2:
				random.shuffle(actions_block4)
				expert_demo += actions_block4
				expert_demo += [14]
			nblocks -= 1
			imodule += 1
		return expert_demo
		

	def step(self, action_idx):
		'''
		Return: new state, reward, terminated, truncated, info
		'''
		action_tuple = self.action_dict[int(action_idx)] # (action name, *kwargs)
		action_name = action_tuple[0]
		state_change_tuple = self.state_change_dict[int(action_idx)] # (state vec index, new state value)
		fiber_state_dict = self.fiber_state_dict # {state vec idx: (area1, area2)} 
		area_state_dict = self.area_state_dict # {area_name: state vec starting idx}
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
			astart = area_state_dict[BLOCKS] # starting index of area state including blocks area 
			if [*self.last_active_assembly.values()].count(-1)==len(self.last_active_assembly): # BAD, project when no active winners exist
				reward -= self.action_cost 
			elif self.just_projected: # BAD, consecutive project
				reward -= self.action_cost
			else: # GOOD, valid project
				self.state, self.assembly_dict, self.fiber_state_dict, self.last_active_assembly, self.num_assemblies = utils.synthetic_project(self.state, self.assembly_dict, self.fiber_state_dict, self.last_active_assembly, self.num_assemblies, self.verbose)
				for area_name in self.last_active_assembly.keys():  # update state for each area
					if (parse_cfg.skip_relocated and area_name==RELOCATED) or (area_name==BLOCKS):
						continue # only node and head areas need to update
					# update last active assembly in state 
					self.state[self.area_state_dict[area_name]] = self.last_active_assembly[area_name] 
					# update the number of BLOCKS related assemblies in this area
					count = 0 
					for assembly_info in self.assembly_dict[area_name]:
						connected_areas, connected_assemblies = assembly_info[0], assembly_info[1]
						if BLOCKS in connected_areas:
							count += 1
					self.state[self.area_state_dict[area_name]+1] = count
				# readout stack	
				readout = utils.synthetic_readout(self.assembly_dict, self.last_active_assembly, self.head, self.num_blocks)
				r, self.all_correct, self.correct_record = utils.calculate_readout_reward(readout, self.goal, self.correct_record, self.unit_reward, self.reward_decay_factor)
				reward += r 
			self.just_projected = True
		elif action_name == "activate_block":
			bidx = int(self.state[state_change_tuple[0]]) # currently activated block id
			newbidx = int(bidx) + state_change_tuple[1] # the new block id to be activated (prev or next block)
			if newbidx < 0 or newbidx >= self.num_blocks: # BAD, new block id is out of range
				reward -= self.action_cost
			else: # GOOD, valid activate
				self.state[state_change_tuple[0]] = newbidx # update block id in state vec
				self.last_active_assembly[BLOCKS] = newbidx # update the last active assembly
				if bidx == -1: # if no block activated before, and now has activated
					self.num_assemblies += 1 # increment assembly number record
			self.just_projected = False
		else:
			raise ValueError(f"\tError: action_idx {action_idx} is not recognized!")
		self.current_time += 1 # increment step in the episode 
		if self.current_time >= self.max_steps:
			truncated = True
		terminated = self.all_correct and utils.all_fiber_closed(self.state, self.fiber_state_dict)
		return self.state.copy(), reward, terminated, truncated, info


		
	def __create_state_representation(self):
		'''
		Initialize the episode state in the environmenmt. 
		Return:
			state: state representation (numpy array float32)
				[fiber inhibition status...,
				last activated assembly idx in the area, number of blocks-connected assemblies in the area, ...]
			dictionary: map action index to change in state vector
				{action_idx: ([state vector indices needed to be modified], [new values in these state indices])}
			area_state_dict: map area name to indices in state vector
				{area: [corresponding indices in state vector]}
			fiber_state_dict: mapping state vector index to fiber between two areas
				{state idx: (area1, area2)}
			assembly_dict: dictionary storing assembly associations that currently exist in the brain
				{area: [assembly_idx0[source areas[A0, A1], source assembly_idx[a0, a1]], 
						assembly_idx1[[A3], [a3]], 
						assembly_idx2[[A4], [a4]], 
						...]}
				i.e. area has assembly_idx0, which is associated/projected from area A0 assembly a0, and area A1 assembly a1
			last_active_assembly: dictionary storing the latest activated assembly idx in each area
				{area: assembly_idx}
				assembly_idx = -1 means that no previously activated assembly exists
		'''
		state_vec = [] # state vector
		dictionary = {} # action -> state change, {action_idx: ([state vector indices needed to be modified], [new values in these state indices])}
		action_idx = 0 # action index, the order should match that in self.action_dict
		state_vector_idx = 0 # initialize the idx in state vec to be changed
		blocks_state_idx = None # idx in state vec for the currently activated block id
		area_state_dict = {} # dict of index of each area
		fiber_state_dict = {} # mapping of state vec index to fiber name
		assembly_dict = {} # {area: [assembly1 associations...]}
		last_active_assembly = {} # {area: assembly_idx}
		# encode fiber inhibition status
		area_pairs = [] # record pairs of areas already visited
		for area1 in self.all_areas:
			if parse_cfg.skip_relocated and area1==RELOCATED:
				continue
			last_active_assembly[area1] = -1
			assembly_dict[area1] = [] # will become {area: [a_id 0[source a_name[A1, A2], source a_id [a1, a2]], 1[[A3], [a3]], 2[(A4, a4)], ...]}
			for area2 in self.all_areas:
				if parse_cfg.skip_relocated and area2==RELOCATED:
					continue
				if area1 == area2: # no need for fiber to the same area itself
					continue
				if (("_H" in area1) and ("_N0" not in area2)) or (("_H" in area2) and ("_N0" not in area1)):
					# for HEADS fibers, only consider N0. other fibers with HEADS are not considered
					continue
				if [area1, area2] in area_pairs: # skip already encoded area pairs
					continue
				# disinhibit fiber
				dictionary[action_idx] = ([state_vector_idx], 1) # fiber open
				action_idx += 1
				# inhibit fiber
				dictionary[action_idx] = ([state_vector_idx], 0) # fiber close
				action_idx += 1
				# add the current fiber state to state vector
				state_vec.append(0) # fiber should be locked upon initialization
				fiber_state_dict[state_vector_idx] = (area1, area2) # state element represents this fiber
				state_vector_idx += 1 # increment state idx
				# update visited area_pairs
				area_pairs.append([area1, area2])
				area_pairs.append([area2, area1])
		# information for project star
		dictionary[action_idx] = ([],[]) # no pre-specified new state values
		action_idx += 1
		# encode area state (i.e. most recent assembly idx in each area, and the number of BLOCKS-related assemblies in each area)
		# no need to encode area inhibition status since assuming action bundle
		for area_name in self.all_areas: # iterate all areas
			if parse_cfg.skip_relocated and area_name==RELOCATED:
				continue
			# store the starting state vec index for current area
			area_state_dict[area_name] = state_vector_idx
			if area_name==BLOCKS: # the state of blocks area only occupies one index
				blocks_state_idx = state_vector_idx
				state_vec.append(-1) # most recent assembly idx, -1 meaning no assembly record
				state_vector_idx += 1
				continue
			else:
				state_vec.append(-1) # most recent assembly idx, -1 meaning no assembly record
				state_vector_idx += 1 # increment current index
				state_vec.append(0) # number of BLOCKS-connected assemblies
				state_vector_idx += 1
		# info for activate next block
		dictionary[action_idx] = (blocks_state_idx, +1) 
		action_idx += 1
		# info for activate previous block
		dictionary[action_idx] = (blocks_state_idx, -1) 
		# initialize assembly dict for blocks area
		assembly_dict[BLOCKS] = [[[],[]] for _ in range(self.num_blocks)] 
		return np.array(state_vec, dtype=np.float32), dictionary, \
				area_state_dict, fiber_state_dict, assembly_dict, last_active_assembly


	def __create_action_dictionary(self):
		'''
		Create action dictionary: a dict that contains mapping of action index to action name
		Return dictonary {action_idx : (action name, *kwargs)}
		'''
		# action bundle: disinhibit_fiber entails disinhibiting the two areas and the fiber
		# disinhibit_fiber entails opening the fibers in both directions
		idx = 0 # action idx
		dictionary = {} # action idx --> action name
		area_pairs = [] # store pairs of areas already visited
		# disinhibit and inhibit fibers
		for area1 in self.all_areas:
			if parse_cfg.skip_relocated and area1==RELOCATED:
				continue
			for area2 in self.all_areas:
				if parse_cfg.skip_relocated and area2==RELOCATED:
					continue
				if area1 == area2: # no need for fiber to the same area itself
					continue
				if (("_H" in area1) and ("_N0" not in area2)) or (("_H" in area2) and ("_N0" not in area1)):
					continue # for HEADS fibers, only consider N0. other fibers with HEADS are not considered
				if [area1, area2] in area_pairs: # skip already included pairs
					continue 
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
	pprint.pprint(sim.action_dict)
	start_time = time.time()
	print(f'initial state: {sim.state}')
	for num_blocks in range(1, max_blocks+1):
		for _ in range(repeat):
			print(f'\n\n------------ initial state of {num_blocks} blocks\n{sim.reset(num_blocks=num_blocks)[0]}')
			expert_demo = sim.sample_expert_demo() if expert else None
			rtotal = 0 # total reward of episode
			nsteps = sim.max_steps if (not expert) else len(expert_demo)
			for t in range(nsteps):
				action_idx = random.choice(list(range(sim.num_actions))) if (not expert) else expert_demo[t]
				next_state, reward, terminated, truncated, info = sim.step(action_idx)
				rtotal += reward
				print(f't={t}, r={reward}, action={action_idx} {sim.action_dict[action_idx]}, done={terminated}, truncated={truncated}, \n\tnext state={next_state}')
			readout = utils.synthetic_readout(sim.assembly_dict, sim.last_active_assembly, sim.head, len(sim.goal))
			print(f'\nend of episode, synthetic readout {readout}, total reward={rtotal}')
			print(f'episode time lapse: {time.time()-start_time}')
			if expert:
				assert readout == sim.goal
				assert np.isclose(rtotal, sim.episode_max_reward-sim.action_cost*nsteps)





class EnvWrapper(dm_env.Environment):
	'''
	Wraps a Simulator object to be compatible with dm_env.Environment
	Reference: 
		https://github.com/wcarvalho/human-sf/blob/da0c65d04be708199ffe48d5f5118b295bfd43a3/lib/dm_env_wrappers.py#L15
		https://github.com/google-deepmind/dm_env/
		https://github.com/google-deepmind/acme/
	'''
	def __init__(self, environment: Simulator, random_number_generator):
		self._environment = environment
		self._reset_next_step = True
		self._last_info = None
		obs_space = self._environment.state
		act_space = self._environment.num_actions-1 # maximum action index
		self._observation_spec = _convert_to_spec(obs_space, name='observation')
		self._action_spec = _convert_to_spec(act_space, name='action')
		self.rng = random_number_generator

	def reset(self) -> dm_env.TimeStep:
		self._reset_next_step = False
		# self.rng = np.random.default_rng(self.rng.integers(low=0, high=100)) # refresh the rng
		self.rng = np.random.default_rng(random.randint(0,100)) # refresh the rng
		observation, info = self._environment.reset(random_number_generator=self.rng)
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
		min_val, max_val = space.min(), 20
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

	random.seed(1)
	test_simulator(max_blocks=2, expert=True, repeat=1, verbose=False)
	
	absltest.main()

