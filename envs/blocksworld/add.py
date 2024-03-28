from envs.blocksworld import add_cfg
from envs.blocksworld import utils
import numpy as np

import dm_env
from dm_env import test_utils
from absl.testing import absltest
from acme import types, specs
from typing import Any, Dict, Optional
import tree


class Simulator():
	def __init__(self, 
					oldstack = None, # original stack of blocks before adding
					newblockid = None, # the new block id to be added to the top of original stack
					max_blocks = add_cfg.max_blocks, # max number of blocks in any stack (after add)
					max_steps = add_cfg.max_steps,
					action_cost = add_cfg.action_cost,
					reward_decay_factor = add_cfg.reward_decay_factor,
					episode_max_reward = add_cfg.episode_max_reward,
					verbose=False):
		self.all_areas, self.head, self.relocated_area, self.blocks_area = utils.init_simulator_areas()
		self.action_dict = self.__create_action_dictionary() 
		self.n_actions = len(self.action_dict)
		self.action_cost = action_cost
		self.verbose = verbose
		self.max_steps = max_steps # max steps allowed in episode
		self.max_blocks = max_blocks # maximum number of blocks allowed in any stack (including after adding)
		self.episode_max_reward = episode_max_reward
		self.reward_decay_factor = reward_decay_factor

		self.original_stack = list(range(self.max_blocks-1)) if oldstack==None else oldstack # stack before adding. TODO: randomly shuffle block id
		self.newblock = self.max_blocks-1 if newblockid==None else newblockid # the new block to be added. TODO: make it random
		
		self.goal = [newblockid] + original_stack # goal stack after adding
		self.num_blocks = len(goal) # total number of blocks after adding
		assert self.num_blocks <= self.max_blocks, f"number of blocks after adding ({self.num_blocks}) exceeds max ({self.max_blocks})"
		self.correct_record = np.zeros_like(self.goal) # binary record for how many blocks are correct
		self.unit_reward = utils.calculate_unit_reward(self.reward_decay_factor, len(self.goal), self.episode_max_reward)
		self.state, self.state_change_dict, self.stimulate_state_dict, self.fiber_state_dict, self.assembly_dict, self.last_active_assembly = self.__create_state_representation()
		self.num_fibers = len(self.fiber_state_dict.keys())
		self.num_assemblies = 0 # total number of assemblies ever created
		self.just_projected = False # record if the previous action was project
		self.current_time = 0 # current step in the episode
		self.all_correct = False # if the brain has represented all blocks correctly

	def reset(self, oldstack=None, newblockid=None, random_number_generator=np.random):
		self.original_stack = list(range(self.max_blocks-1)) if oldstack==None else oldstack # stack before adding. TODO: randomly shuffle block id
		self.newblock = self.max_blocks-1 if newblockid==None else newblockid # the new block to be added. TODO: make it random
		self.goal = [newblockid] + original_stack # goal stack after adding
		self.num_blocks = len(goal) # total number of blocks after adding
		assert self.num_blocks <= self.max_blocks, f"number of blocks after adding ({self.num_blocks}) exceeds max ({self.max_blocks})"
		self.unit_reward = utils.calculate_unit_reward(self.reward_decay_factor, len(self.goal), self.episode_max_reward)
		self.state, self.state_change_dict, self.stimulate_state_dict, self.fiber_state_dict, self.assembly_dict, self.last_active_assembly = self.__create_state_representation()
		self.all_correct = False # if the brain has represented all blocks correctly
		self.correct_record = np.zeros_like(self.goal)
		self.num_assemblies = 0
		self.just_projected = False
		self.current_time = 0
		return self.state.copy(), None
	
	def import_from_parse(self, parse):
		self.just_projected = parse.just_projected
		self.num_assemblies = parse.num_assemblies
		self.num_blocks = parse.num_blocks
		self.all_correct = False
		self.terminated = False
		# update assembly dict, last active assembly, and corresponding state elements
		for area in parse.all_areas:
			if SKIP_RELOCATED and area==self.relocated_area:
				continue
			self.assembly_dict[area] = copy.deepcopy(parse.assembly_dict[area])
			self.last_active_assembly[area] = copy.deepcopy(parse.last_active_assembly[area])
		# close all fibers are closed when importing
		for i in self.fiber_state_dict.keys():
			self.state[i] = 0
		# update top area in state vector
		top_area_name, top_area_a, bid = utils.top(self.assembly_dict, self.last_active_assembly, self.head)
		new_top_area = None
		if top_area_name == None:
			self.state[-1] = -1
			new_top_area = [area for area in self.all_areas if '_N2' in area][0]
		elif "_N0" in top_area_name:
			self.state[-1] = 0
			new_top_area = [area for area in self.all_areas if '_N2' in area][0]
		elif "_N1" in top_area_name:
			self.state[-1] = 1
			new_top_area = [area for area in self.all_areas if '_N0' in area][0]
		elif "_N2" in top_area_name: 
			self.state[-1] = 2
			new_top_area = [area for area in self.all_areas if '_N1' in area][0]
		# update encoded status in new top area
		self.state[-2] = 0
		if self.last_active_assembly[new_top_area] != -1 and (self.blocks_area in self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0]):
			tmpidx = self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0].index(self.blocks_area)
			if self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][1][tmpidx] == self.newblock:
				self.state[-2] = 1 # new block is encoded in new top area
		# update all correct status
		self.state[-4] = 0
		# update newblockid activate status
		self.state[-3] = 0
		if self.last_active_assembly[self.blocks_area] ==self.newblock:
			self.state[-3] = 1
		# update goal
		stack = utils.synthetic_readout(parse.assembly_dict, parse.last_active_assembly, parse.head, parse.num_blocks, parse.blocks_area)
		if top_area_name==None:
			self.goal = [self.newblock] 
		else:
			self.goal = [self.newblock] + stack[:-1]
		assert len(self.goal)<=self.num_blocks
		return self.state

	def import_from_remove(self, remove):
		self.just_projected = remove.just_projected
		self.num_assemblies = remove.num_assemblies
		self.num_blocks = remove.num_blocks
		self.all_correct = False
		self.terminated = False
		# update assembly dict, last active assembly, and corresponding state elements
		for area in remove.all_areas:
			if SKIP_RELOCATED and area==self.relocated_area:
				continue
			self.assembly_dict[area] = copy.deepcopy(remove.assembly_dict[area])
			self.last_active_assembly[area] = copy.deepcopy(remove.last_active_assembly[area])
		# close all fibers 
		for i in self.fiber_state_dict.keys():
			self.state[i] = 0
		# reset stimulate status to 0
		for i in self.stimulate_state_dict.keys():
			self.state[i] = 0 
		# update top area in state vector
		top_area_name, top_area_a, bid = utils.top(self.assembly_dict, self.last_active_assembly, self.head)
		new_top_area = None
		if top_area_name==None:
			self.state[-1] = -1
			new_top_area = [area for area in self.all_areas if '_N2' in area][0]
		elif "_N0" in top_area_name:
			self.state[-1] = 0
			new_top_area = [area for area in self.all_areas if '_N2' in area][0]
		elif "_N1" in top_area_name:
			self.state[-1] = 1
			new_top_area = [area for area in self.all_areas if '_N0' in area][0]
		elif "_N2" in top_area_name: 
			self.state[-1] = 2		
			new_top_area = [area for area in self.all_areas if '_N1' in area][0]
		# update encoded status in new top area
		self.state[-2] = 0
		if self.last_active_assembly[new_top_area] != -1 and (self.blocks_area in self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0]):
			tmpidx = self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0].index(self.blocks_area)
			if self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][1][tmpidx] == self.newblock:
				self.state[-2] = 1 # new block is encoded in new top area
		# update all correct status
		self.state[-4] = 0
		# update newblockid activate status
		self.state[-3] = 0
		if self.last_active_assembly[self.blocks_area] ==self.newblock:
			self.state[-3] = 1
		# update goal
		stack = utils.synthetic_readout(remove.assembly_dict, remove.last_active_assembly, remove.head, remove.num_blocks, remove.blocks_area)
		if top_area_name==None:
			self.goal = [self.newblock] 
		else:
			self.goal = [self.newblock] + stack[:-1]
		assert len(self.goal)<=self.num_blocks
		return self.state

	def sample_expert_demo(self):
		final_actions = []
		top_area_name, top_area_a, top_block_idx = utils.top(self.assembly_dict, self.last_active_assembly, self.head)
		# first check that all fibers are closed
		for stateidx in self.fiber_state_dict:
			assert self.state[stateidx]==0
		# if not last block, get the new top area
		flip = False
		if top_area_name==None:
			new_top_area = [area for area in self.all_areas if '_N2' in area][0]
			action = ("activate_block", None)
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			action = ("disinhibit_fiber", self.blocks_area, new_top_area)
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			action = ("project_star", None)
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			action = ("disinhibit_fiber", new_top_area, self.head) 
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			action = ("project_star", None)
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			action = ("inhibit_fiber", self.blocks_area, new_top_area)
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			action = ("inhibit_fiber", new_top_area, self.head) 
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			return final_actions
		elif '_N0' in top_area_name:
			new_top_area = [area for area in self.all_areas if '_N2' in area][0]
		elif '_N1' in top_area_name:
			new_top_area = [area for area in self.all_areas if '_N0' in area ][0]
			flip = True
		elif '_N2' in top_area_name:
			new_top_area = [area for area in self.all_areas if '_N1' in area ][0]
			flip = True
		# activate block
		action = ("activate_block", None)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("disinhibit_fiber", self.blocks_area, new_top_area)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("project_star", None)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		# stimulate head
		action = ("stimulate", self.head, top_area_name)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("disinhibit_fiber", new_top_area, top_area_name) if flip else  ("disinhibit_fiber", top_area_name, new_top_area)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		# project new block to head
		action = ("disinhibit_fiber", new_top_area, self.head) 
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("project_star", None)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("inhibit_fiber", new_top_area, self.head) 
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		# close all fibers
		action = ("inhibit_fiber", new_top_area, top_area_name) if flip else  ("inhibit_fiber", top_area_name, new_top_area)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("inhibit_fiber", self.blocks_area, new_top_area)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		return final_actions

	def step(self, action_idx):
		'''
		Return: new state, reward, terminated, truncated, info
		'''
		action_tuple = self.action_dict[action_idx] # (action name, *kwargs)
		action_name = action_tuple[0]
		state_change_tuple = self.state_change_dict[action_idx] # (state vec index, new state value)
		fiber_state_dict = self.fiber_state_dict # {state vec idx: (area1, area2)} 
		reward = -self.action_cost # default cost for performing any action
		terminated = False # whether episode completes
		truncated = False # whether episode ends due to max steps reached
		info = None
		if (action_name == "disinhibit_fiber") or (action_name == "inhibit_fiber"):
			area1, area2 = action_tuple[1], action_tuple[2]
			if self.state[state_change_tuple[0]] == state_change_tuple[1]: # BAD, fiber is already disinhibited/inhibited
				reward -= self.action_cost
			self.state[state_change_tuple[0]] = state_change_tuple[1] # update state 
			self.just_projected = False
		elif action_name == "project_star": # state_change_tuple = ([],[]) 
			if [*self.last_active_assembly.values()].count(-1)==len(self.last_active_assembly): # BAD, project with no winners
				reward -= self.action_cost
			elif self.just_projected: # BAD, consecutive project
				reward -= self.action_cost
			else: # GOOD, valid project
				self.assembly_dict, self.last_active_assembly, self.num_assemblies = utils.synthetic_project(self.state, self.assembly_dict, self.fiber_state_dict, self.last_active_assembly, self.num_assemblies, self.verbose, blocks_area=self.blocks_area)
				readout = utils.synthetic_readout(self.assembly_dict, self.last_active_assembly, self.head, self.num_blocks, self.blocks_area)
				r, self.all_correct, self.correct_record = utils.calculate_readout_reward(readout, self.goal, self.correct_record, self.unit_reward, self.reward_decay_factor)
				reward += r
				# update all correct status
				if self.all_correct:
					self.state[-4] = 1
				else:
					self.state[-4] = 0
				# update top area 
				top_area_name, top_area_a, bid = utils.top(self.assembly_dict, self.last_active_assembly, self.head)
				new_top_area = None
				if top_area_name==None:
					self.state[-1] = -1
					new_top_area = [area for area in self.all_areas if '_N2' in area][0]
				elif "_N0" in top_area_name:
					self.state[-1] = 0
					new_top_area = [area for area in self.all_areas if '_N2' in area][0]
				elif "_N1" in top_area_name:
					state[-1] = 1
					new_top_area = [area for area in self.all_areas if '_N0' in area ][0]
				elif "_N2" in top_area_name: 
					self.state[-1] = 2
					new_top_area = [area for area in self.all_areas if '_N1' in area ][0]
				# update encoded status in new top area
				self.state[-2] = 0
				if self.state[-2]!= 1 and (self.last_active_assembly[new_top_area] != -1) and (self.blocks_area in self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0]):
					tmpidx = self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0].index(self.blocks_area)
					if self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][1][tmpidx] == self.newblock:
						self.state[-2] = 1 # new block is encoded in new top area
			self.just_projected = True
		elif action_name == "stimulate":
			area1, area2 = action_tuple[1], action_tuple[2]
			self.stimulate(area1, area2)
			if self.state[state_change_tuple[0]] == state_change_tuple[1]: # BAD, fiber is already stimulated
				reward -= self.action_cost
			self.state[state_change_tuple[0]] = state_change_tuple[1] # update state
			self.just_projected = False
		elif action_name == "activate_block":
			if self.state[state_change_tuple[0]]==state_change_tuple[1]: # BAD, block is already activated
				reward -= self.action_cost
			self.state[state_change_tuple[0]] = state_change_tuple[1] # update state
			self.last_active_assembly[self.blocks_area] = self.newblock
			self.just_projected = False
		else:
			raise ValueError(f"\tError: action_idx {action_idx} is not recognized!")
		self.current_time += 1
		if self.current_time >= self.max_steps:
			truncated = True
		terminated = self.all_correct and (utils.all_fiber_closed(self.state, self.fiber_state_dict))
		if terminated:
			reward += self.action_cost
		return self.state.copy(), reward, terminated, truncated, info

	def stimulate(self, area1, area2):
		# stimulate area2, using the currently activated assembly in area1
		if (self.last_active_assembly[area1] != -1) and (area2 in self.assembly_dict[area1][self.last_active_assembly[area1]][0]):
			a2 = self.assembly_dict[area1][self.last_active_assembly[area1]][0].index(area2)
			self.last_active_assembly[area2] = self.assembly_dict[area1][self.last_active_assembly[area1]][1][a2]
		
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
		action_idx = 0 # action_idx = 0 # action index, the order should match that in self.action_dict
		state_vector_idx = 0 # initialize the idx in state vec to be changed
		fiber_state_dict = {} # mapping of state vec index to fiber name
		stimulate_state_dict = {}
		assembly_dict = {} # {area: [assembly1 associations...]}
		last_active_assembly = {} # {area: assembly_idx}
		# encode fiber inhibition status (for action bundle, no need to encode area inhibition status)
		area_pairs = [] # store pairs of areas already visited
		for area1 in self.all_areas:
			if add_cfg.skip_relocated and area1==self.relocated_area:
				continue
			last_active_assembly[area1] = -1
			assembly_dict[area1] = [] # will become # {area: [a_id 0[source a_name[A1, A2], source a_id [a1, a2]], 1[[A3], [a3]], 2[(A4, a4)], ...]}
			for area2 in self.all_areas:
				if add_cfg.skip_relocated and area2==self.relocated_area:
					continue
				if area1 == area2: # no need for fiber to the same area itself
					continue
				if (("_H" in area1) and (area2==self.blocks_area)) or (("_H" in area2) and (area1==self.blocks_area)):
					# for HEADS fibers, only consider N0, N1, N2, do not consider connection with self.blocks_area
					continue
				if [area1, area2] in area_pairs: # skip already encoded area pairs
					continue
				# disinhibit fiber
				dictionary[action_idx] = ([state_vector_idx], 1) # fiber opened
				action_idx += 1
				# inhibit fiber
				dictionary[action_idx] = ([state_vector_idx], 0) # fiber closed
				action_idx += 1
				# add the current fiber state to state vector
				state_vec.append(0) # fiber should be locked in the beginning
				fiber_state_dict[state_vector_idx] = (area1, area2) # add pair to corresponding state idx
				state_vector_idx += 1 # increment state idx
				# stimulate status
				if ('_H' in area1) or (area2==self.blocks_area):
					state_vec.append(0)
					dictionary[action_idx] = ([state_vector_idx], 1)
					stimulate_state_dict[state_vector_idx] = (area1, area2)
					state_vector_idx += 1
					action_idx += 1
				elif ('_H' in area2) or (area1==self.blocks_area):
					state_vec.append(0)
					dictionary[action_idx] = ([state_vector_idx], 1)
					stimulate_state_dict[state_vector_idx] = (area2, area1)
					state_vector_idx += 1
					action_idx += 1
				# update area_pairs
				area_pairs.append([area1, area2])
				area_pairs.append([area2, area1])
		# project star
		dictionary[action_idx] = ([],[]) # no pre-specified new state values
		action_idx += 1
		# all correct status, initialize as 0
		state_vec.append(0)
		state_vector_idx += 1
		# activate new block id in self.blocks_area, newblockid activation status initialized as 0
		state_vec.append(0)
		dictionary[action_idx] = ([state_vector_idx], 1) # if newblockid activated, set the status to 1
		action_idx += 1
		state_vector_idx += 1
		# new top area encode status initialized as 0
		state_vec.append(0)
		state_vector_idx += 1
		# initial area for top block should be -1
		state_vec.append(-1)
		state_vector_idx += 1
		# initialize assembly dict for blocks
		assembly_dict[self.blocks_area] = [[[],[]] for _ in range(self.num_blocks)] 
		return np.array(state_vec, dtype=np.float32), \
					dictionary, \
					stimulate_state_dict, \
					fiber_state_dict, \
					assembly_dict, \
					last_active_assembly

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
		# disinhibit, inhibit fibers, stimulate area pairs
		for area1 in self.all_areas:
			if add_cfg.skip_relocated and area1==self.relocated_area:
				continue
			for area2 in self.all_areas:
				if add_cfg.skip_relocated and area2==self.relocated_area:
					continue # skip relocated area
				if area1 == area2: 
					continue # no need for fiber to the same area itself
				if (("_H" in area1) and (area2==self.blocks_area)) or (("_H" in area2) and (area1==self.blocks_area)):
					continue # for HEADS fibers, only consider N0, N1, N2, do not consider connection with self.blocks_area
				if [area1, area2] in area_pairs:
					continue  # skip already included pairs
				dictionary[idx] = ("disinhibit_fiber", area1, area2)
				idx += 1
				dictionary[idx] = ("inhibit_fiber", area1, area2)
				idx += 1
				if ('_H' in area1) or (area2==self.blocks_area): # TODO: why need stimulate? replace every stimulate by project?
					dictionary[idx] = ("stimulate", area1, area2)
					idx += 1
				elif ('_H' in area2) or (area1==self.blocks_area):
					dictionary[idx] = ("stimulate", area2, area1)
					idx += 1
				# update area_pairs 
				area_pairs.append([area1, area2])
				area_pairs.append([area2, area1])
		# project star
		dictionary[idx] = ("project_star", None)
		idx += 1
		# activate new block id in self.blocks_area
		dictionary[idx] = ("activate_block", None)
		idx += 1
		return dictionary

