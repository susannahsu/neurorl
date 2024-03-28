import numpy as np
import random

from envs.blocksworld import utils
from envs.blocksworld.cfg import configurations
import envs.blocksworld.parse.Simulator as ParseSimulator

import dm_env
from dm_env import test_utils
from absl.testing import absltest
from acme import types, specs
from typing import Any, Dict, Optional
import tree

cfg = configurations['remove']

class Simulator(ParseSimulator):
	def __init__(self, 
				original_stack=None, # original stack of blocks before adding
				max_blocks = cfg['max_blocks'], # max number of blocks in any stack (after add)
				max_steps = cfg['max_steps'],
				action_cost = cfg['action_cost'],
				reward_decay_factor = cfg['reward_decay_factor'],
				episode_max_reward = cfg['episode_max_reward'],
				verbose=False):
		super().__init__(max_blocks = max_blocks,
						max_steps = max_steps,
						action_cost = action_cost,
						reward_decay_factor = reward_decay_factor,
						episode_max_reward = episode_max_reward,
						skip_relocated = skip_relocated,
						verbose = verbose)
		assert cfg['cfg'] == 'remove', f"cfg is {cfg['cfg']}"
		self.action_dict = self.create_action_dictionary() 
		self.num_actions = len(self.action_dict)

		self.original_stack = list(range(self.max_blocks)) if original_stack==None else original_stack # stack before adding. TODO: randomly shuffle block id
		
		assert 1 <= len(self.original_stack) <= self.max_blocks, f"number of blocks before removing ({len(self.original_stack)}) is not between [1, {self.max_blocks}]"
		self.goal = self.original_stack[1:] + [None] # goal stack after removing
		self.num_blocks = len(self.goal) # total number of blocks in the readout stack
		self.correct_record = np.zeros_like(self.goal) # binary record for how many blocks are correct
		self.unit_reward = utils.calculate_unit_reward(self.reward_decay_factor, len(self.goal), self.episode_max_reward)
		self.state, self.state_change_dict, self.stimulate_state_dict, self.fiber_state_dict, self.assembly_dict, self.last_active_assembly = self.create_state_representation()
		self.num_fibers = len(self.fiber_state_dict.keys())
		self.num_assemblies = 0 # total number of assemblies ever created
		self.just_projected = False # record if the previous action was project
		self.current_time = 0 # current step in the episode
		self.all_correct = False # whether the brain has represented all blocks correctly

		
	def reset(self, original_stack=None, random_number_generator=np.random):
		self.original_stack = list(range(self.max_blocks)) if original_stack==None else original_stack # stack before adding. TODO: randomly shuffle block id
		assert 1 <= len(self.original_stack) <= self.max_blocks, f"number of blocks before removing ({len(self.original_stack)}) is not between [1, {self.max_blocks}]"
		self.goal = self.original_stack[1:] + [None] # goal stack after removing
		self.num_blocks = len(self.goal) # total number of blocks in readout stack
		assert self.num_blocks <= self.max_blocks, f"number of blocks ({self.num_blocks}) exceeds max ({self.max_blocks})"
		self.unit_reward = utils.calculate_unit_reward(self.reward_decay_factor, len(self.goal), self.episode_max_reward)
		self.state, self.state_change_dict, self.stimulate_state_dict, self.fiber_state_dict, self.assembly_dict, self.last_active_assembly = self.create_state_representation()
		self.all_correct = False 
		self.correct_record = np.zeros_like(self.goal)
		self.num_assemblies = 0
		self.just_projected = False
		self.current_time = 0
		return self.state.copy(), None


	def close(self):
		self.current_time = 0
		self.state = None
		self.state_change_dict = None 
		self.fiber_state_dict = None 
		self.assembly_dict = None 
		self.last_active_assembly = None
		self.all_correct = False 
		self.correct_record = np.zeros_like(self.goal)
		self.num_assemblies = 0
		self.just_projected = False
		return 
	

	def sample_expert_demo(self):
		final_actions = []
		top_area_name, top_area_a, top_block_idx = utils.top(self.assembly_dict, self.last_active_assembly, self.head)
		# first check if any fiber needs to be inhibited
		inhibit_actions = []
		for stateidx in self.fiber_state_dict:
			if self.state[stateidx] == 1:
				inhibit_actions.append(("inhibit_fiber", self.fiber_state_dict[stateidx][0], self.fiber_state_dict[stateidx][1]))
			assert self.state[stateidx]==0, f"something wrong with fiber status in state vector {self.state}, idx {stateidx} should be inhibited"
		for action in inhibit_actions:
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		# check is last block
		if utils.is_last_block(self.assembly_dict, self.head, top_area_name, top_area_a, self.blocks_area):
			# print('\t\tsample expert remove, is last block!')
			action = ("deactivate_head", None)
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			return final_actions
		# if not last block, get the new top area
		if '_N0' in top_area_name:
			new_top_area = [area for area in self.all_areas if '_N1' in area][0]
		elif '_N1' in top_area_name:
			new_top_area = [area for area in self.all_areas if '_N2' in area ][0]
		elif '_N2' in top_area_name:
			new_top_area = [area for area in self.all_areas if '_N0' in area ][0]
		# activate corresponding assemblies
		action = ("stimulate", self.head, top_area_name)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("stimulate", top_area_name, new_top_area)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("stimulate", new_top_area, self.blocks_area)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		# build new connection
		action = ("disinhibit_fiber", new_top_area, self.head)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("disinhibit_fiber", self.blocks_area, new_top_area)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("project_star", None)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("inhibit_fiber", new_top_area, self.head)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		action = ("inhibit_fiber", self.blocks_area, new_top_area)
		final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
		return final_actions


	def step(self, action_idx):
		action_tuple = self.action_dict[action_idx] # (action name, *kwargs)
		action_name = action_tuple[0]
		state_change_tuple = self.state_change_dict[action_idx] # (state vec index, new state value)
		fiber_state_dict = self.fiber_state_dict # {state vec idx: (area1, area2)} 
		# area_state_dict = self.area_state_dict # {area_name: state vec starting idx}
		reward = -self.action_cost # default cost for performing any action
		terminated = False
		truncated = False
		info = None
		if (action_name== "disinhibit_fiber") or (action_name == "inhibit_fiber"):
			area1, area2 = action_tuple[1], action_tuple[2]
			if self.state[state_change_tuple[0]] == state_change_tuple[1]: # BAD, fiber is already disinhibited/inhibited
				reward -= self.action_cost
			self.state[state_change_tuple[0]] = state_change_tuple[1] # update state
			self.just_projected = False
		elif action_name == "project_star": # state_change_tuple = ([],[]) s
			if [*self.last_active_assembly.values()].count(-1)==len(self.last_active_assembly): # BAD, no active winners exist in advance
				reward -= self.action_cost
			elif self.just_projected: # BAD, consecutive project
				reward -= self.action_cost
			else: # GOOD, valid project
				self.assembly_dict, self.last_active_assembly, self.num_assemblies = utils.synthetic_project(self.state, self.assembly_dict, self.fiber_state_dict, self.last_active_assembly, self.num_assemblies, self.verbose, blocks_area=self.blocks_area)
				'''
				# update neuron state for each area
				for area_name in self.last_active_assembly.keys(): 
					if (self.skip_relocated and area_name==self.relocated_area) or (area_name==self.blocks_area):
						continue # only node and head areas need to update
					# update last active assembly id in each area
					self.state[self.area_state_dict[area_name]] = self.last_active_assembly[area_name] 
					# update the number of self.blocks_area related assemblies in each area
					count = 0 
					for assembly in self.assembly_dict[area_name]:
						connected_areas, connected_area_idxs = assembly[0], assembly[1]
						if self.blocks_area in connected_areas:
							count += 1
					self.state[self.area_state_dict[area_name]+1] = count
				'''
				readout = utils.synthetic_readout(self.assembly_dict, self.last_active_assembly, self.head, self.num_blocks, self.blocks_area)
				r, self.all_correct, self.correct_record = utils.calculate_readout_reward(readout, self.goal, self.correct_record, self.unit_reward, self.reward_decay_factor)
				reward += r
				top_area_name, top_area_a, bid = utils.top(self.assembly_dict, self.last_active_assembly, self.head) # update top area and is last block
				if top_area_name==None:
					self.state[-2] = -1
				elif "_N0" in top_area_name:
					self.state[-2] = 0
				elif "_N1" in top_area_name:
					self.state[-2] = 1
				elif "_N2" in top_area_name: 
					self.state[-2] = 2
				else:
					self.state[-2] = -1
				# update is last block in state vector, and goal
				# if utils.is_last_block(self.assembly_dict, self.head, top_area_name, top_area_a, self.blocks_area): # TODO: no need to update this during stepping
				# 	self.state[-1] = 1
				# else:
				# 	self.state[-1] = 0
			self.just_projected = True
		elif action_name == "stimulate":
			area1, area2 = action_tuple[1], action_tuple[2]
			self.stimulate(area1, area2)
			'''
			# update neuron state for each area
			for area_name in self.last_active_assembly.keys(): 
				if (self.skip_relocated and area_name==self.relocated_area) or (area_name==self.blocks_area):
					continue # only node and head areas need to update
				# update last active assembly id in each area
				self.state[self.area_state_dict[area_name]] = self.last_active_assembly[area_name] 
			'''
			if self.state[state_change_tuple[0]] == state_change_tuple[1]: # BAD, fiber is already stimulated
				reward -= self.action_cost
			self.state[state_change_tuple[0]] = state_change_tuple[1] # update state
			self.just_projected = False
		elif action_name == "deactivate_head":
			self.last_active_assembly[self.head] = -1
			if self.state[-1]==1: # TODO: why?
				self.all_correct = True
			self.just_projected = False
		else:
			raise ValueError(f"action_idx {action_idx} not recognized!")
		self.current_time += 1 # increment step in the episode 
		if self.current_time >= self.max_steps:
			truncated = True
		terminated = self.all_correct and utils.all_fiber_closed(self.state, self.fiber_state_dict)
		return self.state.copy(), reward, terminated, truncated, info


	def stimulate(self, area1, area2):
		if (self.last_active_assembly[area1] != -1) and (area2 in self.assembly_dict[area1][self.last_active_assembly[area1]][0]):
			a2 = self.assembly_dict[area1][self.last_active_assembly[area1]][0].index(area2)
			self.last_active_assembly[area2] = self.assembly_dict[area1][self.last_active_assembly[area1]][1][a2]
		
		
	def create_state_representation(self):
		'''
		Create initial state vector, 
				and a dictionary mapping action index to state change 
				(i.e. for performing each action, what part of the state vec needs to be changed).
		'''
		state_vec = [] # initialize the state vector
		dictionary = {} # initialize action->state change dict, {action_idx: ([state vector indices needed to be modified], [new values in these state indices])}
		action_idx = 0 # initialize the action index for iteration, the order should match that in self.action_dict
		state_vector_idx = 0 # initialize the idx in state vec to be changed
		# blocks_state_idx = None # idx in state vec for the currently activated block id
		# area_state_dict = {} # dict of state vector index for each area
		fiber_state_dict = {} # mapping of state vec index to fiber name
		stimulate_state_dict = {}
		assembly_dict = {} # {(area1, area2): (assembly idx 1, assembly idx 2)}
		last_active_assembly = {} # {area: assembly_idx}
		top_area_state_idx = None # index in state vector storing the top block node area index
		is_last_block_state_idx = None # index in state vector storing if the top block is the last block in stack
		# fiber inhibition status (for action bundle, no need to encode area inhibition status)
		area_pairs = [] # store pairs of areas already visited
		for area1 in self.all_areas:
			if self.skip_relocated and area1==self.relocated_area:
				continue
			last_active_assembly[area1] = -1
			assembly_dict[area1] = [] # will become # {area: [a_id 0[source a_name[A1, A2], source a_id [a1, a2]], 1[[A3], [a3]], 2[(A4, a4)], ...]}
			for area2 in self.all_areas:
				if self.skip_relocated and area2==self.relocated_area:
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
				elif ('_N0' in area1 and '_N1' in area2) or ('_N1' in area1 and '_N2' in area2)	or ('_N2' in area1 and '_N0' in area2):
					state_vec.append(0)
					dictionary[action_idx] = ([state_vector_idx], 1)
					stimulate_state_dict[state_vector_idx] = (area1, area2)
					state_vector_idx += 1
					action_idx += 1
				elif ('_N0' in area2 and '_N1' in area1) or ('_N1' in area2 and '_N2' in area1)	or ('_N2' in area2 and '_N0' in area1):
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
		'''
		# area state (i.e. most recent assembly idx in each area, and the number of self.blocks_area-related assemblies in each area)
		for area_name in self.all_areas: # iterate all areas
			if self.skip_relocated and area_name==self.relocated_area:
				continue
			# store the starting state vec index for current area
			area_state_dict[area_name] = state_vector_idx
			if area_name==self.blocks_area: # the state of blocks area only occupies one index
				blocks_state_idx = state_vector_idx
				state_vec.append(-1) # most recent assembly idx, -1 meaning no assembly record
				state_vector_idx += 1
				continue
			else:
				state_vec.append(-1) # most recent assembly idx, -1 meaning no assembly record
				state_vector_idx += 1 # increment current index
				state_vec.append(0) # number of self.blocks_area-connected assemblies
				state_vector_idx += 1
		'''
		# deactivate head
		dictionary[action_idx] = ([], [])
		action_idx += 1
		# initial area for top block should be -1
		state_vec.append(-1)
		top_area_state_idx = state_vector_idx
		state_vector_idx += 1
		# is last block should be False
		state_vec.append(0)
		is_last_block_state_idx = state_vector_idx
		state_vector_idx += 1
		# initialize assembly dict for blocks
		assembly_dict[self.blocks_area] = [[[],[]] for _ in range(self.num_blocks)] 
		return np.array(state_vec, dtype=np.float32), \
					dictionary, \
					stimulate_state_dict, \
					fiber_state_dict, \
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
		dictionary = {} # action idx --> action name
		area_pairs = [] # store pairs of areas already visited
		# disinhibit, inhibit fibers, stimulate area pairs
		for area1 in self.all_areas:
			if self.skip_relocated and area1==self.relocated_area:
				continue
			for area2 in self.all_areas:
				if self.skip_relocated and area2==self.relocated_area:
					continue
				if area1 == area2: # no need for fiber to the same area itself
					continue
				if (("_H" in area1) and (area2==self.blocks_area)) or (("_H" in area2) and (area1==self.blocks_area)):
					# for HEADS fibers, only consider N0, N1, N2, do not consider connection with self.blocks_area
					continue
				if [area1, area2] in area_pairs: # skip already included pairs
					continue 
				dictionary[idx] = ("disinhibit_fiber", area1, area2)
				idx += 1
				dictionary[idx] = ("inhibit_fiber", area1, area2)
				idx += 1
				if ('_H' in area1) or (area2==self.blocks_area):
					dictionary[idx] = ("stimulate", area1, area2)
					idx += 1
				elif ('_H' in area2) or (area1==self.blocks_area):
					dictionary[idx] = ("stimulate", area2, area1)
					idx += 1
				elif ('_N0' in area1 and '_N1' in area2) or ('_N1' in area1 and '_N2' in area2)	or ('_N2' in area1 and '_N0' in area2):
					dictionary[idx] = ("stimulate", area1, area2)
					idx += 1
				elif ('_N0' in area2 and '_N1' in area1) or ('_N1' in area2 and '_N2' in area1)	or ('_N2' in area2 and '_N0' in area1):
					dictionary[idx] = ("stimulate", area2, area1)
					idx += 1
				# update area_pairs 
				area_pairs.append([area1, area2])
				area_pairs.append([area2, area1])
		# project star
		dictionary[idx] = ("project_star", None)
		idx += 1
		# deactivate head
		dictionary[idx] = ("deactivate_head", None)
		idx += 1
		return dictionary


def test_simulator(max_blocks=7, expert=True, repeat=10, verbose=False):
	import time
	sim = Simulator(max_blocks=max_blocks, verbose=verbose)
	print(f'action_dict: {sim.action_dict}\ninitial state: {sim.state}')
	start_time = time.time()
	for num_blocks in range(1, max_blocks+1):
		for _ in range(repeat):
			print(f'\n\n------------ initial state of {num_blocks} blocks\n{sim.reset()[0]}')
			expert_demo = sim.sample_expert_demo() if expert else None
			print(f'expert demo: {expert_demo}')
			rtotal = 0 # total reward of episode
			nsteps = sim.max_steps if (not expert) else len(expert_demo)
			for t in range(nsteps):
				action_idx = random.choice(list(range(sim.num_actions))) if (not expert) else expert_demo[t]
				next_state, reward, terminated, truncated, info = sim.step(action_idx)
				rtotal += reward
				print(f't={t}, r={reward}, action={action_idx} {sim.action_dict[action_idx]}, done={terminated}, truncated={truncated}, \n\tnext state={next_state}')
			readout = utils.synthetic_readout(sim.assembly_dict, sim.last_active_assembly, sim.head, len(sim.goal), sim.blocks_area)
			print(f'\nend of episode, synthetic readout {readout}, total reward={rtotal}')
			print(f'episode time lapse: {time.time()-start_time}')
			if expert:
				assert readout == sim.goal, f"readout {readout} does not match goal {sim.goal}"
				assert np.isclose(rtotal, sim.episode_max_reward-sim.action_cost*nsteps), f"rtotal {rtotal} does not match theoretical {sim.episode_max_reward-sim.action_cost*nsteps}"

if __name__ == "__main__":

	random.seed(1)
	test_simulator(max_blocks=2, expert=True, repeat=1, verbose=False)
	