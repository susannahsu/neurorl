from envs.blocksworld.AC.bw_apps import *
from envs.blocksworld import add_cfg
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
					oldstack = None, # original stack of blocks before adding
					newblockid = None, # the new block id to be added to the top of original stack
					max_blocks = add_cfg.max_blocks, # max number of blocks in any stack (after add)
					max_stacks = add_cfg.max_stacks, # default 1, brain should only take 1 stack
					max_steps = add_cfg.max_steps,
					action_cost = add_cfg.action_cost,
					reward_decay_factor = add_cfg.reward_decay_factor,
					episode_max_reward = add_cfg.episode_max_reward,
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
		assert max_stacks == 1, raise ValueError(f"add_cfg.py max_stacks should be 1, but has {max_stacks}")
		print(f"add.py Simulator initialized, regions={regions}")
		oa = add_prefix(regions=[item for sublist in regions for item in sublist], prefix=prefix)
		oa = oa + [RELOCATED]
		self.prefix = prefix
		self.other_areas = oa
		self.all_areas = [BLOCKS]
		self.all_areas.extend(self.other_areas) 
		self.head = [element for element in self.all_areas if '_H' in element][0]
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
		assert self.num_blocks <= self.max_blocks, raise ValueError(f"number of blocks after adding ({self.num_blocks}) exceeds max ({self.max_blocks})")
		self.correct_record = np.zeros_like(self.goal) # binary record for how many blocks are correct
		self.unit_reward = self.episode_max_reward / sum([self.reward_decay_factor**(b+1) for b in range(len(self.goal))])
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
		assert self.num_blocks <= self.max_blocks, raise ValueError(f"number of blocks after adding ({self.num_blocks}) exceeds max ({self.max_blocks})")
		self.unit_reward = self.episode_max_reward / sum([self.reward_decay_factor**(b+1) for b in range(len(self.goal))])
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
			if SKIP_RELOCATED and area==RELOCATED:
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
		if self.last_active_assembly[new_top_area] != -1 and (BLOCKS in self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0]):
			tmpidx = self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0].index(BLOCKS)
			if self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][1][tmpidx] == self.newblock:
				self.state[-2] = 1 # new block is encoded in new top area
		# update all correct status
		self.state[-4] = 0
		# update newblockid activate status
		self.state[-3] = 0
		if self.last_active_assembly[BLOCKS] ==self.newblock:
			self.state[-3] = 1
		# update goal
		stack = utils.synthetic_readout(parse.assembly_dict, parse.last_active_assembly, parse.head, parse.num_blocks)
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
			if SKIP_RELOCATED and area==RELOCATED:
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
		if self.last_active_assembly[new_top_area] != -1 and (BLOCKS in self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0]):
			tmpidx = self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0].index(BLOCKS)
			if self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][1][tmpidx] == self.newblock:
				self.state[-2] = 1 # new block is encoded in new top area
		# update all correct status
		self.state[-4] = 0
		# update newblockid activate status
		self.state[-3] = 0
		if self.last_active_assembly[BLOCKS] ==self.newblock:
			self.state[-3] = 1
		# update goal
		stack = utils.synthetic_readout(remove.assembly_dict, remove.last_active_assembly, remove.head, remove.num_blocks)
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
			action = ("disinhibit_fiber", BLOCKS, new_top_area)
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			action = ("project_star", None)
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			action = ("disinhibit_fiber", new_top_area, self.head) 
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			action = ("project_star", None)
			final_actions.append( list(self.action_dict.keys())[list(self.action_dict.values()).index(action)] )
			action = ("inhibit_fiber", BLOCKS, new_top_area)
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
		action = ("disinhibit_fiber", BLOCKS, new_top_area)
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
		action = ("inhibit_fiber", BLOCKS, new_top_area)
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
				self.state, self.assembly_dict, self.fiber_state_dict, self.last_active_assembly, self.num_assemblies = utils.synthetic_project(self.state, self.assembly_dict, self.fiber_state_dict, self.last_active_assembly, self.num_assemblies, self.verbose)
				readout = utils.synthetic_readout(self.assembly_dict, self.last_active_assembly, self.head, self.num_blocks)
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
				if self.state[-2]!= 1 and (self.last_active_assembly[new_top_area] != -1) and (BLOCKS in self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0]):
					tmpidx = self.assembly_dict[new_top_area][self.last_active_assembly[new_top_area]][0].index(BLOCKS)
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
			self.last_active_assembly[BLOCKS] = self.newblock
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
		

	def synthetic_project(self):
		verbose = self.verbose
		prev_last_active_assembly = copy.deepcopy(self.last_active_assembly) # {area: idx}
		new_num_assemblies = self.num_assemblies
		print('initial new_num_assemblies', new_num_assemblies) if verbose else 0
		prev_num_assemblies = None
		prev_assembly_dict = copy.deepcopy(self.assembly_dict) # {area: [a_id 0[source a_name[A1, A2], source a_id [a1, a2]], 1[[A3], [a3]], 2[(A4, a4)], ...]}
		iround = 0
		all_visited = False # check if all opened areas are visited
		while (new_num_assemblies != prev_num_assemblies) and (iround <= 5) and (not all_visited): # keep projecting while new assemblies are being created
			# TODO: need to check in what case the loop never ends
			print("-------------------- new project round", iround) if verbose else 0
			prev_num_assemblies = new_num_assemblies # update total number of assemblies

			# generate project map
			receive_from = {} # {area_destination: [area_source1, area_source2, ...]}
			all_visited = False
			opened_areas = {}
			for idx in self.fiber_state_dict.keys(): # check which fibers are open
				if self.state[idx]==1: # if fiber is open
					area1, area2 = self.fiber_state_dict[idx] # get areas on both ends
					if area1 != BLOCKS:
						opened_areas = set([area1]).union(opened_areas)
					if area2 != BLOCKS:
						opened_areas = set([area2]).union(opened_areas)
					# can serve as source only if there is record of assembly in the area
					if (prev_last_active_assembly[area1] != -1) and (area2 != BLOCKS): # blocks area need not receive
						receive_from[area2] = set([area1]).union(receive_from.get(area2, set())) # area1 as source, to area2
					if (prev_last_active_assembly[area2] != -1) and (area1 != BLOCKS):
						receive_from[area1] = set([area2]).union(receive_from.get(area1, set())) # bidirectional, area2 can also be source
			print('receive_from') if verbose else 0
			pprint.pprint(receive_from) if verbose else 0
			print('prev_last_active_assembly\n', prev_last_active_assembly) if verbose else 0
			print('opened areas', opened_areas) if verbose else 0

			# do project
			assembly_dict = copy.deepcopy(prev_assembly_dict) # use assembly dict from last round of project
			last_active_assembly = copy.deepcopy(prev_last_active_assembly) # use last activated assembly from previous round of project
			# {area: [0[[A1, A2], [a1, a2]], 1[[A3], [a3]], 2[(A4, a4)], ...]}
			print('prev_assembly_dict') if verbose else 0
			pprint.pprint(prev_assembly_dict) if verbose else 0
			for destination in sorted(receive_from.keys(), reverse=True):
				# collect sources for this destination area
				sources = list(receive_from[destination])
				sources_permutation = list(itertools.permutations(sources)) # list of tuples 
				print('{} as destination, permutations of sources: {}'.format(destination, sources_permutation)) if verbose else 0
				active_assembly_id_in_destination = -1 

				# check if destination already has assembly connected with sources
				for assembly_idx, assembly_content in enumerate(prev_assembly_dict[destination]):
					# assembly_content: [[A1, A2, ...], [a1, a2, ...]]
					connected_areas = assembly_content[0]
					connected_assembly_ids = assembly_content[1]
					print("\tchecking destination assembly id {}: connected areas{}, connected ids {}".format(assembly_idx, connected_areas, connected_assembly_ids)) if verbose else 0
					if (tuple(connected_areas) in sources_permutation): # if destination has received from the sources
						print("\t\tCandidate?") if verbose else 0
						# check if the assembly ids in sources all match with record
						match = True 
						for A, i in zip(connected_areas, connected_assembly_ids): # go through info in the existing assembly
							if prev_last_active_assembly[A] != i: # if the new source assembly id does not match the existing record
								match = False
								print("\t\tBut ids do not match") if verbose else 0
						if match: # everything match, the connection exists
							active_assembly_id_in_destination = assembly_idx # should be a number >= 0
							print("\t\tMatch!") if verbose else 0
							break # break the search
					if set(sources).issubset(set(connected_areas)): # if sources is a smaller set that overlaps with existing connections
						print("\t\tCandidate (subset)?") if verbose else 0
						match = True
						for A, i in zip(connected_areas, connected_assembly_ids):
							if (A in sources) and (prev_last_active_assembly[A] != i):
								match = False
						if match:
							active_assembly_id_in_destination = assembly_idx
							print("\t\tMatch!") if verbose else 0
							break
				# search if any of the sources is already connected with the latest activated assembly in dest
				if (active_assembly_id_in_destination == -1) and (prev_last_active_assembly[destination] != -1):
					print("\tsearching for partial candidates...") if verbose else 0
					dest_idx = prev_last_active_assembly[destination]
					print('\t\t',dest_idx, prev_assembly_dict[destination]) if verbose else 0
					connected_areas, connected_assembly_ids = prev_assembly_dict[destination][dest_idx][0], prev_assembly_dict[destination][dest_idx][1]
					for source in sources:
						if (source in connected_areas) and (connected_assembly_ids[connected_areas.index(source)] == prev_last_active_assembly[source]):
							# if both area name and index matches
							print("\t\tPartial candidate Match!", source, prev_last_active_assembly[source]) if verbose else 0
							active_assembly_id_in_destination = dest_idx
					# if partial candidate not found, search for non-optimal partial candidate (source connects to a dest assembly that is not currently activated)
					if active_assembly_id_in_destination == -1: 
						print("\tsearching for non-optimal partial candidates...") if verbose else 0
						for source in sources:
							if destination in prev_assembly_dict[source][prev_last_active_assembly[source]][0]:
								source_a_idx = prev_assembly_dict[source][prev_last_active_assembly[source]][0].index(destination)
								active_assembly_id_in_destination = prev_assembly_dict[source][prev_last_active_assembly[source]][1][source_a_idx]
								print("\t\tnon-optimal partial candidate Match! source {} {} --> dest {}".format(source, prev_last_active_assembly[source],active_assembly_id_in_destination)) if verbose else 0
				# search ends
				print("\tsearch ends, active_assembly_id_in_destination:", active_assembly_id_in_destination) if verbose else 0

				# if connection does not exist, create new assembly in dest
				if active_assembly_id_in_destination == -1:
					print("\tcreating new assembly...") if verbose else 0
					# create new assembly in destination [[A1, A2, ...], [a1, a2, ...]]
					assembly_dict[destination].append([sources, [prev_last_active_assembly[S] for S in sources]]) 
					active_assembly_id_in_destination = len(assembly_dict[destination])-1 # new assembly id
					new_num_assemblies += 1 # increment total number of assemblies in brain
					print('\tcreation done.') if verbose else 0
				print('\tactive_assembly_id_in_destination={}'.format(active_assembly_id_in_destination)) if verbose else 0
				
				try:
					assert len(assembly_dict[destination]) > active_assembly_id_in_destination
				except:
					print("assembly_dict[destination]:{}, new_dest_id={}").format(assembly_dict[destination], active_assembly_id_in_destination)

				# add the new assembly info to source areas, update destination assembly dict if necessary
				for source in sources:
					print("\tchecking assembly dict for source...", source) if verbose else 0
					match = False # checks if prev_assembly_dict[source][prev_last_active_assembly[source]] contains any assembly in dest
					for i, (A, a) in enumerate(zip(prev_assembly_dict[source][prev_last_active_assembly[source]][0], prev_assembly_dict[source][prev_last_active_assembly[source]][1])):
						print('\t\tlast active source is connected to: A, a: ', A, a) if verbose else 0
						# check if any of the connected areas from source is destination
						if A==destination: # prev_assembly_dict[source][prev_last_active_assembly[source]] contains any assembly in dest
							match = True 
							if (a!=active_assembly_id_in_destination): # if the last activate source a is connected with wrong dest a
								# replace source a --> destination new a
								updateidx = None
								if destination in assembly_dict[source][prev_last_active_assembly[source]][0]:
									updateidx = assembly_dict[source][prev_last_active_assembly[source]][0].index(destination)
								if updateidx != None:
									assembly_dict[source][prev_last_active_assembly[source]][1][updateidx] = active_assembly_id_in_destination
									print("\t\tsource dict area match dest. Update source dict.") if verbose else 0
								# for symmetry, also check dest dict
								old_dest_id = prev_assembly_dict[source][prev_last_active_assembly[source]][1][i]
								new_dest_id = active_assembly_id_in_destination
								new_source_id = prev_last_active_assembly[source]
								for AA, aa in zip(prev_assembly_dict[destination][old_dest_id][0], prev_assembly_dict[destination][old_dest_id][1]):
									if (AA==source):
										# remove assembly from the destination old area to the new source assembly
										popidx = None
										if AA in assembly_dict[destination][old_dest_id][0]:
											popidx = assembly_dict[destination][old_dest_id][0].index(AA)
											if assembly_dict[destination][old_dest_id][1][popidx]!= new_source_id:
												popidx = None
										if popidx != None:
											assembly_dict[destination][old_dest_id][0].pop(popidx)
											assembly_dict[destination][old_dest_id][1].pop(popidx)
											print("\t\tdest dict old id removed", old_dest_id, AA, aa) if verbose else 0
										# add the new connection from new dest id to new source id
										try:
											if AA not in assembly_dict[destination][new_dest_id][0]:  ## TODO
												assembly_dict[destination][new_dest_id][0].append(AA)
												assembly_dict[destination][new_dest_id][1].append(aa)
												print('\t\tdest dict new id added', new_dest_id, AA, aa) if verbose else 0
										except:
											print("assembly_dict[destination]:{}, new_dest_id={}, source={}, AA={}, aa={}").format(assembly_dict[destination], new_dest_id, source, AA, aa)
										else:
											idx = assembly_dict[destination][new_dest_id][0].index(AA)
											assembly_dict[destination][new_dest_id][1][idx] = aa
											print("\t\tdest dict new id updated", new_dest_id, AA, aa) if verbose else 0
								
					# check if new dest dict needs to be updated wrt source
					popidx = None
					for j, (AA, aa) in enumerate(zip(assembly_dict[destination][active_assembly_id_in_destination][0], assembly_dict[destination][active_assembly_id_in_destination][1])):
						print("\t\tchecking if dest dict needs to be updated wrt source") if verbose else 0
						if (AA==source) and (aa!=prev_last_active_assembly[source]): # if new dest is connected with a wrong id, remove wrong id
							old_source_id = assembly_dict[destination][active_assembly_id_in_destination][1][j]
							assembly_dict[destination][active_assembly_id_in_destination][1][j] = prev_last_active_assembly[source]
							print('\t\tdest dict updated, to new source a') if verbose else 0
							if destination in assembly_dict[source][old_source_id][0]:
								popidx = assembly_dict[source][old_source_id][0].index(destination)
								# if assembly_dict[source][old_source_id][1][popidx]!=prev_assembly_dict[source][prev_last_active_assembly[source]][1][i]:
								if assembly_dict[source][old_source_id][1][popidx]!=active_assembly_id_in_destination:
									print("\t\treset") if verbose else 0
									popidx = None
					if popidx != None:
						assembly_dict[source][old_source_id][0].pop(popidx)
						assembly_dict[source][old_source_id][1].pop(popidx)
						print("\t\tsource dict removed", old_source_id, popidx) if verbose else 0
						print("\t\tnew source dict", assembly_dict[source]) if verbose else 0
					
					# if prev_assembly_dict[source][prev_last_active_assembly[source]] is not connected with any assembly in dest 
					if not match: # append new dest to source dict 
						if destination not in assembly_dict[source][prev_last_active_assembly[source]][0]:
							assembly_dict[source][prev_last_active_assembly[source]][0].append(destination)
							assembly_dict[source][prev_last_active_assembly[source]][1].append(active_assembly_id_in_destination)
							print("\t\tsource dict added dest a") if verbose else 0
					# if dest is not connected with source at all, add the source to dest dict
					if source not in assembly_dict[destination][active_assembly_id_in_destination][0]:
						assembly_dict[destination][active_assembly_id_in_destination][0].append(source)
						assembly_dict[destination][active_assembly_id_in_destination][1].append(prev_last_active_assembly[source])
						print("\t\tdest dict did not have source a, added source a") if verbose else 0
				
					# check every assembly in the source
					visited = 0
					popa = []
					popidx = []
					for i, (Alist, alist) in enumerate(assembly_dict[source]):
						for j, (A, a) in enumerate(zip(Alist, alist)):
							if A==destination and a==active_assembly_id_in_destination:
								visited += 1
								if i != prev_last_active_assembly[source]:
									# if assembly i in source is connected with the destination,
									# but the destination is not connected with it, need to delete the connection
									visited -= 1
									popa.append(i)
									popidx.append(j)
					if len(popa)>0:
						for i, j in zip(popa[::-1], popidx[::-1]):
							print('\t\tsource dict popped assembly {} connection {}, in {}'.format(i,j, assembly_dict[source][i])) if verbose else 0
							assembly_dict[source][i][0].pop(j)
							assembly_dict[source][i][1].pop(j)
					# if no assembly connecting from source last active to destination active
					if visited==0 and (destination not in assembly_dict[source][prev_last_active_assembly[source]][0]): 
						# add the new connection
						assembly_dict[source][prev_last_active_assembly[source]][0].append(destination)
						assembly_dict[source][prev_last_active_assembly[source]][1].append(active_assembly_id_in_destination)
						print('\t\tsource dict added destination') if verbose else 0
										
 				# update the last activated assembly in destination
				last_active_assembly[destination] = active_assembly_id_in_destination
				print('\ttotal number of assemblies', new_num_assemblies) if verbose else 0

				# remove dest from opened area
				opened_areas.remove(destination)
				if len(opened_areas)==0:
					all_visited = True
					print('\tall_visited=True') if verbose else 0

			# project finishes, update assembly dict
			prev_assembly_dict = assembly_dict
			prev_last_active_assembly = last_active_assembly
			iround += 1

		# project rounds finish
		self.num_assemblies = new_num_assemblies
		self.assembly_dict = prev_assembly_dict
		self.last_active_assembly = prev_last_active_assembly
		print("\n--------------------- end of project rounds, num_assemblies:", self.num_assemblies) if verbose else 0
		print("assembly_dict") if verbose else 0
		pprint.pprint(self.assembly_dict) if verbose else 0
		print('last_active_assembly\n', self.last_active_assembly) if verbose else 0
		return 


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
			if add_cfg.skip_relocated and area1==RELOCATED:
				continue
			last_active_assembly[area1] = -1
			assembly_dict[area1] = [] # will become # {area: [a_id 0[source a_name[A1, A2], source a_id [a1, a2]], 1[[A3], [a3]], 2[(A4, a4)], ...]}
			for area2 in self.all_areas:
				if add_cfg.skip_relocated and area2==RELOCATED:
					continue
				if area1 == area2: # no need for fiber to the same area itself
					continue
				if (("_H" in area1) and (area2==BLOCKS)) or (("_H" in area2) and (area1==BLOCKS)):
					# for HEADS fibers, only consider N0, N1, N2, do not consider connection with BLOCKS
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
				if ('_H' in area1) or (area2==BLOCKS):
					state_vec.append(0)
					dictionary[action_idx] = ([state_vector_idx], 1)
					stimulate_state_dict[state_vector_idx] = (area1, area2)
					state_vector_idx += 1
					action_idx += 1
				elif ('_H' in area2) or (area1==BLOCKS):
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
		# activate new block id in BLOCKS, newblockid activation status initialized as 0
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
		assembly_dict[BLOCKS] = [[[],[]] for _ in range(self.num_blocks)] 
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
			if add_cfg.skip_relocated and area1==RELOCATED:
				continue
			for area2 in self.all_areas:
				if add_cfg.skip_relocated and area2==RELOCATED:
					continue # skip relocated area
				if area1 == area2: 
					continue # no need for fiber to the same area itself
				if (("_H" in area1) and (area2==BLOCKS)) or (("_H" in area2) and (area1==BLOCKS)):
					continue # for HEADS fibers, only consider N0, N1, N2, do not consider connection with BLOCKS
				if [area1, area2] in area_pairs:
					continue  # skip already included pairs
				dictionary[idx] = ("disinhibit_fiber", area1, area2)
				idx += 1
				dictionary[idx] = ("inhibit_fiber", area1, area2)
				idx += 1
				if ('_H' in area1) or (area2==BLOCKS): # TODO: why need stimulate? replace every stimulate by project?
					dictionary[idx] = ("stimulate", area1, area2)
					idx += 1
				elif ('_H' in area2) or (area1==BLOCKS):
					dictionary[idx] = ("stimulate", area2, area1)
					idx += 1
				# update area_pairs 
				area_pairs.append([area1, area2])
				area_pairs.append([area2, area1])
		# project star
		dictionary[idx] = ("project_star", None)
		idx += 1
		# activate new block id in BLOCKS
		dictionary[idx] = ("activate_block", None)
		idx += 1
		return dictionary

