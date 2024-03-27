from envs.blocksworld.AC.brain import *

BLOCKS = "BLOCKS"
FIRST = "FIRST"
SECOND = "SECOND"
ABOVE = "ABOVE"
RELATION = "RELATION"
RELOCATED = "RELOCATED" # this region saves which blocks have been moved from the TABLE to some stack

class BlocksBrain:
	def __init__(self, blocks_number, other_areas, p=0.1, eak=10, nean=10000, neak=100, db=0.2):
		self.brain = Brain(p=p)
		self.num_blocks = blocks_number # max number of blocks the BLOCKS explicit area can represent
		self.explicit_area_k = eak
		self.non_explicit_area_n = nean
		self.non_explicit_area_k = neak
		self.explicit_area_n = blocks_number * eak
		self.all_areas = [BLOCKS]
		self.all_areas.extend(other_areas) 
		self.brain.add_explicit_area(area_name=BLOCKS, n=self.explicit_area_n, k=eak, beta=db)
		for area_name in other_areas:
			self.brain.add_nonexplicit_area(area_name=area_name, n=nean, k=neak, beta=db)
		self.fiber_states = {} 
		for from_area in self.all_areas:
			self.fiber_states[from_area] = {}
			for to_area in self.all_areas:
				self.fiber_states[from_area][to_area] = set([0]) # initialize every fiber with a lock
		self.area_states = {}
		for area in self.all_areas:
			self.area_states[area] = set([0]) # initialize each area with a lock

	def print_info(self):
		print("\nBlocksBrain info: \
				\n\tnum_blocks={},\
				\n\texplicit_area_k={},\
				\n\tnon_explicit_area_n={},\
				\n\tnon_explicit_area_k={},\
				\n\tall_areas={},\
				\n\tfiber_states={},\
				\n\tarea_states={},".format(self.num_blocks, self.explicit_area_k,
									self.non_explicit_area_n, self.non_explicit_area_k,
									self.all_areas, self.fiber_states, self.area_states))
		self.brain.print_info()

	def inhibit_area(self, area_name, lock=0):
		self.area_states[area_name].add(lock)

	def inhibit_fiber(self, area1, area2, lock=0):
		self.fiber_states[area1][area2].add(lock)
		self.fiber_states[area2][area1].add(lock)

	def disinhibit_area(self, area_name, lock=0):
		self.area_states[area_name].discard(lock)

	def disinhibit_fiber(self, area1, area2, lock=0):
		self.fiber_states[area1][area2].discard(lock)
		self.fiber_states[area2][area1].discard(lock)

	def disinhibit_areas(self, area_names, lock=0):
		for area_name in area_names:
			self.disinhibit_area(area_name=area_name, lock=lock)

	def disinhibit_fibers(self, area_pairs, lock=0):
		for pair in area_pairs:
			self.disinhibit_fiber(area1=pair[0], area2=pair[1], lock=lock)

	def disinhibit_all_fibers(self, area_names, lock=0):
		for area1 in area_names:
			for area2 in area_names:
				if area1 != area2:
					self.disinhibit_fiber(area1=area1, area2=area2, lock=lock)

	def inhibit_areas(self, area_names, lock=0):
		for area_name in area_names:
			self.inhibit_area(area_name=area_name, lock=lock)

	def inhibit_fibers(self, area_pairs, lock=0):
		for pair in area_pairs:
			self.inhibit_fiber(area1=pair[0], area2=pair[1], lock=lock)

	def inhibit_all_fibers(self, area_names, lock=0):
		for area1 in area_names:
			for area2 in area_names:
				if area1 != area2:
					self.inhibit_fiber(area1=area1, area2=area2, lock=lock)

	def activate_block(self, index):
		# index should start from 0
		assert 0<=index<self.num_blocks, "index should be in [{}~{}]".format(0, self.num_blocks-1)
		area = self.brain.areas[BLOCKS]
		k = area.k
		assembly_start = index*k 
		area.winners = np.array(range(assembly_start, assembly_start+k))
		area.fix_assembly()

	def activate_assembly(self, index, activation_area):
		# index should start from 0
		assert activation_area in self.brain.areas.keys(), \
				"activation_area {} does not exist in brain.areas".format(activation_area) 
		assert 0<=index < self.brain.areas[activation_area].n // self.brain.areas[activation_area].k, \
				"index should be in [{}~{}]".format(0, self.brain.areas[activation_area].n // self.brain.areas[activation_area].k-1)
		area = self.brain.areas[activation_area]
		k = area.k
		assembly_start = index*k 
		area.winners = np.array(range(assembly_start, assembly_start+k))
		area.fix_assembly()

	def get_project_map(self, verbose=False):
		project_map = defaultdict(set)
		for area1 in self.all_areas:
			if verbose:
				print("area1: {}".format(area1))
			as1 = self.area_states.get(area1, {})
			if not as1: # area1 needs to be unlocked/disinhibited, no lock in set
				if verbose:
					print("In get_project_map as1 empty")
				for area2 in self.all_areas: 
					if (area1 != BLOCKS) or (area2 != BLOCKS): # BLOCKS cannot project to BLOCKS
						if verbose:
							print("In get_project_map area1={}, area2={}".format(area1, area2))
						as2 = self.area_states.get(area2, {})
						if not as2: # area2 needs to be unlocked
							fs1 = self.fiber_states.get(area1, {})
							if not fs1.get(area2, {}): # fiber between area1 and 2 needs to be unlocked
								if len(self.brain.areas[area1].winners)>0: # area1 needs to have winners to project to area2
									project_map[area1].add(area2)
								if len(self.brain.areas[area2].winners)>0: # area2 needs to have winners to project to itself
									# TODO: this may include BLOCKS -> BLOCKS? 
									# TODO: this may allow area2 -> area2 even if the self connection fiber is locked?
									project_map[area2].add(area2)
		return project_map

	def __fix_assemblies_for_project(self, project_map):
		for area in project_map.keys():
			if (not (BLOCKS in project_map.keys())) or (not(area in project_map[BLOCKS])):
				# if area is not receiving from BLOCKS, or if BLOCKS is not projecting
				self.brain.areas[area].fix_assembly()
			elif area != BLOCKS:
				# if area is receiving from BLOCKS
				self.brain.areas[area].unfix_assembly()
				# wipe winners so that area does not project to other areas or itself
				self.brain.areas[area].winners = np.array([]) 

	def project_star(self, project_rounds=30, verbose=False):
		project_map = self.get_project_map()
		# prepare for first round of projection, prevent other projections except from BLOCKS 
		self.__fix_assemblies_for_project(project_map) 
		# __fix_assemblies_for_project might reset the winners of an area to empty, so recompute project_map.
		project_map = self.get_project_map()
		print("Got project map: {}".format(project_map)) if verbose else 0
		for i in range(project_rounds):
			project_map = self.get_project_map()
			self.brain.project(stim_to_area={}, area_to_area=project_map) 
			if verbose:
				project_map = self.get_project_map()
				print("Got project map: {}".format(project_map))

	def test_project(self, verbose=False):
		# Temporarily disable plasticity so this projection doesn't change connectomes.
		self.brain.no_plasticity = True
		project_map = self.get_project_map()
		# prepare for first round of projection, prevent other projections except from BLOCKS 
		self.__fix_assemblies_for_project(project_map)
		# __fix_assemblies_for_project might reset the winners of an area to empty, so recompute project_map.
		project_map = self.get_project_map()
		for area in project_map.keys():
			if area in project_map[area]:
				project_map[area].remove(area) # remove recurrent projection
		print("\tProjecting in test_project {}".format(project_map)) if verbose else 0
		self.brain.project(stim_to_area={}, area_to_area=project_map)
		self.brain.no_plasticity = False

	def __set_to_vector(self, S):
		return np.array(list(S))

	def is_assembly(self, area_name, min_overlap=0.75, verbose=False):
		'''
			test if the new winners activated by sudo project (no plasticity) 
			is the same winners as the assembly learned through project star (multiple iterations and weight update)
		'''
		print("\tExecuting is_assembly on {}".format(area_name)) if verbose else 0
		print("\tStates of {}: {}".format(area_name, self.area_states.get(area_name, {}))) if verbose else 0
		if len(self.area_states.get(area_name, {})) > 0: # if area is locked/inhibited
			return False
		area = self.brain.areas[area_name]
		if len(area.winners)==0: # if no winner
			return False
		self.brain.no_plasticity = True # just doing a sudo project testing assembly, no need to update
		winners_before = set(area.winners.tolist())
		threshold = min_overlap * area.k
		project_map = {}
		project_map[area_name] = set([area_name])
		self.brain.project({}, project_map)
		winners_after = set(area.winners.tolist())
		self.brain.no_plasticity =  False
		# Restore previous winners (so testing for assembly does not affect system).
		area.winners = self.__set_to_vector(winners_before)
		if len(winners_before.intersection(winners_after)) >= threshold:
			return True # there is a stable assembly being activated after project
		return False

	def get_block_index(self, area_name=BLOCKS, min_overlap=0.75, verbose=False):
		"""
		Returns the index of the active block assembly in `area_name`. 
		If none of the `blocks` is (sufficiently) active, returns None
		"""
		area = self.brain.areas[area_name]
		assert area_name == BLOCKS, "Error: getting block index from a non-BLOCKS area {}".format(area_name)
		if len(area.winners) == 0:
			print("\tWarning: Cannot get index (block) because no assembly in {}, return None".format(area_name)) if verbose else 0
			return None
		winners = set(area.winners.tolist())
		area_k = area.k
		threshold = min_overlap * area_k
		for block_index in range(self.num_blocks):
			block_assembly_start = block_index * area_k
			block_assembly_end = block_assembly_start + area_k

			overlap_count = sum(block_assembly_start <= w < block_assembly_end for w in winners)
			if overlap_count >= threshold:
				return block_index

		print("\tWarning: no block found when calling get_block_index(), returning None") if verbose else 0
		return None





















