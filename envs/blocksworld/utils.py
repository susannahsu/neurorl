
def top(assembly_dict, last_active_assembly, head):
	'''
	Return: top area name, top area assembly id, top block idx
	'''
	if last_active_assembly[head] == -1:
		return None, None, None
	area, a, bid = None, None, None
	candidate_areas, candidate_as = assembly_dict[head][last_active_assembly[head]]
	for area, a in zip(candidate_areas, candidate_as):
		if BLOCKS in assembly_dict[area][a][0]:
			idx = assembly_dict[area][a][0].index(BLOCKS)
			bid = assembly_dict[area][a][1][idx]
			area = area
			a = a
			bid = bid
	return area, a, bid

def is_last_block(assembly_dict, head, top_area, top_area_a):
	'''
	Return True if the top_area_a assembly in top_area is the last block in the chain
	'''
	if top_area==None:
		return True
	# check if the block encoded in top area assembly a is the only block in the brain
	for A in assembly_dict[top_area][top_area_a][0]:
		if A != BLOCKS and A != head: # assembly is connected with other node areas
			if ('_N0' in top_area and '_N1' in A) or ('_N1' in top_area and '_N2' in A) or ('_N2' in top_area and '_N0' in A):
				return False
	return True

def all_fiber_closed(state, fiber_state_dict):
	'''
	Check the state vector and return True if all fibers are closed, False otherwise.
	'''
	return np.all([state[i]==0 for i in fiber_state_dict.keys()])


def synthetic_readout(assembly_dict, last_active_assembly, head, readout_length):
	'''
	Read out the current chain of blocks representation from the brain.
		Assuming the brain only represents max 1 stack. 
	Return a list of blocks (chained from head/top to bottom).
	'''
	readout = [] # list of blocks (assuming only 1 stack in the brain)
	if len(assembly_dict[head])==0 or last_active_assembly[head]==-1:
		return [None] * readout_length # no assembly in head, return [None, None, ...]
	# if assembly exists in head, get the first node connected with head
	areas_from_head, aidx_from_head = assembly_dict[head][last_active_assembly[head]]
	prev_area, prev_area_a = head, last_active_assembly[head]
	area, area_a = None, None # initiate next area to decode from
	if len(areas_from_head) != 0 and len(aidx_from_head)!= 0: # if head assembly is connected with a node area
		area, area_a = areas_from_head[-1], aidx_from_head[-1] # next area to read from
	for iblock in range(readout_length): 
		if area==None and area_a==None: # if current area is not available
			readout.append(None)
			continue
		elif BLOCKS in assembly_dict[area][area_a][0]: # if current area is connected with BLOCKS, decode
			ba = assembly_dict[area][area_a][0].index(BLOCKS)
			bidx = assembly_dict[area][area_a][1][ba]
			readout.append(bidx)
		else: # current area is not connected with BLOCKS
			readout.append(None)
		# find the next area to decode
		areas_from_area, aidx_from_area = assembly_dict[area][area_a] # assemblies connected with current area
		new_area, new_area_a = None, None 
		for A, a in zip(areas_from_area, aidx_from_area): # iterate through current assembly's connections
			if (A != BLOCKS) and (A != prev_area): # next area to decode
				if ('_N0' in area and '_N1' in A) or ('_N1' in area and '_N2' in A) or ('_N2' in area and '_N0' in A):
					new_area, new_area_a = A, a
					break
		prev_area, prev_area_a = area, area_a
		area, area_a = new_area, new_area_a
	return readout


def check_prerequisite(arr, k, value=1):
	'''
	Return True if the first k elements in arr are all equal to value.
	'''
	if len(arr) < k:
		raise ValueError(f"!!!Warning, in utils.check_prerequisite, k={k} is out of range {len(arr)}")
	for i in range(k):
		if arr[i] != value:
			return False
	return True


def calculate_readout_reward(readout, goal, correct_record, unit_reward, reward_decay_factor):
	'''
	Calculate score by comparing current readout with goal. 
		Reward decays from top to bottom block.
		Reward a block only if all its previous (higher) blocks are correct.
	Return: reward, all_correct (boolean), correct_record (binary array)
	'''
	score = 0
	num_correct = 0
	prerequisite = [0 for _ in range(len(goal))] # the second block will received reward only if first block is also readout correctly
	for jblock in range(len(goal)): # read from top to bottom
		if readout[jblock] == goal[jblock]: # block match
			prerequisite[jblock] = 1
			num_correct += 1
			# only reward new correct blocks in this episode
			if correct_record[jblock]==0 and check_prerequisite(prerequisite, jblock):
				score += unit_reward * (reward_decay_factor**(jblock+1)) # reward decays by position: first block is most rewarding
				correct_record[jblock] = 1 # set the block record to 1, record correct history for episode
	all_correct = (num_correct > 0) and (num_correct == len(goal))
	return score, all_correct, correct_record


def synthetic_project(state, assembly_dict, fiber_state_dict, last_active_assembly, num_assemblies, verbose=False, max_project_round=5):
	prev_last_active_assembly = copy.deepcopy(last_active_assembly) # {area: idx}
	prev_assembly_dict = copy.deepcopy(assembly_dict) # {area: [a_id 0[source a_name[A1, A2], source a_id [a1, a2]], 1[[A3], [a3]], 2[(A4, a4)], ...]}
	new_num_assemblies = num_assemblies # current total number of assemblies in the brain
	prev_num_assemblies = None
	iround = 0 # current projection round
	all_visited = False # check if all opened areas are visited
	print(f'initial num_assemblies={new_num_assemblies}') if verbose else 0
	# Keep projecting while new assemblies are being created and other criteria hold. TODO: check endless loop condition
	while (new_num_assemblies != prev_num_assemblies) and (iround <= max_project_round) and (not all_visited): 
		print(f"-------------------- new project round {iround}") if verbose else 0
		# Generate project map
		prev_num_assemblies = new_num_assemblies # update total number of assemblies
		receive_from = {} # {destination_area: [source_area1, source_area2, ...]}
		all_visited = False # whether all opened areas are visited 
		opened_areas = {} # open areas in this round
		for idx in fiber_state_dict.keys(): # get opened fibers from state vector
			if state[idx]==1: # fiber is open
				area1, area2 = fiber_state_dict[idx] # get areas on both ends
				if area1 != BLOCKS: # skip if this is blocks area
					opened_areas = set([area1]).union(opened_areas)
				if area2 != BLOCKS:
					opened_areas = set([area2]).union(opened_areas)
				# check eligibility of areas, can only be source if there exists last active assembly in the area
				if (prev_last_active_assembly[area1] != -1) and (area2 != BLOCKS): # blocks area cannot receive
					receive_from[area2] = set([area1]).union(receive_from.get(area2, set())) # area1 as source, to destination area2
				if (prev_last_active_assembly[area2] != -1) and (area1 != BLOCKS): # bidirectional, area2 can also be source
					receive_from[area1] = set([area2]).union(receive_from.get(area1, set())) # area2 source, area1 destination
		print(f'prev_last_active_assembly: {prev_last_active_assembly}, opened_areas: {opened_areas}, receive_from: {pprint.pprint(receive_from)}, prev_assembly_dict: {pprint.pprint(prev_assembly_dict)}') if verbose else 0
		# Do project
		assembly_dict = copy.deepcopy(prev_assembly_dict) # use assembly dict from prev round of project
		last_active_assembly = copy.deepcopy(prev_last_active_assembly) # use last activated assembly from prev round of project
		for destination in receive_from.keys(): # process every destination area
			sources = list(receive_from[destination]) # all input sources
			sources_permutation = list(itertools.permutations(sources)) # permutation of the sources, list of tuples
			active_assembly_id_in_destination = -1 # assume no matching connection exists by default
			print(f'{destination} as destination, permutations of sources: {sources_permutation}') if verbose else 0
			# search if destination area already has an assembly connected with input sources
			for assembly_idx, assembly_content in enumerate(prev_assembly_dict[destination]): # check existing assembly in dest one by one
				connected_areas, connected_assembly_ids = assembly_content # assembly_content: [[Area1, Area2, ...], [assembly1, assembly2, ...]]
				print(f"\tchecking destination assembly id {assembly_idx}: connected areas {connected_areas}, connected ids {connected_assembly_ids}") if verbose else 0
				if (tuple(connected_areas) in sources_permutation): # if destination assembly connects with all source areas (only area names match)
					print("\t\tCandidate?") if verbose else 0
					match = True # now need to check if the assembly ids all match too
					for A, i in zip(connected_areas, connected_assembly_ids): # go through each area name and assembly id connected with this assembly
						if prev_last_active_assembly[A] != i: # assembly id does not match
							match = False
							print("\t\tBut ids do not match") if verbose else 0
					if match: # everything match, the exact connection from all input sources to destination already exists
						active_assembly_id_in_destination = assembly_idx 
						print("\t\tMatch!") if verbose else 0
						assert active_assembly_id_in_destination >= 0, raise ValueError(f"\t\tFound matching connection between source and dest, but assembly_idx in dest is {active_assembly_id_in_destination}")
						break # exit the search
				if set(sources).issubset(set(connected_areas)): # if dest assembly connects with all source areas (only names match) and other areas
					print("\t\tCandidate (subset)?") if verbose else 0 # TODO: merge this if with previous if?
					match = True # now check if the assembly ids all match sources too 
					for A, i in zip(connected_areas, connected_assembly_ids): # go through each area name and assembly id connected with this assembly
						if (A in sources) and (prev_last_active_assembly[A] != i):
							match = False # assembly id does not match
					if match: # everything match, there is connection from all input sources (and some other areas) to destination
						active_assembly_id_in_destination = assembly_idx
						print("\t\tMatch!") if verbose else 0
						assert active_assembly_id_in_destination >= 0, raise ValueError(f"\t\tFound matching connection between source and dest, but assembly_idx in dest is {active_assembly_id_in_destination}")
						break
			# no existing connection match, search if any of the sources is already connected with the latest activated assembly in dest
			if (active_assembly_id_in_destination == -1) and (prev_last_active_assembly[destination] != -1):
				print("\tsearching for partial candidates...") if verbose else 0
				dest_idx = prev_last_active_assembly[destination]
				print(f'\t\t dest_idx={dest_idx}, prev_assembly_dict[destination]={prev_assembly_dict[destination]}') if verbose else 0
				connected_areas, connected_assembly_ids = prev_assembly_dict[destination][dest_idx][0], prev_assembly_dict[destination][dest_idx][1]
				for source in sources:
					if (source in connected_areas) and (connected_assembly_ids[connected_areas.index(source)] == prev_last_active_assembly[source]):
						# if both area name and index matches
						print(f"\t\tPartial candidate Match! {source} {prev_last_active_assembly[source]}") if verbose else 0
						active_assembly_id_in_destination = dest_idx
				# if partial candidate not found, search for non-optimal partial candidate (source connects to a dest assembly that is not currently activated)
				if active_assembly_id_in_destination == -1: 
					print("\tsearching for non-optimal partial candidates...") if verbose else 0
					for source in sources:
						if destination in prev_assembly_dict[source][prev_last_active_assembly[source]][0]:
							source_a_idx = prev_assembly_dict[source][prev_last_active_assembly[source]][0].index(destination)
							active_assembly_id_in_destination = prev_assembly_dict[source][prev_last_active_assembly[source]][1][source_a_idx]
							print(f"\t\tnon-optimal partial candidate Match! source {source} {prev_last_active_assembly[source]} --> dest {active_assembly_id_in_destination}") if verbose else 0
			print(f"\tsearch ends, active_assembly_id_in_destination={active_assembly_id_in_destination}") if verbose else 0
			# still no existing connection match, create new assembly in destination
			if active_assembly_id_in_destination == -1:
				assembly_dict[destination].append([sources, [prev_last_active_assembly[S] for S in sources]]) # [[A1, A2, ...], [a1, a2, ...]]
				active_assembly_id_in_destination = len(assembly_dict[destination])-1 # new assembly id
				new_num_assemblies += 1 # increment total number of assemblies in brain
				print(f'\tcreated bew assembly in destination, new assembly id {active_assembly_id_in_destination}') if verbose else 0
			assert len(assembly_dict[destination]) > active_assembly_id_in_destination, raise ValueError(f"new_dest_id={active_assembly_id_in_destination} out of bound of assembly_dict[destination]: {assembly_dict[destination]}")
			# reflect the newly activated destination assembly in source areas, update destination assembly dict if necessary
			for source in sources:
				print(f"\tchecking assembly dict for source {source}...") if verbose else 0
				match = False # checks if prev_assembly_dict[source][prev_last_active_assembly[source]] contains any assembly in dest
				for i, (A, a) in enumerate(zip(prev_assembly_dict[source][prev_last_active_assembly[source]][0], prev_assembly_dict[source][prev_last_active_assembly[source]][1])):
					print(f'\t\tlast active source is connected to: A={A}, a={a}') if verbose else 0
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
										print(f"\t\tdest dict old id removed, old_dest_id={old_dest_id}, AA={AA}, aa={aa}") if verbose else 0
									# add the new connection from new dest id to new source id
									try:
										if AA not in assembly_dict[destination][new_dest_id][0]:  ## TODO
											assembly_dict[destination][new_dest_id][0].append(AA)
											assembly_dict[destination][new_dest_id][1].append(aa)
											print(f'\t\tdest dict new id added, new_dest_id={new_dest_id}, AA={AA}, aa={aa}') if verbose else 0
									except:
										print(f"assembly_dict[destination]:{assembly_dict[destination]}, new_dest_id={new_dest_id}, source={source}, AA={AA}, aa={aa}")
									else:
										idx = assembly_dict[destination][new_dest_id][0].index(AA)
										assembly_dict[destination][new_dest_id][1][idx] = aa
										print(f"\t\tdest dict new id updated, new_dest_id={new_dest_id}, AA={AA}, aa={aa}") if verbose else 0
				# TODO: check if this block needs indent to if A==destination
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
					print(f"\t\tsource dict removed old_source_id {old_source_id}, popidx={popidx}, new source dict={assembly_dict[source]}") if verbose else 0
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
								# if assembly i in source is connected with the dest, but dest is not connected with i, need to delete the connection
								visited -= 1
								popa.append(i)
								popidx.append(j)
				if len(popa)>0:
					for i, j in zip(popa[::-1], popidx[::-1]):
						print(f'\t\tsource dict popped assembly={i} connection={j}, in {assembly_dict[source][i]}') if verbose else 0
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
			print(f'\ttotal number of assemblies={new_num_assemblies}') if verbose else 0
			# remove dest from opened area
			opened_areas.remove(destination)
			if len(opened_areas)==0:
				all_visited = True
				print('\tall_visited=True') if verbose else 0
		# Project completes, update assembly dict
		prev_assembly_dict = assembly_dict
		prev_last_active_assembly = last_active_assembly
		iround += 1
	# All project rounds complete
	num_assemblies = new_num_assemblies
	assembly_dict = prev_assembly_dict
	last_active_assembly = prev_last_active_assembly
	print(f"\n--------------------- end of project rounds, num_assemblies={num_assemblies}, last_active_assembly={last_active_assembly}, assembly_dict: ") if verbose else 0
	pprint.pprint(assembly_dict) if verbose else 0
	return state, assembly_dict, fiber_state_dict, last_active_assembly, num_assemblies

