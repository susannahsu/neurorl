from envs.blocksworld.AC.blocks_brain import *

# The number of node areas to be used to represent a block world instance
MAX_NODES_AREAS = 3
# The maximum number of stacks in the block world instance
MAX_STACKS = 5 # TODO: need to change it from 1 back to 5 when running test.py

''' 
The "input" node areas. They are of the form " I2_N3 ", 
which means "input stack number 2, node area number 3";
or " G1_N2 ", 
which means "goal stack number 2, node area number 3". 
The prefix will be given later and can be either 
	"I" ("input"), "G" ("goal"), "B" ("building").
'''
node_areas = []
for j in range(MAX_STACKS):
	node_areas_stack_j = []
	for k in range(MAX_NODES_AREAS):
		node_areas_stack_j.append(str(j)+"_N"+str(k))
	node_areas.append(node_areas_stack_j)
NODES = node_areas

'''
H stands for "heads" and stores the top block of each stack. 
The prefix will be given later and can be either 
	"I" ("input"), "G" ("goal"), "B" ("building").
'''
heads = []
for j in range(MAX_STACKS):
	heads.append(str(j)+"_H")
HEADS = heads

'''
REGIONS collects all regions. 
It is a vector of vectors of strings, one for each stack. 
The prefix will be given later and can be either 
	"I" ("input"), "G" ("goal"), "B" ("building").
'''
regions = []
for j in range(MAX_STACKS):
	regions_stack_j = NODES[j] + [HEADS[j]]
	regions.append(regions_stack_j)
REGIONS = regions
print("Global MAX_STACKS", MAX_STACKS)
print("Global REGIONS", REGIONS)

def add_prefix(regions, prefix):
	new_regions = []
	for area in regions:
		new_regions.append(prefix + area)
	return new_regions


def is_above(blocks, query_a, query_b, p, eak, nean, neak, db):
	'''
	test if query_a is above query_b
	Parameters
	- blocks: array of blocks (top->bottom). permutation of number between 0 and number of blocks-1
	- query_a: first block of the query (has to be one of the blocks in the stack)
	- query_b: second block of the query (has to be one of the blocks in the stack)
	- p: erdos-renyi parameter
	- eak: k for explicit areas
	- nean: n for non explicit areas
	- neak: k for non explicit areas
	- db: defaul plasticity
	Example: is_above([1,2,3,4],1,4,0.1,10,10000,100,0.2)
	This is the hello world program in the paper
	'''
	blocks_number = len(blocks)
	blocks_brain = BlocksBrain(blocks_number=blocks_number, 
								other_areas=[FIRST, SECOND, ABOVE, RELATION],
								p=p, eak=eak, nean=nean, neak=neak, db=db)
	# form stable assemblies for a and b in FIRST and SECOND respectively
	# prepared for later for loop iteration and comparison
	blocks_brain.disinhibit_area(area_name=BLOCKS, lock=0)
	blocks_brain.disinhibit_area(area_name=FIRST, lock=0)
	blocks_brain.disinhibit_fiber(area1=BLOCKS, area2=FIRST, lock=0)
	
	blocks_brain.activate_block(index=query_a)
	blocks_brain.project_star()
	
	blocks_brain.inhibit_area(area_name=FIRST, lock=0)
	blocks_brain.disinhibit_area(area_name=SECOND, lock=0)
	blocks_brain.disinhibit_fiber(area1=BLOCKS, area2=SECOND, lock=0)

	blocks_brain.activate_block(index=query_b)
	blocks_brain.project_star()

	for block in blocks: # checking blocks in the array one by one
		# perform sudo project (no plasticity) to test if current block is one of a or b
		blocks_brain.inhibit_area(area_name=SECOND, lock=0)
		blocks_brain.disinhibit_area(area_name=FIRST, lock=0)
		blocks_brain.activate_block(index=block)
		project_map = {}
		project_map[BLOCKS] = set([BLOCKS, FIRST])
		blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)
		if blocks_brain.is_assembly(area_name=FIRST): 
			# if current block matches a
			# it means that a appears earlier than b in the array
			return True
		blocks_brain.inhibit_area(area_name=FIRST, lock=0)
		blocks_brain.disinhibit_area(area_name=SECOND, lock=0)
		blocks_brain.activate_block(index=block)
		project_map = {}
		project_map[BLOCKS] = set([BLOCKS, SECOND])
		blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)
		if blocks_brain.is_assembly(area_name=SECOND): 
			# if current block matches b
			# it means b appears earlier than a in the array
			return False


def parse(blocks_brain, stacks, prefix, verbose=False, project_rounds=50):
	'''
	Parameters
	- blocks_brain:: a brain (type BlocksBrain) 
	                 created with a number of blocks which is at least as big as the number of blocks we want to parse
	                 and with the regions with the given prefix
	- stacks: array (max length MAX_STACKS) of arrays of blocks, 
			the lasts being integer from 0 to number of blocks-1
	- p: erdos-renyi parameter
	- eak: k for explicit areas
	- nean: n for non explicit areas
	- neak: k for non explicit areas
	- db: defaul plasticity
	- prefix: string "I", "G", "B", or "T" whether the areas we work on are the input areas, the goal areas, the building areas, or the table areas.
	Example: parse!(blocks_brain,[[1,2,3,4],[5,6]],"I")
	'''
	stacks_number = len(stacks)
	blocks_number = 0
	for j in range(stacks_number):
		blocks_number += len(stacks[j])
	# add prefix to regions
	working_regions = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)

	# parse each stack 
	for j in range(stacks_number):
		blocks = stacks[j]
		# defining locally the regions involved to parse stack number j
		head = add_prefix(regions=[HEADS[j]], prefix=prefix)[0]
		nodes = add_prefix(regions=NODES[j], prefix=prefix)

		# global pre-actions
		blocks_brain.disinhibit_area(area_name=BLOCKS, lock=0)
		blocks_brain.disinhibit_area(area_name=head, lock=0)
		blocks_brain.disinhibit_area(area_name=nodes[0], lock=0)
		blocks_brain.disinhibit_fiber(area1=head, area2=nodes[0], lock=0)
		blocks_brain.disinhibit_fiber(area1=BLOCKS, area2=nodes[0], lock=0)
		count = 1

		for block in blocks:
			previous_to_area_index = (count - 2) % MAX_NODES_AREAS
			to_area_index = (count - 1) % MAX_NODES_AREAS
			next_to_area_index = count % MAX_NODES_AREAS
			blocks_brain.activate_block(index=block)


			# project star
			if count == 1:
				blocks_brain.project_star(project_rounds=100, verbose=verbose)
			else:
				blocks_brain.project_star(project_rounds=project_rounds, verbose=verbose)

			for area in working_regions:
				blocks_brain.brain.areas[area].unfix_assembly()

			# post-actions for current block
			if count==1:
				blocks_brain.inhibit_area(area_name=head, lock=0)
				blocks_brain.inhibit_fiber(area1=head, area2=nodes[0], lock=0)
				blocks_brain.inhibit_fiber(area1=BLOCKS, area2=nodes[0], lock=0)
			else:
				blocks_brain.inhibit_area(area_name=nodes[previous_to_area_index], lock=0)
				blocks_brain.inhibit_fiber(area1=BLOCKS, area2=nodes[to_area_index], lock=0)
				blocks_brain.inhibit_fiber(area1=nodes[previous_to_area_index], area2=nodes[to_area_index], lock=0)

			# pre-actions for next block
			blocks_brain.disinhibit_area(area_name=nodes[next_to_area_index], lock=0)
			blocks_brain.disinhibit_fiber(area1=BLOCKS, area2=nodes[next_to_area_index], lock=0)
			blocks_brain.disinhibit_fiber(area1=nodes[to_area_index], area2=nodes[next_to_area_index], lock=0)

			count += 1

		# inhibit all areas and fibers
		blocks_brain.inhibit_areas(area_names=[BLOCKS]+working_regions, lock=0) 
		blocks_brain.inhibit_all_fibers(area_names=[BLOCKS]+working_regions, lock=0) 

	return blocks_brain


def readout(blocks_brain, stacks_number, stacks_lengths, top_areas, prefix, verbose=False):
	'''
	Parameters
	- blocks_brain: a BlocksBrain represting a stack of blocks
	- stacks_number: number of stacks in the block world
	- stacks_lengths: array of numbers of blocks in each stack
	- top_areas: array of indices of the working areas corresponding to the top block for each stack
	- prefix: string "I", "G", "B", or "T" whether the areas we work on are the input areas, the goal areas, the building areas, or the table areas.
	Example: readout(bb,4,[2,3,1,3],[1,1],"G")
	'''
	stacks = [] # will be the output
	blocks_brain.brain.no_plasticity = True # will perform sudo project, not changing weights

	# adding prefix to regions
	working_regions = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	if prefix == "T": # if stacks are on the table, need to check if blocks are relocated
		working_regions.append(RELOCATED) 

	# unfix all assemblies for readout
	for area in [BLOCKS]+working_regions:
		blocks_brain.brain.areas[area].unfix_assembly()

	# read each stack
	for j in range(stacks_number):
		blocks = [] # will be appended to the output
		if len(top_areas) <= j or top_areas[j] == None:
			stacks.append(blocks)
			print("Warning: stack {} is not encoded or the top_area is not provided.".format(j))
			continue
		# defining locally the resgions involved to parse stack number j
		head = add_prefix([HEADS[j]], prefix=prefix)[0]
		nodes = add_prefix(NODES[j], prefix=prefix)

		# first, process the head block 
		# pre-actions
		blocks_brain.disinhibit_area(area_name=head, lock=0)
		if blocks_brain.brain.areas[head].winners.shape[0]==0: # TODO: necessary? if no winners in head, skip this stack
			stacks.append([None] * stacks_lengths[j])
			continue
		blocks_brain.disinhibit_area(area_name=nodes[top_areas[j]], lock=0)
		blocks_brain.disinhibit_fiber(area1=head, area2=nodes[top_areas[j]], lock=0)
		# project
		# from head to the area encoding the first block
		blocks_brain.brain.project(stim_to_area={}, area_to_area={head: set([head, nodes[top_areas[j]]])})
		# post-actions
		blocks_brain.inhibit_area(area_name=head, lock=0)
		blocks_brain.inhibit_fiber(area1=head, area2=nodes[top_areas[j]], lock=0)
		# pre-actions
		blocks_brain.disinhibit_area(area_name=BLOCKS, lock=0)
		blocks_brain.disinhibit_fiber(area1=nodes[top_areas[j]], area2=BLOCKS, lock=0)
		# project
		# from the area encoding the first block to the BLOCKS area
		project_map = {}
		if prefix=="T": # if block on the table, check if it has been RELOCATED
			# if a block has a stable projection to RELOCATED, it means it was tagged as being relocated to a non-table area already
			blocks_brain.disinhibit_area(area_name=RELOCATED, lock=0)
			blocks_brain.disinhibit_fiber(area1=nodes[top_areas[j]], area2=RELOCATED, lock=0)
			project_map[nodes[top_areas[j]]] = set([BLOCKS, nodes[top_areas[j]], RELOCATED])
		else:
			project_map[nodes[top_areas[j]]] = set([BLOCKS, nodes[top_areas[j]]])
		blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)
		if not (blocks_brain.is_assembly(area_name=RELOCATED)):
			# if the prefix is not TABLE, RELOCATED is inhibited and no assembly will be found
			# if the prefix is TABLE, it only appends the block if the block has not been relocated
			# decode from BLOCKS the block index associated with the projection from the first block area
			blocks.append(blocks_brain.get_block_index(area_name=BLOCKS, min_overlap=0.75))
		# post-actions
		blocks_brain.inhibit_area(area_name=BLOCKS, lock=0)
		blocks_brain.inhibit_fiber(area1=nodes[top_areas[j]], area2=BLOCKS, lock=0)
		current_area = top_areas[j]
		next_area = (top_areas[j] + 1) % MAX_NODES_AREAS 

		# then, process the remaining blocks after the head block
		for i in range(1, stacks_lengths[j]): 
			# pre-actions
			blocks_brain.disinhibit_area(area_name=nodes[next_area], lock=0)
			blocks_brain.disinhibit_fiber(area1=nodes[current_area], area2=nodes[next_area], lock=0)
			# project
			# from the current area to the next area
			blocks_brain.brain.project(stim_to_area={}, area_to_area={nodes[current_area]: set([nodes[next_area]])})
			# post-actions
			blocks_brain.inhibit_area(area_name=nodes[current_area], lock=0)
			# pre-actions
			blocks_brain.disinhibit_area(area_name=BLOCKS, lock=0)
			blocks_brain.disinhibit_fiber(area1=nodes[next_area], area2=BLOCKS, lock=0)
			# project
			# from the next area to BLOCKS area
			project_map = {}
			if prefix=="T": # if block on table, check if it has been RELOCATED
				blocks_brain.disinhibit_area(area_name=RELOCATED, lock=0)
				blocks_brain.disinhibit_fiber(area1=nodes[next_area], area2=RELOCATED, lock=0)
				project_map[nodes[next_area]] = set([BLOCKS, nodex[next_area], RELOCATED])
			else:
				project_map[nodes[next_area]] = set([BLOCKS, nodes[next_area]])
			blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)
			if not(blocks_brain.is_assembly(area_name=RELOCATED)):
				# if the prefix is not TABLE, RELOCATED is inhibited and no assembly will be found
				# if the prefix is TABLE, it only appends the block if the block has not been relocated
				# decode from BLOCKS the block index associated with the projection from the next area
				blocks.append(blocks_brain.get_block_index(area_name=BLOCKS, min_overlap=0.75))
			# post-actions
			blocks_brain.inhibit_area(area_name=RELOCATED, lock=0)
			blocks_brain.inhibit_fiber(area1=nodes[next_area], area2=RELOCATED, lock=0)
			blocks_brain.inhibit_area(area_name=BLOCKS, lock=0)
			blocks_brain.inhibit_fiber(area1=nodes[next_area], area2=BLOCKS, lock=0)
			current_area = next_area
			next_area = (next_area + 1) % MAX_NODES_AREAS

		stacks.append(blocks) # append the blocks in this stack to the output
		blocks_brain.inhibit_areas(area_names=[BLOCKS]+working_regions, lock=0) # inhibit everything after each stack
		blocks_brain.inhibit_all_fibers(area_names=[BLOCKS]+working_regions, lock=0)

	blocks_brain.brain.no_plasticity = False # reset to the default mode
	return stacks


def top(blocks_brain, stack_index, prefix, verbose=False):
	'''
	Find the first areas in which an assembly is activated 
	by firing from HEADS.
	Return corresponding node area index,
		and block index (for each stack) in a dictionary
	Parameters
	- blocks_brain: a BlocksBrain representing stacks of blocks
	- stack_index: the index of the stack we look at
	- prefix: string "I", "G", "B", or "T" whether the areas we work on are 
			the input areas, the goal areas, the building areas, or the table areas.
	Example: top(bb,4,"G")
	'''
	# temporarily disable weight update because we will do sudo-project here
	blocks_brain.brain.no_plasticity = True 
	# add prefix to regions
	working_regions = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	# unfix all areas to prepare for sudo-project
	for area in [BLOCKS]+working_regions:
		blocks_brain.brain.areas[area].unfix_assembly()
	# prepare head and nodes for the current stack
	head = add_prefix([HEADS[stack_index]], prefix=prefix)[0]
	nodes = add_prefix(NODES[stack_index], prefix=prefix)
	# global pre-actions
	blocks_brain.disinhibit_area(area_name=head, lock=0)
	# if head has an assembly
	if blocks_brain.is_assembly(area_name=head):
		# iterate all node areas to find which one is connected with head
		for node_area_index in range(MAX_NODES_AREAS):
			# pre-actions
			blocks_brain.disinhibit_area(area_name=nodes[node_area_index], lock=0)
			blocks_brain.disinhibit_fiber(area1=head, area2=nodes[node_area_index], lock=0)
			# sudo project from head to the current node area
			project_map = {}
			project_map[head] = set([head, nodes[node_area_index]])
			blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)
			# if there is valid assembly
			if blocks_brain.is_assembly(area_name=nodes[node_area_index]):
				print("Area {} is the top area".format(nodes[node_area_index])) if verbose else 0
				# post-actions
				blocks_brain.inhibit_area(area_name=head, lock=0)
				blocks_brain.inhibit_fiber(area1=head, area2=nodes[node_area_index], lock=0)
				
				# check block index
				block_index = None
				# pre-actions 
				blocks_brain.disinhibit_areas(area_names=[nodes[node_area_index], BLOCKS], lock=0)
				blocks_brain.disinhibit_fiber(area1=nodes[node_area_index], area2=BLOCKS, lock=0)
				# sudo-project
				project_map = {}
				project_map[nodes[node_area_index]] = set([nodes[node_area_index], BLOCKS])
				blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)
				# get block index
				block_index = blocks_brain.get_block_index(area_name=BLOCKS)
				print("Block {} is the top block".format(block_index)) if verbose else 0
				# post-actions
				blocks_brain.inhibit_area(area_name=BLOCKS, lock=0)
				blocks_brain.inhibit_area(area_name=nodes[node_area_index], lock=0)
				blocks_brain.inhibit_fiber(area1=BLOCKS, area2=nodes[node_area_index], lock=0)

				# return
				blocks_brain.brain.no_plasticity = False # reset to default mode
				return node_area_index, block_index
			# if there is no valid assembly, close areas and fibers for next iteration
			# post-actions
			blocks_brain.inhibit_area(area_name=nodes[node_area_index], lock=0)
			blocks_brain.inhibit_fiber(area1=head, area2=nodes[node_area_index], lock=0)
	# if no area is found, global post-actions and return default result
	blocks_brain.inhibit_areas(area_names=[BLOCKS, RELOCATED]+working_regions, lock=0)
	blocks_brain.inhibit_all_fibers(area_names=[BLOCKS, RELOCATED]+working_regions, lock=0)
	blocks_brain.brain.no_plasticity = False # reset to default mode
	print("No area connected with head found.") if verbose else 0
	return None, None


def pop(blocks_brain, stack_index, prefix, verbose=False):
	'''
	Remove the top block from the stack

	Parameters
	- blocks_brain: a BlocksBrain representing stacks of blocks
	- stack_index: the index of the stack we look at
	- prefix: string "I", "G", "B", or "T" whether the areas we work on are the input areas, the goal areas, the building areas, or the table areas.
	Example: pop!(bb,4,2,"G")
	'''
	top_area, top_block = top(blocks_brain, stack_index, prefix) # get top node area index, block index
	if top_area==None: # TODO
		print("\tWarning: top_area is None, reset to 0")
		pop(blocks_brain, stack_index, prefix, verbose=False)
	new_top_area = (top_area + 1) % MAX_NODES_AREAS 

	# defining locally the regions involved to parse stack number stack_index
	head = add_prefix(regions=[HEADS[stack_index]], prefix=prefix)[0]
	nodes = add_prefix(regions=NODES[stack_index], prefix=prefix)

	print("Top area: {}, top block: {}".format(nodes[top_area], top_block)) if verbose else 0

	if is_last_block(blocks_brain, stack_index, top_area, top_block, prefix, verbose):
		blocks_brain.brain.areas[head].winners = np.array([])
		print("Top block is last block.") if verbose else 0
	print("New top area:", nodes[new_top_area]) if verbose else 0

	blocks_brain.brain.no_plasticity = True
	for area in [BLOCKS,head]+nodes:
		if blocks_brain.is_assembly(area_name=area):
			blocks_brain.brain.areas[area].unfix_assembly()

	# looking for the correct assembly in new area
	# preactions
	blocks_brain.disinhibit_area(area_name=head, lock=0)
	blocks_brain.disinhibit_area(area_name=nodes[top_area], lock=0)
	blocks_brain.disinhibit_fiber(area1=head, area2=nodes[top_area], lock=0)
	project_map = {}
	project_map[head] = set([head, nodes[top_area]])
	blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)
	# postactions
	blocks_brain.inhibit_area(area_name=head, lock=0)
	blocks_brain.inhibit_fiber(area1=head, area2=nodes[top_area], lock=0)

	# preactions
	blocks_brain.disinhibit_area(area_name=nodes[new_top_area], lock=0)
	blocks_brain.disinhibit_fiber(area1=nodes[top_area], area2=nodes[new_top_area], lock=0)
	project_map = {}
	project_map[nodes[top_area]] = set([nodes[top_area], nodes[new_top_area]])
	blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)
	# postactions
	blocks_brain.inhibit_area(area_name=nodes[top_area], lock=0)
	blocks_brain.inhibit_fiber(area1=nodes[top_area], area2=nodes[new_top_area], lock=0)

	# creating new association between new area and head
	blocks_brain.brain.no_plasticity = False
	blocks_brain.disinhibit_areas(area_names=[head,BLOCKS], lock=0)
	blocks_brain.disinhibit_fiber(area1=head, area2=nodes[new_top_area], lock=0)
	blocks_brain.disinhibit_fiber(area1=BLOCKS, area2=nodes[new_top_area], lock=0)
	project_map = {}
	project_map[nodes[new_top_area]] = set([nodes[new_top_area], head, BLOCKS])
	blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)

	# reinforcing the association between new area and head
	# preactions
	blocks_brain.project_star(project_rounds=40, verbose=False)
	# postactions
	blocks_brain.inhibit_areas(area_names=[BLOCKS, head], lock=0)
	blocks_brain.inhibit_area(area_name=nodes[new_top_area], lock=0)
	blocks_brain.inhibit_fibers(area_pairs=[[nodes[new_top_area], head],[nodes[new_top_area], BLOCKS]], lock=0)
	for area in [BLOCKS, head]+nodes:
		if blocks_brain.is_assembly(area_name=area):
			blocks_brain.brain.areas[area].unfix_assembly()
	return new_top_area # return new node index


def put(blocks_brain, stack_index, block, prefix, verbose=False):
	'''
	Add a new block on top of the stack.
	Adding pointer head from new area by projecting new block first to new area and then to head

	Parameters
	- blocks_brain: a BlocksBrain representing stacks of blocks
	- stack_index: the index of the stack we look at
	- block: (Int) number of block to put on top of the stack
	- prefix: string "I", "G", "B", or "T" whether the areas we work on are the input areas, the goal areas, the building areas, or the table areas.
	Example: put!(bb,4,2,"G")
	'''
	# get top node area index, block index
	top_area, top_block = top(blocks_brain, stack_index, prefix) 
	print('\ttop node area {}, top block idx {}'.format(top_area, top_block)) if verbose else 0
	
	# disable plasticity
	blocks_brain.brain.no_plasticity = True

	# add prefix to regions
	working_regions = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)

	for area in [BLOCKS]+ working_regions:
		blocks_brain.brain.areas[area].unfix_assembly()

	# defining locally the regions involved to parse stack number stack_index
	head = add_prefix(regions=[HEADS[stack_index]], prefix=prefix)[0]
	nodes = add_prefix(regions=NODES[stack_index], prefix=prefix)

	if top_area==None: 
		print("\tWarning: no block exists in original stack. Begin with new block.")
		new_top_area = MAX_NODES_AREAS - 1
	else:
		new_top_area = (top_area - 1) % MAX_NODES_AREAS 
		print('\tnew top area', new_top_area) if verbose else 0

	blocks_brain.disinhibit_areas(area_names=[nodes[new_top_area], BLOCKS, head], lock=0)
	blocks_brain.disinhibit_fibers(area_pairs=[[BLOCKS, nodes[new_top_area]], [nodes[new_top_area], head]], lock=0)

	# create new assembly in area nodes[new_node_index] from BLOCKS
	blocks_brain.activate_block(index=block)
	project_map = {}
	project_map[BLOCKS] = set([BLOCKS, nodes[new_top_area]])
	blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)

	# create new assembly in head from nodes[new_top_area]
	project_map = {}
	project_map[nodes[new_top_area]] = set([head, nodes[new_top_area]])
	blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)

	project_rounds = 50
	if top_area != None and top_block != None:
		blocks_brain.disinhibit_area(nodes[top_area], lock=0)
		blocks_brain.disinhibit_fiber(area1=nodes[new_top_area], area2=nodes[top_area])
	blocks_brain.activate_block(index=block)
	blocks_brain.project_star(project_rounds=project_rounds)

	for area in [BLOCKS, head]+nodes:
		if blocks_brain.is_assembly(area_name=area):
			blocks_brain.brain.areas[area].unfix_assembly()

	blocks_brain.inhibit_areas(area_names=[BLOCKS]+working_regions, lock=0)
	blocks_brain.inhibit_all_fibers(area_names=[BLOCKS]+working_regions, lock=0)

	return new_top_area


def is_last_block(blocks_brain, stack_index, node_index, block, prefix, verbose=False):
	'''
	this function is necessary for dismantle! and pop! and checks if a given block is the last one
	
	Parameters
	- blocks_brain: type BlocksBrain, after having parsed the stacks
	- stacks_index: an integer, it is the index of stack we investigate
	- node_index: an integer, the NODE area the block is projected into
	- block: an integer, the block we check
	- prefix: string "I", "G", "B", or "T" whether the areas we work on are the input areas, the goal areas, the building areas, or the table areas.
	Example: is_last_block(blocks_brain, 5, 2, 3, "I")
	'''
	# define locally the regions involved
	nodes = add_prefix(regions=NODES[stack_index], prefix=prefix)
	
	blocks_brain.brain.no_plasticity = True

	# preactions
	next_index = (node_index+1) % MAX_NODES_AREAS
	blocks_brain.disinhibit_areas(area_names=[nodes[node_index],nodes[next_index],BLOCKS], lock=0)
	blocks_brain.disinhibit_fibers(area_pairs=[[nodes[node_index],nodes[next_index]],[BLOCKS,nodes[node_index]]], lock=0)

	for area in [BLOCKS]+nodes:
		blocks_brain.brain.areas[area].unfix_assembly()


	blocks_brain.activate_block(index=block)
	project_map = {}
	project_map[BLOCKS] = set([nodes[node_index], BLOCKS])
	blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)

	project_map = {}
	project_map[nodes[node_index]] = set([nodes[node_index], nodes[next_index]])
	blocks_brain.brain.project(stim_to_area={}, area_to_area=project_map)

	is_last = not blocks_brain.is_assembly(area_name=nodes[next_index])

	blocks_brain.brain.areas[BLOCKS].unfix_assembly()
	blocks_brain.inhibit_areas(area_names=[nodes[node_index],nodes[next_index]], lock=0)
	blocks_brain.inhibit_fiber(area1=nodes[node_index], area2=nodes[next_index], lock=0)
	blocks_brain.brain.no_plasticity = False
	print("is_last:", is_last) if verbose else 0

	return is_last


























