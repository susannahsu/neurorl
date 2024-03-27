from envs.AC.bw_apps import *

# --------------------------------- test brain.py
def test_add_stimulus():
	brain = Brain(0.6)
	brain.print_info()
	brain.add_stimulus("S", k=10)
	brain.print_info()

def test_add_nonexplicit_area1():
	brain = Brain(0.6)
	brain.print_info()
	area_name="A0"
	brain.add_nonexplicit_area(area_name, n=10,k=2,beta=0.1)
	brain.print_info()
	brain.areas[area_name].print_info()

def test_add_nonexplicit_area2():
	brain = Brain(0.6)
	brain.print_info()
	brain.add_nonexplicit_area("A0", n=10,k=2,beta=0.1)
	brain.print_info()
	brain.areas["A0"].print_info()
	brain.add_nonexplicit_area("A1", n=10,k=4,beta=0.5)
	brain.print_info()
	brain.areas["A0"].print_info()
	brain.areas["A1"].print_info()

def test_add_nonexplicit_area3():
	brain = Brain(0.6)
	brain.add_stimulus("S", k=5)
	brain.print_info()
	brain.add_nonexplicit_area("A0", n=10,k=2,beta=0.1)
	brain.print_info()
	brain.areas["A0"].print_info()

def test_add_nonexplicit_area4():
	brain = Brain(0.6)
	brain.add_stimulus("S", k=5)
	brain.print_info()
	brain.add_nonexplicit_area("A0", n=10,k=2,beta=0.1)
	brain.print_info()
	brain.areas["A0"].print_info()
	brain.add_nonexplicit_area("A1", n=10,k=4,beta=0.5)
	brain.print_info()
	brain.areas["A0"].print_info()
	brain.areas["A1"].print_info()

def test_add_explicit_area1():
	brain = Brain(0.6)
	brain.add_stimulus("S", k=5)
	brain.print_info()
	brain.add_explicit_area("A0", n=10,k=2,beta=0.1)
	brain.print_info()
	brain.areas["A0"].print_info()

def test_add_explicit_area2():
	brain = Brain(0.6)
	brain.add_stimulus("S", k=5)
	brain.print_info()
	brain.add_explicit_area("A0", n=10,k=2,beta=0.1)
	brain.print_info()
	brain.areas["A0"].print_info()
	brain.add_explicit_area("A1", n=10,k=4,beta=0.5)
	brain.print_info()
	brain.areas["A0"].print_info()
	brain.areas["A1"].print_info()

def test_add_explicit_area3():
	brain = Brain(0.6)
	brain.add_stimulus("S", k=5)
	brain.add_explicit_area("A0", n=10,k=2,beta=0.1)
	brain.add_explicit_area("A1", n=10,k=4,beta=0.5)
	brain.add_nonexplicit_area("A2", n=10,k=3,beta=0.8)
	brain.print_info()
	brain.areas["A0"].print_info()
	brain.areas["A1"].print_info()
	brain.areas["A2"].print_info()

def test_add_explicit_area4():
	brain = Brain(0.6)
	brain.add_stimulus("S1", k=5)
	brain.add_stimulus("S2", k=7)
	brain.add_explicit_area("A0", n=10,k=2,beta=0.1)
	brain.add_explicit_area("A1", n=10,k=4,beta=0.5)
	brain.add_nonexplicit_area("A2", n=10,k=3,beta=0.8)
	brain.print_info()
	brain.areas["A0"].print_info()
	brain.areas["A1"].print_info()
	brain.areas["A2"].print_info()

def test_project_into1():
	brain = Brain(0.6)
	brain.add_stimulus("S", k=5)
	brain.add_explicit_area("A0", n=10,k=2,beta=0.1)
	brain.add_explicit_area("A1", n=10,k=4,beta=0.5)
	brain.areas["A0"].winners = torch.tensor([1,2])
	brain.areas["A0"].w = 2
	brain.areas["A1"].winners = torch.tensor([3,4,5,6])
	brain.areas["A1"].w = 4
	brain.project_into(to_area=brain.areas["A0"], from_stimuli={"S"}, from_areas={"A1"})

def test_project_into2():
	brain = Brain(0.6)
	brain.add_stimulus("S", k=5)
	brain.add_explicit_area("A0", n=10,k=2,beta=0.1)
	brain.add_explicit_area("A1", n=8,k=4,beta=0.5)
	brain.areas["A0"].winners = torch.tensor([1,2])
	brain.areas["A0"].w = 2
	brain.areas["A1"].winners = torch.tensor([3,4,5,6])
	brain.areas["A1"].w = 4
	brain.project_into(to_area=brain.areas["A0"], from_stimuli={"S"}, from_areas={"A0", "A1"})

def test_project_into3():
	brain = Brain(0.6)
	brain.add_stimulus("S1", k=5)
	brain.add_stimulus("S2", k=6)
	brain.add_explicit_area("A0", n=10,k=2,beta=0.1)
	brain.add_explicit_area("A1", n=8,k=4,beta=0.5)
	brain.areas["A0"].winners = torch.tensor([1,2])
	brain.areas["A0"].w = 2
	brain.areas["A1"].winners = torch.tensor([3,4,5,6])
	brain.areas["A1"].w = 4
	brain.project_into(to_area=brain.areas["A0"], from_stimuli={"S1", "S2"}, from_areas={"A0", "A1"})

def test_project1():
	brain = Brain(0.6)
	brain.add_stimulus("S1", k=5)
	brain.add_stimulus("S2", k=6)
	brain.add_explicit_area("A0", n=10,k=2,beta=0.1)
	brain.add_explicit_area("A1", n=8,k=4,beta=0.5)
	brain.areas["A0"].winners = torch.tensor([1,2])
	brain.areas["A0"].w = 2
	brain.areas["A1"].winners = torch.tensor([3,4,5,6])
	brain.areas["A1"].w = 4
	brain.project(stim_to_area={"S1": {"A0"}, "S2": {"A0", "A1"}}, area_to_area={"A0": {"A0"}, "A1": {"A0", "A1"}})

def test_project2():
	brain = Brain(0.6)
	brain.add_stimulus("S1", k=5)
	brain.add_stimulus("S2", k=6)
	brain.add_explicit_area("A0", n=10,k=2,beta=0.1)
	brain.add_explicit_area("A1", n=8,k=4,beta=0.5)
	brain.areas["A0"].winners = torch.tensor([1,2])
	brain.areas["A0"].w = 2
	brain.areas["A1"].winners = torch.tensor([3,4,5,6])
	brain.areas["A1"].w = 4
	brain.project(stim_to_area={"S1": {"A0", "A1"}, "S2": {"A0", "A1"}}, area_to_area={"A0": {"A1"}, "A1": {"A0", "A1"}})
# --------------------------------- end test brain.py


# --------------------------------- test blocks_brain.py
def test_blocksbrain1():
	bb = BlocksBrain(blocks_number=2, other_areas=["A1", "A2"])
	bb.print_info()
	for n, a in bb.brain.areas.items():
		a.print_info()

def test_blocksbrain2():
	bb = BlocksBrain(blocks_number=2, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.print_info()
	for n, a in bb.brain.areas.items():
		a.print_info()

def test_inhibit1():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	print(bb.area_states)
	bb.inhibit_area("A1", lock=0)
	print(bb.area_states)
	print(bb.fiber_states)
	bb.inhibit_fiber("BLOCKS", "A1", lock=1)
	print(bb.fiber_states)

def test_disinhibit1():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	print(bb.area_states)
	bb.inhibit_area("A1", lock=0)
	print(bb.area_states)
	print(bb.fiber_states)
	bb.inhibit_fiber("BLOCKS", "A1", lock=1)
	print(bb.fiber_states)
	bb.disinhibit_area("A1", lock=1)
	print(bb.area_states)
	bb.disinhibit_area("A1", lock=0)
	print(bb.area_states)
	bb.disinhibit_fiber("BLOCKS", "A2", lock=1)
	print(bb.fiber_states)
	bb.disinhibit_fiber("BLOCKS", "A1", lock=1)
	print(bb.fiber_states)

def test_activate1():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.activate_block(index=0)
	bb.brain.areas["BLOCKS"].print_info()
	bb.activate_block(index=1)
	bb.brain.areas["BLOCKS"].print_info()
	bb.activate_block(index=2)
	bb.brain.areas["BLOCKS"].print_info()
	bb.activate_block(index=4)
	bb.brain.areas["BLOCKS"].print_info()

def test_activate2():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.activate_block(index=0)
	bb.activate_block(index=1)
	bb.brain.areas["BLOCKS"].print_info()
	bb.activate_assembly(index=2, activation_area="BLOCKS")
	bb.brain.areas["BLOCKS"].print_info()
	bb.activate_assembly(index=3, activation_area="BLOCKS")

def test_activate3():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.activate_block(index=0)
	bb.activate_block(index=1)
	bb.brain.areas["BLOCKS"].print_info()
	bb.brain.areas["A1"].print_info()
	bb.activate_assembly(index=2, activation_area="BLOCKS")
	bb.activate_assembly(index=3, activation_area="A1")
	bb.brain.areas["A1"].print_info()
	bb.brain.areas["BLOCKS"].print_info()
	bb.activate_assembly(index=100, activation_area="A1")
	
def test_get_project_map1():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.brain.add_stimulus(stim_name="S2", k=6)
	pmap = bb.get_project_map(verbose=True)
	print("\n", pmap)

def test_get_project_map2():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.brain.add_stimulus(stim_name="S2", k=6)
	bb.disinhibit_area(area_name="BLOCKS", lock=0)
	bb.disinhibit_area(area_name="A1", lock=0)
	bb.disinhibit_area(area_name="A2", lock=0)
	bb.disinhibit_fiber(area1="BLOCKS", area2="A1", lock=0)
	pmap = bb.get_project_map(verbose=True)
	print("\n", pmap)

def test_get_project_map3():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.brain.add_stimulus(stim_name="S2", k=6)
	bb.activate_block(index=0)
	bb.disinhibit_area(area_name="BLOCKS", lock=0)
	bb.disinhibit_area(area_name="A1", lock=0)
	bb.disinhibit_area(area_name="A2", lock=0)
	bb.disinhibit_fiber(area1="BLOCKS", area2="A1", lock=0)
	bb.disinhibit_fiber(area1="BLOCKS", area2="BLOCKS", lock=0)
	pmap = bb.get_project_map(verbose=True)
	print("\n", pmap)

def test_get_project_map4():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.brain.add_stimulus(stim_name="S2", k=6)
	bb.activate_block(index=0)
	bb.activate_assembly(index=5, activation_area="A1")
	bb.disinhibit_area(area_name="BLOCKS", lock=0)
	bb.disinhibit_area(area_name="A1", lock=0)
	bb.disinhibit_area(area_name="A2", lock=0)
	bb.disinhibit_fiber(area1="BLOCKS", area2="A1", lock=0)
	pmap = bb.get_project_map(verbose=True)
	print("\n", pmap)

def test_get_project_map5():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.brain.add_stimulus(stim_name="S2", k=6)
	bb.activate_block(index=0)
	bb.activate_assembly(index=5, activation_area="A1")
	bb.activate_assembly(index=6, activation_area="A2")
	bb.disinhibit_area(area_name="BLOCKS", lock=0)
	bb.disinhibit_area(area_name="A1", lock=0)
	bb.disinhibit_area(area_name="A2", lock=0)
	bb.disinhibit_fiber(area1="BLOCKS", area2="A1", lock=0)
	bb.disinhibit_fiber(area1="A2", area2="BLOCKS", lock=0)
	pmap = bb.get_project_map(verbose=True)
	print("\n", pmap)

def test_get_project_map6():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.brain.add_stimulus(stim_name="S2", k=6)
	bb.activate_block(index=0)
	bb.activate_assembly(index=5, activation_area="A1")
	bb.activate_assembly(index=6, activation_area="A2")
	bb.disinhibit_area(area_name="BLOCKS", lock=0)
	bb.disinhibit_area(area_name="A1", lock=0)
	bb.disinhibit_area(area_name="A2", lock=0)
	bb.disinhibit_fiber(area1="BLOCKS", area2="A1", lock=0)
	bb.disinhibit_fiber(area1="A2", area2="A1", lock=0)
	pmap = bb.get_project_map(verbose=True)
	print("\n", pmap)

def test_test_project1():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.brain.add_stimulus(stim_name="S2", k=6)
	bb.activate_block(index=0)
	bb.disinhibit_area(area_name="BLOCKS", lock=0)
	bb.disinhibit_area(area_name="A1", lock=0)
	bb.disinhibit_area(area_name="A2", lock=0)
	bb.disinhibit_fiber(area1="BLOCKS", area2="A1", lock=0)
	bb.disinhibit_fiber(area1="A2", area2="A1", lock=0)
	bb.test_project(verbose=True)

def test_test_project2():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.brain.add_stimulus(stim_name="S2", k=6)
	bb.brain.areas["A1"].winners = torch.tensor(range(100))
	bb.brain.areas["A2"].winners = torch.tensor(range(100,200))
	bb.activate_block(index=0)
	bb.disinhibit_area(area_name="BLOCKS", lock=0)
	bb.disinhibit_area(area_name="A1", lock=0)
	bb.disinhibit_area(area_name="A2", lock=0)
	bb.disinhibit_fiber(area1="BLOCKS", area2="A1", lock=0)
	bb.disinhibit_fiber(area1="A2", area2="A1", lock=0)
	bb.disinhibit_fiber(area1="A2", area2="BLOCKS", lock=0)
	bb.test_project(verbose=True)
	for k, a in bb.brain.areas.items():
		a.print_info()
	
def test_is_assembly1():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.brain.add_stimulus(stim_name="S2", k=6)
	bb.activate_block(index=0)
	bb.activate_assembly(index=5, activation_area="A1")
	print(bb.is_assembly(area_name="A1", verbose=True))

def test_get_block_index2():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.brain.add_stimulus(stim_name="S2", k=6)
	bb.activate_block(index=4)
	print(bb.get_block_index())

def test_get_block_index3():
	bb = BlocksBrain(blocks_number=3, other_areas=["A1", "A2"])
	bb.brain.add_stimulus(stim_name="S1", k=5)
	bb.brain.add_stimulus(stim_name="S2", k=6)
	bb.activate_block(index=2)
	print(bb.get_block_index(area_name="A1"))
# --------------------------------- end test blocks_brain.py


# --------------------------------- test bw_apps.py
def test_node_names():
	print("NODES:", NODES)
	print("HEADS:", HEADS)
	print("REGIONS:", REGIONS)
	print("NODES added prefix:", add_prefix(NODES[0], "I"))

def test_is_above1():
	r = is_above([0,1,2], query_a=0, query_b=1, p=0.1, eak=10, nean=100, neak=10, db=0.2)
	print("\n\n", r)
	assert r==True

def test_is_above2():
	r = is_above([0,1], query_a=1, query_b=0, p=0.1, eak=8, nean=50, neak=10, db=0.2)
	print("\n\n", r)
	assert r==False

def test_is_above3():
	r = is_above([0,1,2,3,4], query_a=3, query_b=4, p=0.1, eak=10, nean=100, neak=10, db=0.2)
	print("\n\n", r)
	assert r==True

def test_is_above4():
	r = is_above([0,1,2,3,4], query_a=4, query_b=0, p=0.1, eak=10, nean=100, neak=10, db=0.2)
	print("\n\n", r)
	assert r==False

def test_is_above5():
	r = is_above([0,1,2,3,4], query_a=5, query_b=0, p=0.1, eak=10, nean=100, neak=10, db=0.2)
	print("\n\n", r)
	
def test_parse1():
	print("NODES:", NODES)
	print("HEADS:", HEADS)
	print("REGIONS:", REGIONS)
	prefix="I"
	stacks = [[0],[1,2]]
	print("input stack:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	bb = BlocksBrain(blocks_number=3, other_areas=oa, p=0.1, eak=10, nean=100, neak=10, db=0.2)
	parse(bb, stacks=stacks, prefix='I')
	stack_maps=[set([BLOCKS, "I0_N0"]), 
				set([BLOCKS, "I1_N0"])]
	for s in range(len(stacks)):
		print("\n---------stack", s)
		bb.disinhibit_areas(area_names=[BLOCKS]+oa[s*4:(s+1)*4])
		bb.disinhibit_all_fibers(area_names=[BLOCKS]+oa[s*4:(s+1)*4])
		bb.activate_block(index=stacks[s][0])
		project_map = bb.get_project_map()
		project_map[BLOCKS] = stack_maps[s]
		bb.brain.project(stim_to_area={}, area_to_area=project_map)
		for a in bb.brain.areas:
			if bb.is_assembly(area_name=a):
				print(a)
		bb.inhibit_areas(area_names=bb.all_areas)
		bb.inhibit_all_fibers(area_names=bb.all_areas)
		
def test_parse2():
	prefix="G"
	stacks = [[0],[1,2]]
	print("input stack:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=10, nean=100, neak=10, db=0.2)
	parse(bb, stacks=stacks, prefix=prefix)
	stack_maps=[set([BLOCKS, prefix+"0_N0"]), 
				set([BLOCKS, prefix+"1_N0"])]
	for s in range(len(stacks)):
		print("\n---------stack", s)
		bb.disinhibit_areas(area_names=[BLOCKS]+oa[s*4:(s+1)*4])
		bb.disinhibit_all_fibers(area_names=[BLOCKS]+oa[s*4:(s+1)*4])
		bb.activate_block(index=stacks[s][0])
		project_map = bb.get_project_map()
		project_map[BLOCKS] = stack_maps[s]
		bb.brain.project(stim_to_area={}, area_to_area=project_map)
		for a in bb.brain.areas:
			if bb.is_assembly(area_name=a):
				print(a)
		bb.inhibit_areas(area_names=bb.all_areas)
		bb.inhibit_all_fibers(area_names=bb.all_areas)

def test_parse3():
	prefix="G"
	stacks = [[0,1,2,3],[4,5,6,7]]
	print("input stack:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=10, other_areas=oa, p=0.1, eak=10, nean=100, neak=10, db=0.2)
	parse(bb, stacks=stacks, prefix=prefix)
	stack_maps=[set([BLOCKS, prefix+"0_N0"]), 
				set([BLOCKS, prefix+"1_N0"])]
	for s in range(len(stacks)):
		print("\n---------stack", s)
		bb.disinhibit_areas(area_names=[BLOCKS]+oa[s*4:(s+1)*4])
		bb.disinhibit_all_fibers(area_names=[BLOCKS]+oa[s*4:(s+1)*4])
		bb.activate_block(index=stacks[s][0])
		project_map = bb.get_project_map()
		project_map[BLOCKS] = stack_maps[s]
		bb.brain.project(stim_to_area={}, area_to_area=project_map)
		for a in bb.brain.areas:
			if bb.is_assembly(area_name=a):
				print(a)
		bb.inhibit_areas(area_names=bb.all_areas)
		bb.inhibit_all_fibers(area_names=bb.all_areas)

def test_parse4():
	prefix="G"
	stacks = [[0,1,2,3],[4,5,6,7]]
	print("input stack:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=10, other_areas=oa, p=0.1, eak=10, nean=100, neak=10, db=0.2)
	parse(bb, stacks=stacks, prefix=prefix)
	stack_maps=[set([BLOCKS, prefix+"0_N2"]), 
				set([BLOCKS, prefix+"1_N2"])]
	for s in range(len(stacks)):
		print("\n---------stack", s)
		bb.disinhibit_areas(area_names=[BLOCKS]+oa[s*4:(s+1)*4])
		bb.disinhibit_all_fibers(area_names=[BLOCKS]+oa[s*4:(s+1)*4])
		bb.activate_block(index=stacks[s][0])
		project_map = bb.get_project_map()
		project_map[BLOCKS] = stack_maps[s]
		bb.brain.project(stim_to_area={}, area_to_area=project_map)
		for a in bb.brain.areas:
			if bb.is_assembly(area_name=a):
				print(a)
		bb.inhibit_areas(area_names=bb.all_areas)
		bb.inhibit_all_fibers(area_names=bb.all_areas)

def test_parse5():
	prefix="G"
	stacks = [[0,1,2,3]]
	print("input stack:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=10, other_areas=oa, p=0.1, eak=10, nean=100, neak=10, db=0.2)
	parse(bb, stacks=stacks, prefix=prefix)
	stack_maps=[set([BLOCKS, prefix+"0_N2"])]
	for s in range(len(stacks)):
		print("\n---------stack", s)
		bb.disinhibit_areas(area_names=[BLOCKS]+oa[s*4:(s+1)*4])
		bb.disinhibit_all_fibers(area_names=[BLOCKS]+oa[s*4:(s+1)*4])
		bb.activate_block(index=stacks[s][0])
		project_map = bb.get_project_map()
		project_map[BLOCKS] = stack_maps[s]
		bb.brain.project(stim_to_area={}, area_to_area=project_map)
		for a in bb.brain.areas:
			if bb.is_assembly(area_name=a):
				print(a)
		bb.inhibit_areas(area_names=bb.all_areas)
		bb.inhibit_all_fibers(area_names=bb.all_areas)

def test_readout1():
	prefix = "G"
	stacks = [[0,1],[2]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=50, nean=10000, neak=50, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,1], top_areas=[0,0], prefix=prefix)
	print("readout stacks", r)

def test_readout2():
	prefix = "G"
	stacks = [[0,1,2,3],[4,5,6,7]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=10, other_areas=oa, p=0.1, eak=100, nean=10000, neak=100, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[4,4], top_areas=[0,0], prefix=prefix)
	print("readout stacks", r)

def test_readout3():
	prefix = "G"
	stacks = [[0,1,2,3],[]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	# print("oa:", oa)
	bb = BlocksBrain(blocks_number=10, other_areas=oa, p=0.1, eak=100, nean=10000, neak=100, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[4,0], top_areas=[0,None], prefix=prefix)
	print("readout stacks", r)

def test_readout4():
	prefix = "G"
	stacks = [[0]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	# print("oa:", oa)
	bb = BlocksBrain(blocks_number=1, other_areas=oa, p=0.1, eak=50, nean=10000, neak=50, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	print("winners in head", bb.brain.areas[prefix+"0_H"].winners)
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[1], top_areas=[0,None], prefix=prefix)
	print("readout stacks", r)

def test_readout5():
	prefix = "G"
	stacks = [[0,1,2,3]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	# print("oa:", oa)
	bb = BlocksBrain(blocks_number=10, other_areas=oa, p=0.1, eak=100, nean=20000, neak=100, db=0.1)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[4], top_areas=[0], prefix=prefix)
	print("readout stacks", r)

def test_readout6():
	prefix = "G"
	stacks = [list(range(30))]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	bb = BlocksBrain(blocks_number=50, other_areas=oa, p=0.1, eak=50, nean=1e6, neak=50, db=0.1)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[30], top_areas=[0], prefix=prefix)
	print("readout stacks", r)

def test_top1():
	prefix = "G"
	stacks = [[0,1]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2], top_areas=[0], prefix=prefix)
	print("readout stacks", r)
	print("top...")
	node_area_index, block_index = top(bb, stack_index=0, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))

def test_top2():
	prefix = "G"
	stacks = [[1,2,3]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[3], top_areas=[0], prefix=prefix)
	print("readout stacks", r)
	print("top...")
	node_area_index, block_index = top(bb, stack_index=0, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))

def test_top3():
	prefix = "G"
	stacks = [[3,2,1]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[3], top_areas=[0], prefix=prefix)
	print("readout stacks", r)
	print("top...")
	node_area_index, block_index = top(bb, stack_index=0, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))

def test_top4():
	prefix = "G"
	stacks = [[0,1],[2]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,1], top_areas=[0,0], prefix=prefix)
	print("readout stacks", r)
	print("top...")
	node_area_index, block_index = top(bb, stack_index=0, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))
	print("top...")
	node_area_index, block_index = top(bb, stack_index=1, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))

def test_top5():
	prefix = "G"
	stacks = [[1,0],[3,2,4]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,3], top_areas=[0,0], prefix=prefix)
	print("readout stacks", r)
	print("top...")
	node_area_index, block_index = top(bb, stack_index=0, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))
	print("top...")
	node_area_index, block_index = top(bb, stack_index=1, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))

def test_pop1():
	prefix = "G"
	stacks = [[0,1]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2], top_areas=[0], prefix=prefix)
	print("readout stacks", r)
	
	print("pop...")
	nt = pop(bb, stack_index=0, prefix=prefix, verbose=True)
	print("top...", nt)	
	node_area_index, block_index = top(bb, stack_index=0, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[1], top_areas=[node_area_index], prefix=prefix)
	print("readout stacks", rr)
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[1], top_areas=[nt], prefix=prefix)
	print("readout stacks", rr)

def test_pop2():
	prefix = "G"
	stacks = [[0,1], [2]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,1], top_areas=[0,0], prefix=prefix)
	print("readout stacks", r)
	
	print("pop...")
	nt = pop(bb, stack_index=0, prefix=prefix, verbose=True)
	print("top...", nt)	
	node_area_index, block_index = top(bb, stack_index=0, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[1,1], top_areas=[node_area_index,0], prefix=prefix)
	print("readout stacks", rr)
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[1,1], top_areas=[nt,0], prefix=prefix)
	print("readout stacks", rr)
	
def test_pop3():
	prefix = "G"
	stacks = [[0,1], [2]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,1], top_areas=[0,0], prefix=prefix)
	print("readout stacks", r)
	
	print("pop...")
	nt = pop(bb, stack_index=1, prefix=prefix, verbose=True)
	print("top...", nt)	
	node_area_index, block_index = top(bb, stack_index=1, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,0], top_areas=[0,node_area_index], prefix=prefix)
	print("readout stacks", rr)
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,0], top_areas=[0,nt], prefix=prefix)
	print("readout stacks", rr)

def test_pop4():
	prefix = "G"
	stacks = [[1,0], [4,2]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,2], top_areas=[0,0], prefix=prefix)
	print("readout stacks", r)
	
	print("\npop...")
	nt = pop(bb, stack_index=1, prefix=prefix, verbose=True)
	print("top...", nt)	
	node_area_index, block_index = top(bb, stack_index=1, prefix=prefix, verbose=False)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,1], top_areas=[0,node_area_index], prefix=prefix)
	print("readout stacks", rr)
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,1], top_areas=[0,nt], prefix=prefix)
	print("readout stacks", rr)

	print("\npop...")
	nt = pop(bb, stack_index=1, prefix=prefix, verbose=True)
	print("top...", nt)	
	node_area_index, block_index = top(bb, stack_index=1, prefix=prefix, verbose=False)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,0], top_areas=[0,node_area_index], prefix=prefix)
	print("readout stacks", rr)
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,0], top_areas=[0,nt], prefix=prefix)
	print("readout stacks", rr)



def test_get_project_map7():
	bb = BlocksBrain(blocks_number=10, other_areas=["H", "N0", "N1", "N2"])
	bb.disinhibit_area(area_name="BLOCKS", lock=0)
	bb.disinhibit_area(area_name="H", lock=0)
	bb.disinhibit_area(area_name="N0", lock=0)
	bb.disinhibit_area(area_name="N1", lock=0)
	bb.disinhibit_area(area_name="N2", lock=0)

	bb.activate_block(index=0)
	bb.activate_assembly(index=5, activation_area="N0")
	bb.activate_assembly(index=6, activation_area="N1")
	
	bb.disinhibit_fiber(area1="BLOCKS", area2="N1", lock=0)
	bb.disinhibit_fiber(area1="N0", area2="N1", lock=0)
	bb.disinhibit_fiber(area1="N0", area2="N2", lock=0)

	pmap = bb.get_project_map()
	print("\nproject map", pmap)
	bb.brain.project({}, area_to_area=pmap, verbose=True)
	


def test_put1():
	prefix = "G"
	stacks = [[0,1]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=10, other_areas=oa, p=0.1, eak=100, nean=1e5, neak=100, db=0.05)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('\tparsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2], top_areas=[0], prefix=prefix)
	print("readout stacks", r)
	node_area_index, block_index = top(bb, stack_index=0, prefix=prefix, verbose=True)
	print("top node_area_index={}, block_index={}".format(node_area_index, block_index))

	print("\nput...")
	nt = put(bb, stack_index=0, block=5, prefix=prefix, verbose=True)
	print("new top node area should be", nt)	
	node_area_index, block_index = top(bb, stack_index=0, prefix=prefix, verbose=True)
	print("decoded top node_area_index={}, block_index={}".format(node_area_index, block_index))
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[len(s)+1 for s in stacks], top_areas=[node_area_index], prefix=prefix)
	print("readout stacks", rr)
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[len(s)+1 for s in stacks], top_areas=[nt], prefix=prefix)
	print("readout stacks", rr)

def test_put2():
	prefix = "G"
	stacks = [[0,1], [2]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,1], top_areas=[0,0], prefix=prefix)
	print("readout stacks", r)
	
	print("put...")
	nt = put(bb, stack_index=0, block=3, prefix=prefix, verbose=True)
	print("new top node area", nt)	
	node_area_index, block_index = top(bb, stack_index=0, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[1,1], top_areas=[node_area_index,0], prefix=prefix)
	print("readout stacks", rr)
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[1,1], top_areas=[nt,0], prefix=prefix)
	print("readout stacks", rr)
	
def test_put3():
	prefix = "G"
	stacks = [[0,1], [2]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,1], top_areas=[0,0], prefix=prefix)
	print("readout stacks", r)
	
	print("put...")
	nt = put(bb, stack_index=1, block=3, prefix=prefix, verbose=True)
	print("new top node area", nt)	
	node_area_index, block_index = top(bb, stack_index=1, prefix=prefix, verbose=True)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,0], top_areas=[0,node_area_index], prefix=prefix)
	print("readout stacks", rr)
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,0], top_areas=[0,nt], prefix=prefix)
	print("readout stacks", rr)

def test_put4():
	prefix = "G"
	stacks = [[1,0], [4,2]]
	print("\ninput stacks:", stacks)
	oa = add_prefix(regions=[item for sublist in REGIONS for item in sublist], prefix=prefix)
	oa = oa + [RELOCATED]
	print("oa:", oa)
	bb = BlocksBrain(blocks_number=5, other_areas=oa, p=0.1, eak=20, nean=500, neak=20, db=0.2)	
	print('parsing...')
	parse(bb, stacks=stacks, prefix=prefix)
	print('parsing finished.')
	r = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,2], top_areas=[0,0], prefix=prefix)
	print("readout stacks", r)
	
	print("\nput...")
	nt = put(bb, stack_index=1, block=3, prefix=prefix, verbose=True)
	print("new top node area", nt)	
	node_area_index, block_index = top(bb, stack_index=1, prefix=prefix, verbose=False)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,1], top_areas=[0,node_area_index], prefix=prefix)
	print("readout stacks", rr)
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,1], top_areas=[0,nt], prefix=prefix)
	print("readout stacks", rr)

	print("\nput...")
	nt = put(bb, stack_index=1, block=0, prefix=prefix, verbose=True)
	print("new top node area", nt)	
	node_area_index, block_index = top(bb, stack_index=1, prefix=prefix, verbose=False)
	print("node_area_index={}, block_index={}".format(node_area_index, block_index))
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,0], top_areas=[0,node_area_index], prefix=prefix)
	print("readout stacks", rr)
	rr = readout(bb, stacks_number=len(stacks), stacks_lengths=[2,0], top_areas=[0,nt], prefix=prefix)
	print("readout stacks", rr)


if __name__ == "__main__":
	# To run test, first set MAX_STACKS = 5 in script bw_apps.py

	# test_readout2()
	# test_readout3()
	# test_readout5()

	test_readout6()
	
	# test_put1()
	# test_put2()
	# test_put3()
	# test_put4()




