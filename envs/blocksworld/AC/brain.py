from scipy.stats import binom, truncnorm
import random
import numpy as np
from collections import defaultdict

VERBOSE_PROJECT = False


class Area:
	def __init__(self, name, n=1e6, k=50, beta=0.1, p=0.1):
		self.name = name
		self.n = n # number of neurons
		self.k = k # capk
		self.beta = beta # plasticity (learning rate)
		self.stimulus_beta = {} # plasticity from stimuli to this area
		self.area_beta = {} # plasticity from areas to this area
		self.w = 0 # number of winners, TODO: isnt this equal to k?
		self.new_w = 0 # number of new winners
		self.winners = np.array([]) # list of winners
		self.new_winners = np.array([]) # list of new winners
		self.saved_w = np.array([])
		self.saved_winners = np.array([])
		self.num_first_winners = 0 # initial number of winners
		self.fixed_assembly = False # whether to freeze/fix assembly/winners in this area
		self.explicit = False # whether to run simulation explicitly. 

	def update_winners(self):
		self.winners = self.new_winners
		if (not self.explicit):
			self.w = self.new_w

	def update_stimulus_beta(self, stim_name, new_beta):
		self.stimulus_beta[stim_name] = new_beta

	def update_area_beta(self, from_area_name, new_beta):
		self.area_beta[from_area_name] = new_beta

	def fix_assembly(self):
		assert len(self.winners) > 0, "Area {} does not have assembly, cannot fix.".format(self.name)
		self.fixed_assembly = True

	def unfix_assembly(self):
		self.fixed_assembly = False

	def print_info(self):
		print("\nArea {} info: \
				\n\tn={},\
				\n\tk={},\
				\n\tbeta={},\
				\n\tstimulus_beta={},\
				\n\tarea_beta={},\
				\n\tw={},\
				\n\tnew_w={},\
				\n\tnum_first_winners={},\
				\n\tsaved_w={},\
				\n\twinners={},\
				\n\tnew_winners={},\
				\n\tsaved_winners={},\
				\n\tfixed_assembly={},\
				\n\texplicit={}".format(self.name, self.n, self.k, self.beta, 
					self.stimulus_beta, self.area_beta, 
					self.w, self.new_w, self.num_first_winners, self.saved_w, 
					self.winners, self.new_winners, self.saved_winners, 
					self.fixed_assembly, self.explicit))


class Stimulus():
	def __init__(self, k):
		self.k = k
	def print_info(self):
		print("\nStimulus info: k={}".format(self.k))



class Brain():
	def __init__(self, p):
		self.areas = {} # list of Area
		self.stimuli = {} # list of Stimulus
		self.stimuli_connectomes = {} # size {num_stimuli: [num_areas, num_neurons]}
		self.connectomes = {} # size {num_areas: {num_areas: [num_neurons, num_neurons]}}
		self.p = p
		self.save_size = True
		self.save_winners = False
		self.no_plasticity = False

	def print_info(self):
		print("\nBrain info: \
				\n\tareas={},\
				\n\tstimuli={},\
				\n\tstimuli_connectomes={},\
				\n\tconnectomes={},\
				\n\tp={},\
				\n\tsave_size={},\
				\n\tsave_winners={},\
				\n\tno_plasticity={}".format(self.areas, self.stimuli,
											self.stimuli_connectomes, self.connectomes,
											self.p, self.save_size, self.save_winners,
											self.no_plasticity))

	def add_stimulus(self, stim_name, k):
		self.stimuli[stim_name] = Stimulus(k=k)
		new_connectomes = {} # will be size {num_stimuli: {num_area: [num_neurons]}}
		for Aname, A in self.areas.items():
			if A.explicit:
				new_connectomes[Aname] = binom.rvs(k, self.p, sample=[A.n]) # each element is a number of successes in a A.n-trial sample
			else:
				new_connectomes[Aname] = np.array([])
			A.stimulus_beta[stim_name] = A.beta
		self.stimuli_connectomes[stim_name] = new_connectomes

	def add_nonexplicit_area(self, area_name, n, k, beta):
		'''
		All connections from the stimuli and from/to all areas of the brain are initialized as empty,
		but they will be set during project phase in order to improve performance.
		'''
		self.areas[area_name] = Area(name=area_name, n=n, k=k, beta=beta)
		for stim_name, stim_connectomes in self.stimuli_connectomes.items():
			stim_connectomes[area_name] = np.array([])
			self.areas[area_name].stimulus_beta[stim_name] = beta
		new_connectomes = {} # will be size {num_areas: [num_neurons, num_neurons]}
		for Aname, A in self.areas.items():
			other_area_size = 0
			if A.explicit:
				other_area_size = A.n
			new_connectomes[Aname] = np.zeros([0, other_area_size])
			if Aname != area_name:
				self.connectomes[Aname][area_name] = np.zeros([other_area_size, 0])
			A.area_beta[area_name] = A.beta
			self.areas[area_name].area_beta[Aname] = beta
		self.connectomes[area_name] = new_connectomes
			

	def add_explicit_area(self, area_name, n, k, beta):
		'''
		BLOCKS area is usually an explicit area.
		Since the area is explicit, the weights of all connections 
		from a stimuli of the brain the new area are initially set randomly according to a 
		binomial distribution with parameters the value k of the stimulus and the value p of the brain
		(with the default plasticity). The weights of all connections from/to an explicit area of the brain
		to the new area are initially and fully set randomly according to a binomial distribution 
		with parameters 1 and the value p of the brain. The weights of all connections from/to a 
		non-explicit area of the brain to the new area are initially set to empty. 
		In all cases, the plasticity of the connections is set to the default plasticity.
		The number of winners of the new area is set equal to the number of its neurons.
		'''
		self.areas[area_name] = Area(name=area_name, n=n, k=k, beta=beta)
		self.areas[area_name].explicit = True
		for stim_name, stim_connectomes in self.stimuli_connectomes.items():
			stim_connectomes[area_name] = binom.rvs(self.stimuli[stim_name].k, self.p, size=[n]) # sample
			self.areas[area_name].stimulus_beta[stim_name] = beta
		new_connectomes = {} # will be size {num_areas: [num_neurons, num_neurons]}
		for Aname, A in self.areas.items():
			if Aname==area_name:
				new_connectomes[Aname] = binom.rvs(1, self.p, size=[n,n]) # sample
			else:
				if A.explicit:
					An = A.n
					new_connectomes[Aname] = binom.rvs(1, self.p, size=[n, An]) # sample
					self.connectomes[Aname][area_name] = binom.rvs(1, self.p, size=[An, n]) # sample
				else:
					new_connectomes[Aname] = np.zeros([n, 0])
					self.connectomes[Aname][area_name] = np.zeros([0, n])
			A.area_beta[area_name] = A.beta
			self.areas[area_name].area_beta[Aname] = beta
		self.connectomes[area_name] = new_connectomes
		self.areas[area_name].w = n

	def __compute_potential_new_winners(self, to_area, total_k, p):
		'''
		Compute the potential new k winners of the area to which we are going to project.
		To this aim compute the threshold alpha for inputs that are above (n-k)/n percentile, 
		use the normal approximation, between alpha and total_k, round to integer, and 
		create the k potential_new_winners.
		'''
		effective_n = to_area.n - to_area.w	 
		alpha = binom.ppf((effective_n - to_area.k) / effective_n, total_k, p)
		print("\nalpha: {}".format(alpha)) if VERBOSE_PROJECT else 0
		reset = False
		if (effective_n - to_area.k) / effective_n < 0:
			print("\n(effective_n-to_area.k)/effective_n={}, alpha={}, resetting alpha to 0. effective_n={}, to_area.n={}, to_area.k={}, to_area.w={}, total_k={}, p={}".format((effective_n - to_area.k) / effective_n, alpha, effective_n, to_area.n, to_area.k, to_area.w, total_k, p))
			alpha = 0
			reset = True
		std = np.sqrt(total_k * p * (1.0 - p))
		mu = total_k * p
		a = (alpha - mu) / std
		b = (total_k - mu) / std
		print("\na: {}, b: {}, mu: {}, std: {}".format(a, b, mu, std)) if (VERBOSE_PROJECT or reset) else 0
		
		potential_new_winners = truncnorm.rvs(a=a, b=b, loc=mu, scale=std, size=to_area.k) # sample

		potential_new_winners = np.round(potential_new_winners) # decimals=0
		print("\nlen(potential_new_winners): {}, max(potential_new_winners): {}, potential_new_winners: {}".format(len(potential_new_winners), max(potential_new_winners), potential_new_winners)) if VERBOSE_PROJECT else 0
		return np.array(potential_new_winners, dtype=np.float32)


	def project_into(self, to_area, from_stimuli, from_areas, VERBOSE_PROJECT=VERBOSE_PROJECT):
		print("\n\nProjecting from_stimuli: {}, from_areas: {}, to_area.name: {}".format(from_stimuli, from_areas, to_area.name)) if VERBOSE_PROJECT else 0
		for from_area in from_areas:
			if not (len(self.areas[from_area].winners != 0) and (self.areas[from_area].w != 0)): # TODO it was assertion
				print("\tWarning: Projecting from area {} with no winners".format(from_area)) if VERBOSE_PROJECT else 0
				return 0
		# Compute previous inputs from the winners of the areas from which we project to the
		# current w winners of the area to which we project (indexed from 1 to w). In particular, 
		# for each winner i of the area to which we project, its total input is computed by summing 
		# the weights of the connectomes connecting either a stimulus or a winner of an area from 
		# which we project to it.			
		name = to_area.name
		prev_winner_inputs = np.zeros(to_area.w)
		for stim in from_stimuli:
			stim_inputs = self.stimuli_connectomes[stim][name]
			prev_winner_inputs[:to_area.w] += stim_inputs[:to_area.w]
		for from_area in from_areas:
			connectome = self.connectomes[from_area][name]
			for w in self.areas[from_area].winners:
				prev_winner_inputs[:to_area.w] += connectome[w, :to_area.w]	
		print("\nlen(prev_winner_inputs): {}".format(prev_winner_inputs.shape[0])) if VERBOSE_PROJECT else 0
		# Simulate to_area.k potential new winners if the area is not explicit.
		if not to_area.explicit:
			# Compute the number of input stimuli and areas, the total number of input connectomes,
			# and the number of input connectomes for each input stimulus and area.
			
			stim_input_sizes = [self.stimuli[stim].k for stim in from_stimuli]
			area_input_sizes = [self.areas[from_area].winners.shape[0] for from_area in from_areas]
			input_sizes = stim_input_sizes + area_input_sizes
			total_k = sum(input_sizes)
			num_inputs = len(input_sizes)

			assert (num_inputs == len(from_stimuli) + len(from_areas)), "The number of inputs should be the sum of winners in source stimuli and areas"
			print("\ntotal_k: {}, input_sizes: {}".format(total_k, input_sizes)) if VERBOSE_PROJECT else 0
			potential_new_winners = self.__compute_potential_new_winners(to_area, total_k, self.p)
			all_potential_winners = np.concatenate((prev_winner_inputs, potential_new_winners))
		else:
			all_potential_winners = prev_winner_inputs
		print("\nall_potential_winners: {}".format(all_potential_winners)) if VERBOSE_PROJECT else 0
		new_winner_indices = np.argsort(all_potential_winners)[::-1][:to_area.k] # descending
		print("\nnew_winner_indices: {}".format(new_winner_indices)) if VERBOSE_PROJECT else 0
		num_first_winners = 0	
		if (not to_area.explicit):
			first_winner_inputs = []
			for i in range(to_area.k):
				if new_winner_indices[i] >= to_area.w:
					first_winner_inputs.append(potential_new_winners[new_winner_indices[i] - to_area.w])
					num_first_winners += 1
					new_winner_indices[i] = to_area.w + num_first_winners - 1
		print("\nnum_first_winners: {}".format(num_first_winners)) if VERBOSE_PROJECT else 0
		to_area.new_winners = new_winner_indices
		to_area.new_w = to_area.w + num_first_winners
		if to_area.fixed_assembly:
			to_area.new_winners = to_area.winners
			to_area.new_w = to_area.w
			first_winner_inputs = []
			num_first_winners = 0
		print("\nto_area.new_winners: {}, to_area.new_w: {}".format(to_area.new_winners, to_area.new_w)) if VERBOSE_PROJECT else 0
		first_winner_to_inputs = {}
		for i in range(num_first_winners):
			input_indices = random.sample(range(total_k), int(first_winner_inputs[i])) # sample without replacement
			inputs = np.zeros(num_inputs)
			total_so_far = 0
			for j in range(num_inputs):
				
				inputs[j] = sum((total_so_far + input_sizes[j]) > (w-1) >= total_so_far for w in input_indices) # TODO

				total_so_far += input_sizes[j]
			first_winner_to_inputs[i] = inputs
		m = 0
		no_plasticity = self.no_plasticity
		tmp_zeros = np.zeros(num_first_winners)
		for stim in from_stimuli:
			if num_first_winners > 0:
				self.stimuli_connectomes[stim][name] = np.cat((self.stimuli_connectomes[stim][name], tmp_zeros))
				self.stimuli_connectomes[stim][name][to_area.w:to_area.w + num_first_winners] = first_winner_to_inputs[:num_first_winners, m]
			stim_to_area_beta = to_area.stimulus_beta[stim]
			if no_plasticity:
				stim_to_area_beta = 0.0
			stim_multiplier = (1 + stim_to_area_beta)
			self.stimuli_connectomes[stim][name][to_area.new_winners] *= stim_multiplier
			print("\nstim: {}, name: {}, self.stimuli_connectomes[stim][name]: {}".format(stim, name, self.stimuli_connectomes[stim][name])) if VERBOSE_PROJECT else 0
			m += 1
		for from_area in from_areas:
			from_area_w = self.areas[from_area].w
			from_area_winners = self.areas[from_area].winners
			nr, nc = self.connectomes[from_area][name].shape
			print("\nExcuting padding of connectomes from {} to {} by adding {} columns".format(from_area, to_area.name, num_first_winners)) if VERBOSE_PROJECT else 0
			self.connectomes[from_area][name] = np.concatenate((self.connectomes[from_area][name], np.zeros([nr, num_first_winners])), axis=1) # hcat
			from_area_winners_list = from_area_winners.tolist()


			not_from_area_winners = [j for j in range(from_area_w) if j not in from_area_winners_list] # TODO
			size_not_from_area_winners = len(not_from_area_winners) # TODO

			for i in range(num_first_winners):
				total_in = first_winner_to_inputs[i][m]
				sample_indices = random.sample(from_area_winners_list, int(total_in)) # no replacement

				in_sample_indices = [j for j in range(from_area_w) if j in sample_indices] # TODO
				self.connectomes[from_area][name][in_sample_indices, to_area.w + i] = 1.0 # TODO
				self.connectomes[from_area][name][not_from_area_winners, to_area.w + i] = binom.rvs(1, self.p, size=[size_not_from_area_winners]) # TODO
				

			area_to_area_beta = to_area.area_beta[from_area]
			if no_plasticity:
				area_to_area_beta = 0.
			print("\nPlasticity in projecting {} --> {} is {}".format(from_area, name, area_to_area_beta)) if VERBOSE_PROJECT else 0
			
			indices = np.array([(i, j) for i in from_area_winners for j in to_area.new_winners]) # TODO
			self.connectomes[from_area][name][indices[:, 0], indices[:, 1]] *= (1.0 + area_to_area_beta) # TODO
			
			print("\nfrom_area: {}, name: {}, self.connectomes[from_area][name]: {}".format(from_area, name, self.connectomes[from_area][name])) if VERBOSE_PROJECT else 0
			m += 1
		self.connectomes[name] = self.connectomes.get(name, {}) 
		dummy_zeros = np.zeros([0,0])
		for other_area in self.areas.keys():
			if not (other_area in from_areas):
				self.connectomes[other_area] = self.connectomes.get(other_area, {})
				con = self.connectomes[other_area].get(name, dummy_zeros)
				nr, nc = con.shape
				print("\nExecuting padding of connectomes from {} to {} by adding {} columns".format(other_area, name, num_first_winners)) if VERBOSE_PROJECT else 0
				self.connectomes[other_area][name] = np.concatenate((con, np.zeros([nr, num_first_winners])), axis=1) # hcat
				num_samples_j = self.areas[other_area].w
				num_samples_i = to_area.new_w - to_area.w
				samples = binom.rvs(1, self.p, size=[num_samples_j, num_samples_i])
				self.connectomes[other_area][name][:num_samples_j, to_area.w:to_area.new_w] = samples
			cno = self.connectomes[name].get(other_area, dummy_zeros) 
			nr, nc = cno.shape
			print("\nExecuting padding of connectomes from {} to {} by adding {} rows".format(name, other_area, num_first_winners)) if VERBOSE_PROJECT else 0
			self.connectomes[name][other_area] = np.concatenate((cno, np.zeros([num_first_winners, nc])), axis=0) # vcat
			print("name: {}, other_area: {}, self.connectomes[name][other_area].shape: {}".format(name, other_area, self.connectomes[name][other_area].shape)) if VERBOSE_PROJECT else 0
			columns = self.connectomes[name][other_area].shape[1]
			print("\nto_area.w: {}, to_area.new_w: {}".format(to_area.w, to_area.new_w)) if VERBOSE_PROJECT else 0
			num_samples_i = to_area.new_w - to_area.w
			num_samples_j = columns
			samples = binom.rvs(1, self.p, size=[num_samples_i, num_samples_j])
			self.connectomes[name][other_area][to_area.w:to_area.new_w, :columns] = samples
			print("\nname: {}, other_area: {}, self.connectomes[name][other_area]: {}".format(name, other_area, self.connectomes[name][other_area])) if VERBOSE_PROJECT else 0
		return num_first_winners


	def project(self, stim_to_area, area_to_area, verbose=VERBOSE_PROJECT):
		'''
		Execute the project from stimuli and/or areas to areas. For each stimulus (key) in the first dictionary,
		the list (value) of areas to which the stimulus has the project is specified. For each area (key),
		in the second dictionary, the list (value) of areas to which the area has the project is specified.
		The function collects, for each area, the list of stimuli and areas that project to it (basically, it
		computes the inverse of the input mappings). Then, for each area which has to "receive" a projection
		(from either stimuli or areas), it invokes the function which actually performs the projection (this
		function returns the number of winners in the destination area). If the new winners have to be later
		analysed, then their list is appended to the the list of lists of winners of the area. When everything
		has been done, the function updates the destination areas.
		'''
		print("\n\nstim_to_area: {}, area_to_area: {}".format(stim_to_area, area_to_area)) if VERBOSE_PROJECT else 0
		stim_in = defaultdict(set)
		area_in = defaultdict(set)

		stim_keys = set(self.stimuli.keys())
		area_keys = set(self.areas.keys())
		
		for (stim, areas) in stim_to_area.items():
			if stim not in stim_keys:
				raise ValueError("{} is not in the stimuli of the brain!".format(stim))
			for area in areas:
				if area not in area_keys:
					raise ValueError("{} is not in the areas of the brain!".format(area))
				stim_in[area].add(stim)
		for (from_area, to_areas) in area_to_area.items():
			if from_area not in area_keys:
				raise ValueError("{} is not in the areas of the brain".format(from_area))
			for to_area in to_areas:
				if to_area not in area_keys:
					raise ValueError("{} is not in the areas of the brain".format(to_area))
				area_in[to_area].add(from_area)

		to_update = set(stim_in.keys()).union(set(area_in.keys())) 
		print("\nto_update: {}".format(to_update)) if verbose else 0
		print("\narea_in: {}".format(area_in)) if verbose else 0
		for area in to_update: 
			num_first_winners = self.project_into(self.areas[area], stim_in.get(area, {}), area_in.get(area, {}))
			self.areas[area].num_first_winners = num_first_winners
			if self.save_winners:
				self.areas[area].saved_winners = np.concatenate((self.areas[area].saved_winners, np.array([self.areas[area].new_winners], dtype=np.float32)))
		for area in to_update:
			self.areas[area].update_winners()
			if self.save_size:
				self.areas[area].saved_w = np.concatenate((self.areas[area].saved_w, np.array([self.areas[area].w], dtype=np.float32)))
























