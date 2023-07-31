from operator import attrgetter
import sys
import random
import json
import numpy as np

import logging
logger = logging.getLogger(__file__)

from sqlalchemy import Float, Integer, String
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql.expression import cast
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy import and_, not_, func

from dallinger import transformations
from dallinger.information import Gene, Meme, State
from dallinger.nodes import Agent, Environment, Source
from dallinger.models import Info, Network, Participant, Node
from dallinger import db
import pysnooper

class ParticleFilter(Network):
	"""Discrete fixed size generations with random transmission"""

	__mapper_args__ = {"polymorphic_identity": "particlefilter"}

	def __init__(self, generation_size, generations, replication, condition, decision_index, role):
		"""Endow the network with some persistent properties."""
		self.current_generation = 0
		self.generation_size = generation_size
		self.max_size = generations * generation_size + 2 # add one to account for initial_source; one for random_attributes
		self.replication = replication
		self.condition = condition
		self.decision_index = decision_index
		self.role = role
		self.set_randomisation()
	
	@hybrid_property
	def current_generation(self):
		"""Make property1 current_generation."""
		return int(self.property1)

	@current_generation.setter
	def current_generation(self, current_generation):
		"""Make current_generation settable."""
		self.property1 = repr(current_generation)

	@current_generation.expression
	def current_generation(self):
		"""Make current_generation queryable."""
		return cast(self.property1, Integer)

	@hybrid_property
	def generation_size(self):
		"""Make property2 generation_size."""
		return int(self.property2)

	@generation_size.setter
	def generation_size(self, generation_size):
		"""Make generation_size settable."""
		self.property2 = repr(generation_size)

	@generation_size.expression
	def generation_size(self):
		"""Make generation_size queryable."""
		return cast(self.property2, Integer)
	
	@hybrid_property
	def replication(self):
		"""Make property3 replication."""
		return int(self.property3)

	@replication.setter
	def replication(self, replication):
		"""Make replication settable."""
		self.property3 = repr(replication)

	@replication.expression
	def replication(self):
		"""Make replication queryable."""
		return cast(self.property3, Integer)

	@hybrid_property
	def decision_index(self):
		"""Make property4 decision_index."""
		return int(self.property4)

	@decision_index.setter
	def decision_index(self, decision_index):
		"""Make decision_index settable."""
		self.property4 = repr(decision_index)

	@decision_index.expression
	def decision_index(self):
		"""Make decision_index queryable."""
		return cast(self.property4, Integer)

	@hybrid_property
	def condition(self):
		"""Make property5 condition."""
		return self.property5

	@condition.setter
	def condition(self, condition):
		"""Make condition settable."""
		self.property5 = condition

	@condition.expression
	def condition(cls):
		"""Make condition queryable."""
		return cls.property5

	def log(self, text, key="?????"):
		"""Print a string to the logs."""
		print(">>>> {} {}".format(key, text))
		sys.stdout.flush()

	def set_randomisation(self):
		n = float(self.generation_size)
		self.payout_colours = dict(zip(range(int(n)), (["left"] * int(n / 2.)) + (["right"] * int(n / 2.))))
		self.button_orders = dict(zip(range(int(n)), (["left"] * int(n / 2.)) + (["right"] * int(n / 2.))))

	def slots(self, generation = None, ignore_id = None, assigned = False, full = False, shadow = False):

		query = (
			db.session.query(Particle).join(Participant)
			.filter(Particle.network_id == self.id)
			.filter(Particle.failed == False)
			)

		# this func is for assigned | full | assigned and full
		assert assigned or full

		if generation is not None:
			query = query.filter(Particle.property2 == repr(generation))

		# Don't count this participant
		if ignore_id is not None:
			query = query.filter(Participant.id != ignore_id)

		# + slots that have been asigned
		# - slots that have been filled 
		if assigned and not full:
			query = query.filter(Participant.status != "approved")

		# + slots that have been asigned
		# + slots that have been filled
		if assigned and full:
			pass

		# - slots that have been asigned
		# + slots that have been filled
		if not assigned and full:
			query = query.filter(Participant.status == "approved")

		if shadow:
			query = query.filter(Particle.property5.contains("OVF"))

		else:
			query = query.filter(not_(Particle.property5.contains("OVF")))

		return [int(node.property1) for node in query.all()]

	def slot_occupancy(self, generation, ignore_id = None):
		"""How many participants (approved or still working) have been 
		(provisionally or definitively) assigned to any already-assigned slot?"""
		slot_counts = (
			db.session.query(func.count(Particle.participant_id.distinct()).label('count'), Particle.property1) 
			.group_by(Particle.property1) 
			.filter(Particle.property2 == repr(generation)) 
			.filter(Particle.network_id == self.id) 
			.filter_by(failed = False)
		)

		if ignore_id is not None:
			slot_counts = slot_counts.filter(Particle.network_id == self.id) 
		# [(slot, count), ...]
		return [c[::-1] for c in slot_counts.all()]

	def full_slots(self, generation = None, ignore_id = None, shadow = False):
		if generation is None:
			generation = int(self.current_generation)
		return self.slots(generation = generation, ignore_id = ignore_id, shadow = shadow, full = True, assigned = False)
		
	def assigned_slots(self, generation = None, ignore_id = None, shadow = False):
		if generation is None:
			generation = int(self.current_generation)
		return self.slots(generation = generation, ignore_id = ignore_id, shadow = shadow, full = False, assigned = True)

	# @pysnooper.snoop()
	def overoccupied_slots(self, generation, ignore_id = None):
		"""Returns all slots that have been *provisionally asigned to shadow participants* @ generation"""
		slot_counts = self.slot_occupancy(generation, ignore_id = ignore_id)

		# return only slots with more than one participant 
		return [int(shadow_slot) for (shadow_slot, occupancy) in slot_counts if occupancy > 1]

	def availability(self, generation = None):
		if generation is None:
			generation = int(self.current_generation)
		
		full_slots = self.full_slots(generation = generation)
		if len(full_slots) == int(self.generation_size):
			return "full"

		assigned_slots = self.assigned_slots(generation = generation)
		if len(assigned_slots) + len(full_slots) < int(self.generation_size):
			return "recruiting"

		return "accepting"

	def log_slots(self, generation = None,
						assigned_slots = None,
						full_slots = None,
						overassigned_slots = None,
						chosen_slot = None,
						show_shadow = True,
						chosen_shadow = None,
						show_pointer = False,
						ignore_id = None,
						ignore_shadow_id = None,
					):
		return
		key = "models.py >> log_slots: "
		if generation is None:
			generation = int(self.current_generation)
		if assigned_slots is None:
			assigned_slots = self.assigned_slots(int(generation))
		if full_slots is None:
			full_slots = self.full_slots(int(generation))
		if overassigned_slots is None:
			overassigned_slots = self.overoccupied_slots(int(generation), ignore_id = ignore_shadow_id)
		schematic = ["-"] * int(self.generation_size)
		for slot in assigned_slots:
			schematic[int(slot)] = "+"
		for slot in full_slots:
			schematic[int(slot)] = "*"
		if chosen_slot is not None:
			schematic[int(chosen_slot)] = "@"
		self.log("{};R{};G{}: {}".format(self.condition, self.replication, generation, schematic), key)

		if show_pointer:
			pointer_schematic = [":"] * self.generation_size
			if chosen_slot is not None:
				pointer_schematic[int(chosen_slot)] = "^"
			if chosen_shadow is not None:
				pointer_schematic[int(chosen_shadow)] = "v"
			self.log("{};R{};G{}: {}".format(self.condition, self.replication, generation, pointer_schematic), key)            
	
		if show_shadow:
			shadow_schematic = ["-"] * int(self.generation_size)
			for slot in overassigned_slots:
				shadow_schematic[int(slot)] = "$"
			if chosen_shadow is not None:
				shadow_schematic[int(chosen_shadow)] = "@"
			self.log("{};R{};G{}: {}".format(self.condition, self.replication, generation, shadow_schematic), key)
		
	def assign_slot(self, participant):
		key = "models.py >> assign_slot: "
		"""Assign a *provisional* slot to a new participant """
		
		# slots assigned to an approved participant
		full_slots = self.full_slots(self.current_generation)

		# slots that have been assigned to any not-yet-approved participant(s)
		assigned_slots = self.assigned_slots(self.current_generation)
		
		node_slots_still_availible = [slot for slot in range(self.generation_size) if ((slot not in assigned_slots) and (slot not in full_slots))]
		
		if not node_slots_still_availible:
			self.log("All slots have been filled or already assigned in network {} [Cond: {}; Rep: {}; Gen: {}]".format(self.id, self.condition, self.replication, self.current_generation), key)
			multi_occupancy_slots = self.overoccupied_slots(self.current_generation)
			
			unoccupied_shadow_slots = [slot for slot in range(self.generation_size) if ((slot not in full_slots) and (slot not in multi_occupancy_slots))]
			
			if not unoccupied_shadow_slots:
				unfilled_slots = [slot for slot in range(self.generation_size) if slot not in full_slots]

				if not unfilled_slots:
					random_slot = random.choice(list(range(self.generation_size)))
					self.log("No slots remain unfilled for Participant {} in network {} [Cond: {}; Rep: {}; Gen: {}]. Assigning shadow slot {} at random.".format(participant.id, self.id, self.condition, self.replication, self.current_generation, random_slot), key)
					return random_slot
				
				random_slot = random.choice(unfilled_slots)
				self.log_slots(assigned_slots = assigned_slots, chosen_shadow = random_slot, full_slots = full_slots, overassigned_slots = multi_occupancy_slots)
				self.log("No empty experimental or empty shadow slots availible for Participant {} in network {} [Cond: {}; Rep: {}; Gen: {}]. Provisionally assigning shadow slot {} at random from all {} unfilled slots.".format(participant.id, self.id, self.condition, self.replication, self.current_generation, random_slot, len(unfilled_slots)), key)
				return random_slot

			random_slot = random.choice(unoccupied_shadow_slots)
			self.log_slots(assigned_slots = assigned_slots, chosen_shadow = random_slot, full_slots = full_slots, overassigned_slots = multi_occupancy_slots)
			self.log("No Epxerimental slots availible for Participant {} in network {} [Cond: {}; Rep: {}; Gen: {}]. Provisionally assigning Shadow slot {} at random (from {} availible).".format(participant.id, self.id, self.condition, self.replication, self.current_generation, random_slot, len(unoccupied_shadow_slots)), key)
			return random_slot
		
		random_slot = random.choice(node_slots_still_availible)
		self.log_slots(assigned_slots = assigned_slots, chosen_slot = random_slot, full_slots = full_slots)
		self.log("{} Epxerimental slots remain availible for Participant {} in network {} [Cond: {}; Rep: {}; Gen: {}]. Provisionally assigning slot {} at random.".format(len(node_slots_still_availible), participant.id, self.id, self.condition, self.replication,self.current_generation, random_slot), key)
		return random_slot

	def reassign_slot(self, current_slot, generation):
		"""Choose a new slot for a participant who's provisionally assigned slot is taken"""
		key = "models.py >> reassign_slot: "
		relevant_variables = json.loads(NetworkRandomAttributes.query.filter_by(network_id = self.id).one().details)
		payout_colors, button_orders = relevant_variables["payout_color"], relevant_variables["button_order"]
		equivelant_slots = [
			int(k) for (k, v) in payout_colors.items() 
			if ((v == payout_colors[str(current_slot)]) 
			and (int(k) != int(current_slot)) 
			and (button_orders[str(k)] == button_orders[str(current_slot)]))
		]
		self.log("There are {} slots with equivelant randomisation conditions to slot {}".format(len(equivelant_slots), current_slot), key)
		availible_slots = [slot for slot in equivelant_slots if slot not in self.full_slots(generation)]
		self.log("{} of the {} slots equivelant to slot {} are availible.".format(len(availible_slots), len(equivelant_slots), current_slot), key)
		return None if not availible_slots else random.choice(availible_slots)

	def assign_to_overflow(self, participant_id):
		key = "models.py >> assign_to_overflow ({}): ".format(participant_id)
		participant = Participant.query.get(participant_id)
		if not participant:
			self.log("Participant object not found for participant: {}".format(participant_id, key))
			return
		
		nodes = participant.nodes(type = Particle)
		if not nodes:
			self.log("No nodes found for participant: {}".format(participant_id, key))
			return
		
		for participant_node in nodes:
			if not ("OVF" in participant_node.property5):
				participant_node.property5 = participant_node.property5 + ":OVF"
		self.log("Assigned all {} of participants {}'s nodes to the overflow.".format(len(nodes), participant_id), key)

	def freeze_generation(self, generation):
		"""Automatically assign any remaining non-approved particiapnts to the overflow"""
		key = "models.py >> freeze_generation ({}): ".format(generation)
		participant_ids = (
			db.session.query(Particle.participant_id).join(Participant).join(ParticleFilter)
			.filter(Participant.status != "approved")
			.filter(Particle.failed == False)
			.filter(Particle.property2 == repr(generation))
			.filter(not_(Particle.property5.contains("OVF")))
			.distinct(Particle.participant_id)
			.all()
			)

		if not participant_ids:
			self.log("[{}; R{}; G{}] Frozen generation {}; No outstanding participants.".format(self.condition, self.replication, generation, generation, len(participant_ids)), key)
			return 

		for participant_id in participant_ids:
			self.assign_to_overflow(participant_id[0])

		self.log("[{}; R{}; G{}] Frozen generation {}; {} outstanding participants automatically assigned to overflow.".format(self.condition, self.replication, generation, generation, len(participant_ids)), key)

	# @pysnooper.snoop()
	def distribute(self, node, nodes):
		key = "models.py >> distribute: "
		"""Decide whether a participant keeps the provisional slot or is reassigned."""
		if np.any(["OVF" in n.property5 for n in nodes]):
			self.log("Participant {} has already been assigned to the overflow [Gen: {}; Cond: {}; Slot: {} (= Shadow)]. Tagging all nodes.".format(node.participant_id, node.generation, node.condition, node.slot), key)
			self.assign_to_overflow(node.participant_id)
			self.log_slots(generation = int(node.property2), chosen_shadow = int(node.property1), show_pointer = True)
			return

		if self.generation_complete(generation = int(node.generation), ignore_id = node.participant_id):
			self.log("Generation already completed. Participant {} [Gen: {}; Cond: {}; Slot: {} (= Shadow)] assigned to overflow.".format(node.participant_id, node.generation, node.condition, node.slot), key)
			self.assign_to_overflow(node.participant_id)
			self.log_slots(generation = int(node.property2), chosen_shadow = int(node.property1), show_pointer = True)
			return

		assigned_slot = node.slot
		slot_occupied = assigned_slot in self.full_slots(int(node.generation), ignore_id = node.participant_id)
		if slot_occupied:
			self.log("Participant {} [Gen: {}; Cond: {}; Slot: {} (= Shadow)] is in a slot that is already occupied. Attempting reassignment.".format(node.participant_id, node.generation, node.condition, node.slot), key)
			new_slot = self.reassign_slot(current_slot = assigned_slot, generation = int(node.generation))
			
			if new_slot is not None:
				for participant_node in nodes:
					participant_node.slot = new_slot
				self.log("Reassignment succesful. Participant {} [Gen: {}; Cond: {}; Slot: {}] has been given a new slot ({}) and assigned status: Experimental. All nodes updated.".format(node.participant_id, node.generation, node.condition, node.slot, new_slot), key)
				self.log_slots(generation = int(node.property2), chosen_slot = int(new_slot), show_pointer = True)
			
			else:
				self.log("Reassignment failed. Participant {} [Gen: {}; Cond: {}; Slot: {} (= Shadow)] has been assigned status: Overflow. All nodes updated.".format(node.participant_id, node.generation, node.condition, node.slot), key)
				self.assign_to_overflow(node.participant_id)
				self.log_slots(generation = int(node.property2), chosen_shadow = int(node.property1), show_pointer = True)

		else:
			self.log("Participant {} [Gen: {}; Cond: {}; Slot: {} (= availible)] has been assigned status: Experimental".format(node.participant_id, node.generation, node.condition, node.slot), key)
			self.log_slots(generation = int(node.property2), chosen_slot = int(node.property1), show_pointer = True, ignore_shadow_id = node.participant_id)

	def generation_complete(self, generation = None, ignore_id = None):
		"""Is the generation complete -- are all slots full?"""
		if generation is None:
			generation = self.current_generation
		if ignore_id is not None:
			return np.all([slot in self.full_slots(generation, ignore_id = ignore_id) for slot in range(int(self.generation_size))])
		return np.all([slot in self.full_slots(generation) for slot in range(int(self.generation_size))])

	def overflow_uptake_this_generation(self, generation = None):
		"""How many participants started working or have completed working this generation?"""
		if generation is None:
			generation = self.current_generation
		uptake = (Particle.query.filter_by(failed = False, network_id = self.id)
			.filter(Particle.property2 == repr(generation))
			.count() - self.generation_size)
		return max([0, uptake])

	def create_node(self, participant, slot = None):
		if slot is None:
			slot = self.assign_slot(participant)
		return Particle(network=self, participant=participant, slot = slot)

	def overflow_nodes_approved_this_generation(self, generation):
		return Particle.query.filter_by(network_id = self.id, failed = False).filter(and_(Particle.property2 == repr(int(generation)), Particle.property5.contains("OVF"))).count()

	def _select_oldest_source(self):
		return min(self.nodes(type=Environment), key=attrgetter("creation_time"))

	def _sample_parent(self, generation):
		"""randomly sample a non-overflow node from generation"""
		# property5 is condition; property5 is genration;
		return random.choice(Particle.query.filter_by(failed = False).filter(and_(Particle.property2 == repr(int(generation)), not_(Particle.property5.contains("OVF")), Particle.network_id == self.id)).all())

	# @pysnooper.snoop()
	def add_node(self, node, generation = None):

		node.generation = int(self.current_generation) if generation is None else int(generation)

		if int(node.generation) == 0:
			parent = self._select_oldest_source()
		
		else:
			parent = self._sample_parent(int(node.generation) - 1)
		
		if parent is not None:
			parent.connect(whom=node)
			parent.transmit(to_whom=node)

		node.receive()

	# @pysnooper.snoop()
	def calculate_full(self):
		"""Set whether the Particle Filter is *full*."""
		# self.full = len(self.nodes()) >= (self.max_size or 0)
		approved_particles = (
			db.session.query(Particle).join(Participant)
			.filter(Particle.network_id == self.id)
			.filter(Participant.status == "approved")
			.filter(Particle.failed == False)
			.filter(not_(Particle.property5.contains("OVF")))
			.count()
		)
		threshold = (int(self.max_size) - 2)
		self.full = int(approved_particles) >= threshold

class Particle(Node):
	"""The Rogers Agent."""

	__mapper_args__ = {"polymorphic_identity": "particle"}

	@hybrid_property
	def slot(self):
		"""Convert property1 to genertion."""
		return int(self.property1)

	@slot.setter
	def slot(self, slot):
		"""Make slot settable."""
		self.property1 = repr(slot)

	@slot.expression
	def slot(self):
		"""Make slot queryable."""
		return cast(self.property1, Integer)

	@hybrid_property
	def generation(self):
		"""Convert property2 to genertion."""
		return int(self.property2)

	@generation.setter
	def generation(self, generation):
		"""Make generation settable."""
		self.property2 = repr(generation)

	@generation.expression
	def generation(self):
		"""Make generation queryable."""
		return cast(self.property2, Integer)

	@hybrid_property
	def decision_index(self):
		"""Convert property3 to decision_index."""
		return int(self.property3)

	@decision_index.setter
	def decision_index(self, decision_index):
		"""Mark decision_index settable."""
		self.property3 = repr(decision_index)

	@decision_index.expression
	def decision_index(self):
		"""Make decision_index queryable."""
		return cast(self.property3, Integer)

	@hybrid_property
	def proportion(self):
		"""Make property4 proportion."""
		return float(self.property4)

	@proportion.setter
	def proportion(self, proportion):
		"""Make proportion settable."""
		self.property4 = repr(proportion)

	@proportion.expression
	def proportion(self):
		"""Make proportion queryable."""
		return cast(self.property4, Float)

	@hybrid_property
	def condition(self):
		"""Make property5 condition."""
		return self.property5

	@condition.setter
	def condition(self, condition):
		"""Make condition settable."""
		self.property5 = condition

	@condition.expression
	def condition(cls):
		"""Make condition queryable."""
		return cls.property5

	def __init__(self, network, participant=None, contents=None, details=None, slot=None):
		"""Create a node."""
		# check the network hasn't failed
		if network.failed:
			raise ValueError(
				"Cannot create node in {} as it has failed".format(network)
			)
		# check the participant hasn't failed
		if participant is not None and participant.failed:
			raise ValueError(
				"{} cannot create a node as it has failed".format(participant)
			)
		# # check the participant is working
		# if participant is not None and participant.status != "working":
		# 	raise ValueError(
		# 		"{} cannot create a node as they are not working".format(participant)
		# 	)

		self.network = network
		self.network_id = network.id
		network.calculate_full()

		if participant is not None:
			self.participant = participant
			self.participant_id = participant.id

		self.condition = self.network.property5
		self.slot = slot


	# def __init__(self, contents=None, details = None, network = None, participant = None, slot = None):
	# 	super(Particle, self).__init__(network, participant)
	# 	self.condition = self.network.property5
	# 	self.slot = slot
	# 	# self.proportion = self.network.proportion


class NetworkRandomAttributes(Node):
	"""The participant."""

	@declared_attr
	def __mapper_args__(cls):
		"""The name of the source is derived from its class name."""
		return {
			"polymorphic_identity": cls.__name__.lower()
		}
   
	@hybrid_property
	def current_generation(self):
		"""Convert property2 to genertion."""
		return int(self.property2)

	@current_generation.setter
	def current_generation(self, current_generation):
		"""Make current_generation settable."""
		self.property2 = repr(current_generation)

	@current_generation.expression
	def current_generation(self):
		"""Make current_generation queryable."""
		return cast(self.property2, Integer)

	@hybrid_property
	def overflow_pool(self):
		"""Convert property3 to genertion."""
		return int(self.property3)

	@overflow_pool.setter
	def overflow_pool(self, overflow_pool):
		"""Make overflow_pool settable."""
		self.property3 = repr(overflow_pool)

	@overflow_pool.expression
	def overflow_pool(self):
		"""Make overflow_pool queryable."""
		return cast(self.property3, Integer)

	# #@pysnooper.snoop()
	def sample_parents(self):
		N = self.generations
		n = self.network.generation_size
		parents = {}
		for g in range(1, N):
			parents[g] = dict(zip(list(range(n)), random.choices(list(range(n)), k = n)))
		return parents

	def sample_payout_colors(self):
		n = float(self.network.generation_size)
		return dict(zip(range(int(n)), (["green"] * int(n / 2.)) + (["blue"] * int(n / 2.))))

	def sample_button_order(self):
		n = float(self.network.generation_size)
		# uncomment below for real exp; commented so no need to run four per gen
		return dict(zip(range(int(n)), (["left"] * int(n / 4.)) + (["right"] * int(n / 4.)) + (["left"] * int(n / 4.)) + (["right"] * int(n / 4.))))
		# return dict(zip(range(int(n)), (["left"] * int(n / 2.)) + (["right"] * int(n / 2.))))

	def __init__(self, network, generations, overflow_pool, details = None):
		super(NetworkRandomAttributes, self).__init__(network)
		self.generations = generations
		self.details = json.dumps({"parentschedule": self.sample_parents(), "payout_color": self.sample_payout_colors(), "button_order": self.sample_button_order()})
		self.current_generation = 0
		self.overflow_pool = overflow_pool

class Referee(Node):
	"""The experiment's referee.
	The referee can pause and resume the experiment.
	"""

	@declared_attr
	def __mapper_args__(cls):
		"""The name of the source is derived from its class name."""
		return {
			"polymorphic_identity": cls.__name__.lower()
		}
   
	@hybrid_property
	def paused(self):
		"""Convert property1 to genertion."""
		return self.property1

	@paused.setter
	def paused(self, paused):
		"""Make paused settable."""
		self.property1 = repr(paused)

	@paused.expression
	def paused(self):
		"""Make current_generation queryable."""
		return self.property1

	def __init__(self, network, details = None, initial_recruitment_size = 0):
		super(Referee, self).__init__(network)
		self.paused = "live"


class RecruitmentRequestRecord(Info):
	"""An Info that represents a parametrisable technology with a utility function."""

	@declared_attr
	def __mapper_args__(cls):
		"""The name of the source is derived from its class name."""
		return {
			"polymorphic_identity": cls.__name__.lower()
		}

	@hybrid_property
	def generation(self):
		"""Convert property2 to genertion."""
		return int(self.property2)

	@generation.setter
	def generation(self, generation):
		"""Make generation settable."""
		self.property2 = repr(generation)

	@generation.expression
	def generation(self):
		"""Make generation queryable."""
		return cast(self.property2, Integer)

	def __init__(self, origin, generation, contents=None, details = None):
		self.origin = origin
		self.origin_id = origin.id
		self.network_id = origin.network_id
		self.network = origin.network
		self.generation = generation

class TrialBonus(Info):
	"""An Info that represents a parametrisable technology with a utility function."""

	@declared_attr
	def __mapper_args__(cls):
		"""The name of the source is derived from its class name."""
		return {
			"polymorphic_identity": cls.__name__.lower()
		}

	@hybrid_property
	def bonus(self):
		"""Use property1 to store the bonus."""
		try:
			return float(self.property1)
		except TypeError:
			return None

	@bonus.setter
	def bonus(self, bonus):
		"""Assign bonus to property1."""
		self.property1 = float(bonus)

	@bonus.expression
	def bonus(self):
		"""Retrieve bonus via property1."""
		return cast(self.property1, float)

	def parse_data(self, contents):
		self.bonus = json.loads(contents)["current_bonus"]

	def __init__(self, origin, contents=None, details = None, initialparametrisation = None):
		self.origin = origin
		self.origin_id = origin.id
		self.network_id = origin.network_id
		self.network = origin.network
		self.parse_data(contents)

class ComprehensionTest(Info):
	"""An Info that represents a comprehension test."""

	@declared_attr
	def __mapper_args__(cls):
		"""The name of the source is derived from its class name."""
		return {
			"polymorphic_identity": cls.__name__.lower()
		}

	@hybrid_property
	def passed(self):
		"""Use property1 to store the technology's parameters and their ranges as a json string."""
		try:
			return bool(self.property1)
		except TypeError:
			return None

	@passed.setter
	def passed(self, p):
		"""Assign passed to property1."""
		self.property1 = p

	@passed.expression
	def passed(self):
		"""Retrieve passed via property1."""
		return cast(self.property1, bool)

	def evaluate_answers(self):
		return all([q == "1" for q in self.questions.values()])

	def __init__(self, origin, contents=None, details = None, initialparametrisation = None):
		self.origin = origin
		self.origin_id = origin.id
		self.network_id = origin.network_id
		self.network = origin.network
		self.questions = json.loads(contents)
		self.passed = self.evaluate_answers()
		self.contents = contents

class GenerativeModel(Environment):
	"""The Data-generating Environment."""
	
	@declared_attr
	def __mapper_args__(cls):
		"""The name of the source is derived from its class name."""
		return {
			"polymorphic_identity": cls.__name__.lower()
		}

	@hybrid_property
	def proportion(self):
		"""Make property5 proportion."""
		return float(self.property5)

	@proportion.setter
	def proportion(self, proportion):
		"""Make proportion settable."""
		self.property5 = repr(proportion)

	@proportion.expression
	def proportion(self):
		"""Make proportion queryable."""
		return cast(self.property5, Float)

	def create_state(self, proportion):
		"""Create an environmental state."""
		self.proportion = proportion
		State(origin=self, contents=proportion)



class BiasReport(Info):

	@declared_attr
	def __mapper_args__(cls):
		"""The name of the source is derived from its class name."""
		return {
			"polymorphic_identity": cls.__name__.lower()
		}




