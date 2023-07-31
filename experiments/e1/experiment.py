from dallinger.experiment import Experiment
from dallinger.information import Meme
from dallinger.models import Network, Node, Participant
from dallinger.config import get_config
from sqlalchemy import and_, not_, func
from sqlalchemy.sql.expression import cast
from sqlalchemy import Integer
from dallinger import db
from flask import Blueprint, Response
from copy import deepcopy

import collections
from operator import itemgetter
import numpy as np 
import random
import json
import pysnooper
from operator import attrgetter
from collections import Counter

import logging
logger = logging.getLogger(__file__)

DEBUG = False

class UWPFWP(Experiment):
	"""Utility Weighted Particle Filter with People.
	 
	 TODO: 
	 - assign recruitment functionality to a background task: https://gist.github.com/czardoz/2e5bf13a233ae28fda9f
	 - add all four conditions
	 - revert button orderings
	 - revert debug
	 - rvert proportions
	 - gen size
	 - gens
	 - conditions

	 - 
	"""

	@property
	def public_properties(self):
		return {
		'generation_size':8, 
		'generations': 4,
		'planned_overflow':50,
		'num_replications_per_condition':10,
		'num_fixed_order_experimental_networks_per_experiment': 0,
		'num_random_order_experimental_networks_per_experiment': 1 if DEBUG else 8,
		'num_practice_networks_per_experiment': 1 if DEBUG else 2,
		'cover_story': 'true',
		'payout_blue':'true',
		'bonus_max': 1,
		"yoke": False
		}

	def __init__(self, session=None):
		super(UWPFWP, self).__init__(session)
		import models
		self.models = models
		self.set_known_classes()
		
		# These variables are potentially needed on every invocation 
		self.set_params()

		# setup is only needed when launching the experiment 
		if session and not self.networks():
			self.setup()
		self.save()

	def set_known_classes(self):
		self.known_classes["trialbonus"] = self.models.TrialBonus
		self.known_classes["biasReport"] = self.models.BiasReport
		self.known_classes["particle"] = self.models.Particle
		self.known_classes['comprehensiontest'] = self.models.ComprehensionTest
		self.known_classes['generativemodel'] = self.models.GenerativeModel
		self.known_classes["particlefilter"] = self.models.ParticleFilter
		self.known_classes["networkrandomattributes"] = self.models.NetworkRandomAttributes
		self.known_classes["referee"] = self.models.Referee

	def set_params(self):
		"""
		Notes:
		- A condition is a manipulation
		- An experiment is a single replication of a condition
		- Information does not flow between experiments
		- A network is a single "Trial", with a constant proportion but generation-verying stimulus realisations of that proportion
		- Every experiment includes some practice networks and some experimental networks (fixed order and random order)

		Conditions:
		- SOC:N-U
		- SOC:W-U
		- ASO:N-U
		- ASO:W-U
		- SWI:N-U this one doesn't exist
		- SWU:W-U original utility information
		- SWB:W-U bias index
		- SWT:W-U truth index

		"""

		# Public Parameters
		for key, value in self.public_properties.items():
			setattr(self, key, value)

		# Internal parameters
		self.practice_network_proportions = [.53, .47] if not DEBUG else [.52]
		self.fixed_order_experimental_network_proportions = []
		self.random_order_experimental_network_proportions = [.48, .52, .51, .49,.48, .52, .51, .49] if not DEBUG else [.52]
		self.condition_counts = {
			"ASO:W-U":self.num_replications_per_condition, 
			"ASO:N-U":self.num_replications_per_condition, 
			"SOC:W-U":self.num_replications_per_condition,
			"SOC:N-U":self.num_replications_per_condition
		} if not DEBUG else {
			#"ASO:W-U":self.num_replications_per_condition, 
			"SOC:W-U":self.num_replications_per_condition
		}

		# Derrived Quantities
		self.num_experiments = sum(self.condition_counts.values())
		self.num_experimental_participants_per_generation = self.generation_size * self.num_experiments
		self.num_experimental_networks_per_experiment = self.num_fixed_order_experimental_networks_per_experiment + self.num_random_order_experimental_networks_per_experiment
		self.num_networks_per_participant = self.num_practice_networks_per_experiment + self.num_experimental_networks_per_experiment
		self.num_experimental_networks_total = (self.num_practice_networks_per_experiment + self.num_experimental_networks_per_experiment) * self.num_experiments
		self.initial_recruitment_size = (self.generation_size * self.num_experiments) + self.planned_overflow
		self.num_experimental_participants_per_generation = self.num_experiments * self.generation_size
		self.num_experimental_nodes_per_generation = self.num_experimental_networks_total * self.generation_size

	def create_network(self, condition, replication, role, decision_index, proportion):
		net = self.models.ParticleFilter(
				generations=self.generations,
				generation_size=self.generation_size,
				replication=replication,
				condition=condition,
				decision_index=decision_index,
				role=role
			)
		self.session.add(net)
		
		# create the Genertative process for stimuli
		datasource = self.models.GenerativeModel(network=net)
		datasource.create_state(proportion=proportion)
		return net
		
	def setup(self):
		"""First time setup."""
		for (condition, replications) in self.condition_counts.items():
			for replication in range(replications):
				for p in range(self.num_practice_networks_per_experiment):
					network = self.create_network(condition = condition, replication = replication, role = 'practice', decision_index = p, proportion = self.practice_network_proportions[p])
					self.models.NetworkRandomAttributes(network = network, generations=self.generations, overflow_pool = 0)
					
				for f in range(self.num_fixed_order_experimental_networks_per_experiment):
					decision_index = self.num_practice_networks_per_experiment + f
					network = self.create_network(condition = condition, replication = replication, role = 'experiment', decision_index = decision_index, proportion = self.fixed_order_experimental_network_proportions[f])
					self.models.NetworkRandomAttributes(network = network, generations=self.generations, overflow_pool = 0)

				for r in range(self.num_random_order_experimental_networks_per_experiment):
					decision_index = self.num_practice_networks_per_experiment + self.num_fixed_order_experimental_networks_per_experiment + r
					network = self.create_network(condition = condition, replication = replication, role = 'experiment', decision_index = decision_index, proportion = self.random_order_experimental_network_proportions[r])
					self.models.NetworkRandomAttributes(network = network, generations=self.generations, overflow_pool = 0)

		# add the referee
		self.models.Referee(network = network, initial_recruitment_size = self.initial_recruitment_size)
		self.session.commit()

	# #@pysnooper.snoop()
	def get_network_for_existing_participant(self, participant, participant_nodes):
		"""Obtain a netwokr for a participant who has already been assigned to a condition by completeing earlier rounds"""
		
		# What condition is this participant in?
		participant_condition = participant_nodes[0].property5 # node.property5 = condition

		if ":OVF" in participant_condition: participant_condition = participant_condition.replace(":OVF", "")

		# which networks has this participant already completed?
		networks_participated_in = [node.network_id for node in participant_nodes]

		# What replciation is this participant in?
		participant_replication = self.models.Network.query.get(networks_participated_in[0]).property3 # network.property3 = replication
		
		# How many decisions has the particiapnt already made?
		completed_decisions = len(networks_participated_in)

		# When the participant has completed all networks in their condition, their experiment is over
		# returning None throws an error to the fronted which directs to questionnaire and completion
		if completed_decisions == self.num_practice_networks_per_experiment + self.num_experimental_networks_per_experiment:
			return None

		nfixed = self.num_practice_networks_per_experiment + self.num_fixed_order_experimental_networks_per_experiment

		# If the participant must still follow the fixed network order
		if completed_decisions < nfixed:
			# find the network that is next in the participant's schedule
			# match on completed decsions b/c decision_index counts from zero but completed_decisions count from one
			return self.models.Network.query.filter(and_(self.models.Network.property4 == repr(completed_decisions), self.models.Network.property5 == participant_condition, self.models.Network.property3 == participant_replication)).one()

		# If it is time to sample a network at random
		else:
			# find networks which match the participant's condition and werent' fixed order nets
			matched_condition_experimental_networks = self.models.Network.query.filter(and_(cast(self.models.Network.property4, Integer) >= nfixed, self.models.Network.property5 == participant_condition, self.models.Network.property3 == participant_replication)).all()
			
			# subset further to networks not already participated in (because here decision index doesnt guide use)
			availible_options = [net for net in matched_condition_experimental_networks if net.id not in networks_participated_in]
			
			# choose randomly among this set
			chosen_network = random.choice(availible_options)

		return chosen_network

	# @pysnooper.snoop()
	def triage(self, candidate_networks):
		key = "experiment.py >> triage: "
		# Which networks have slots that have not even been started?
		availible_networks = [net for (net, status) in candidate_networks.items() if status == "recruiting"]

		if not availible_networks:
			# self.log("All experimental networks have assigned or filled all of their slots.", key)
			# these networks have assigned all their slots but not yet filled them up
			availible_shadow_networks = [net for (net, status) in candidate_networks.items() if status == "accepting"]

			# This should only happen at the end of the experiment
			# If there are no accepting networks, then we should have rolled a new generation
			if not availible_shadow_networks:
				
				# If there are no candidate networks
				# All must be *Full*
				# The requesting node will be assigned to overflow
				if len(candidate_networks) == 0:
					# self.log("All networks are completely full. The experiment must be over. Returning a random network.", key)
					return random.choice(self.models.ParticleFilter.query.filter_by(failed = False).filter(self.models.ParticleFilter.property4 == repr(0)).all())
				
				# We should never be in the situation where there are only "status = full" nets in candidate_net
				# self.log("All networks are full for this generation. An overflow node must be requesting an old network. This shouldn't happen. Returning a random seed net.", key)	
				return random.choice(list(candidate_networks.keys()))

			# self.log("These networks still have unfilled slots: {}. Selecting one at random.".format([net.id for net in availible_shadow_networks]), key)
			return random.choice(availible_shadow_networks)

		# self.log("The availible networks are: {}".format([net.id for net in availible_networks]), key)
		return random.choice(availible_networks)
	
	# @pysnooper.snoop()
	def get_network_for_new_participant(self, participant):
		key = "experiment.py >> get_network_for_new_participant ({}); ".format(participant.id)
		current_generation = self.current_generation()
		seed_networks = (
			self.models.ParticleFilter.query
			.filter_by(full = False, failed = False)
			.filter(self.models.ParticleFilter.property4 == repr(0))
		)
		
		candidate_networks = dict([(net, net.availability()) for net in seed_networks.all()])
		return self.triage(candidate_networks)

	#@pysnooper.snoop()
	def get_network_for_participant(self, participant):
		"""Find a network for a participant."""
		key = "experiment.py >> get_network_for_participant ({}); ".format(participant.id)
		participant_nodes = participant.nodes()
		if not participant_nodes:
			decision_index = 0
			chosen_network = self.get_network_for_new_participant(participant)
		else:
			decision_index = len(participant_nodes) + 1
			chosen_network = self.get_network_for_existing_participant(participant, participant_nodes)

		if chosen_network is not None:
			self.log("Assigned to network: {}; Condition: {}; Replication: {}; Generation: {}; PDI: {}".format(chosen_network.id, chosen_network.condition, chosen_network.replication, chosen_network.current_generation, decision_index), key)

		else:
			self.log("Requested a network but was assigned None.".format(len(participant_nodes)), key)

		return chosen_network

	# #@pysnooper.snoop()
	def create_node(self, network, participant, ignore_duplicates = False):
		"""Make a new node for participants."""
		memes = [i for i in participant.infos() if i.type == "meme"]
		if len(memes) >= (self.num_practice_networks_per_experiment + self.num_experimental_networks_per_experiment) and not ignore_duplicates:
			raise Exception

		nodes = participant.nodes()
		slot = nodes[0].slot if nodes else None
		return network.create_node(participant, slot)

	# @pysnooper.snoop()
	def add_node_to_network(self, node, network):
		"""Add participant's node to a network."""
		if isinstance(node, self.models.Particle):
			participant_nodes = list(filter(lambda n: n.id != node.id, self.models.Particle.query.filter_by(participant_id = node.participant_id, failed = False).all()))
			if participant_nodes:
				network.add_node(node, generation = participant_nodes[0].generation)
			else:
				network.add_node(node)

			node.proportion = float(self.models.GenerativeModel.query.filter_by(network_id = network.id).one().property5) # property5 = proportion
			node.decision_index = self.models.Particle.query.filter_by(participant_id=node.participant_id, failed = False, type = 'particle').count()
		
		else:
			network.add_node(node)

	def update_overflow_pool(self, this_generations_uptake):
		key = "experiment.py >> update_overflow_pool: "
		for network_random_atttributes in self.models.NetworkRandomAttributes.query.all():
			network_random_atttributes.overflow_pool = int(network_random_atttributes.overflow_pool) + this_generations_uptake
		self.log("Updated all overflow uptake statistics in NetworkRandomAttributes cache.", key)

	# @pysnooper.snoop(prefix = "@snoopsten: ")
	def overflow_uptake_count_this_generation(self, generation = None):
		"""How many participants started working or have completed working this generation in each overflow replication?"""
		return sum([net.overflow_uptake_this_generation(generation = generation) for net in self.models.ParticleFilter.query.filter(self.models.ParticleFilter.property4 == repr(0)).all()])

	def overflow_uptake_count_total(self):
		key = "experiment.py >> overflow_uptake_count_total: "
		# count participant ids for all approved overflow participants before the final generation
		num_participant_ids_approved_ovf = (
			db.session.query(self.models.Particle.participant_id).join(self.models.Participant).join(self.models.ParticleFilter)
			.filter(self.models.Participant.status == "approved")
			.filter(self.models.Particle.failed == False)
			.filter(self.models.ParticleFilter.property4 == repr(0))
			.filter(self.models.Particle.property2 != repr(int(self.generations) - 1))
			.filter(self.models.Particle.property5.contains("OVF"))
			.distinct(self.models.Particle.participant_id)
			.count()
			)

		# see self.tag_failed_overflow
		num_participant_ids_ovfprf = (
			db.session.query(self.models.Particle.participant_id).join(self.models.Participant).join(self.models.ParticleFilter)
			.filter(self.models.ParticleFilter.property4 == repr(0))
			.filter(self.models.Particle.property2 != repr(int(self.generations) - 1))
			.filter(self.models.Particle.property5.contains("OVFPRF"))
			.distinct(self.models.Particle.participant_id)
			.count()
			)
		self.log("Approved OVF participants: {}; OVFPRF participants: {}".format(num_participant_ids_approved_ovf, num_participant_ids_ovfprf), key)
		return int(num_participant_ids_approved_ovf) + int(num_participant_ids_ovfprf) + self.planned_overflow

	def approved_participants(self, generation = None, overflow = False, ignore_final_gen = True):
		"""How many experimental participants have been aproved so far?"""
		participant_ids = (
			db.session.query(self.models.Particle.participant_id).join(self.models.Participant).join(self.models.ParticleFilter)
			.filter(self.models.Participant.status == "approved")
			.filter(self.models.Particle.failed == False)
			.distinct(self.models.Particle.participant_id)
			)

		participant_ids = (
			participant_ids.filter(not_(self.models.Particle.property5.contains("OVF"))) if not overflow else
			participant_ids.filter(self.models.Particle.property5.contains("OVF"))
			)

		if ignore_final_gen:
			participant_ids = participant_ids.filter(not_(self.models.Particle.property5.contains(":FGSX")))

		if generation is not None:
			participant_ids = participant_ids.filter(self.models.Particle.property2 == repr(generation))
		return participant_ids.count()
		

	# @pysnooper.snoop(prefix = "@snoop: ")
	def calculate_required_overrecruitment(self, generation = None):
		return min([self.planned_overflow, self.overflow_uptake_count_this_generation(generation = generation)])

	def current_generation(self):
		return self.models.ParticleFilter.query.filter_by(failed=False).first().current_generation

	def generation_complete(self, generation, approved_participants):
		if int(generation) == 0:
			return int(approved_participants) == int(self.initial_recruitment_size - self.planned_overflow)
		return int(approved_participants) == self.num_experimental_participants_per_generation

	def finish_generation(self, generation):
		# have the seed networks freeze their generation (& log their slot structure)
		for net in self.models.ParticleFilter.query.filter(self.models.ParticleFilter.property4 == repr(0)).all():
			net.freeze_generation(generation)

	def pin_generation(self, generation):
		key ="experiment.py >> pin_generation: "
		for net in self.networks():
			net.current_generation = int(generation)
			nra = self.models.NetworkRandomAttributes.query.filter_by(network_id = net.id).one()
			nra.current_generation = int(generation)
		self.log("Manually pinned generation {}: all networks now at generation: {}".format(generation, generation), key)

	def rollover_generation(self, generation = None):
		key ="experiment.py >> rollover_generation: "
		if generation is None:
			generation = self.current_generation()
		for net in self.networks():
			net.current_generation = int(net.current_generation) + 1
			nra = self.models.NetworkRandomAttributes.query.filter_by(network_id = net.id).one()
			nra.current_generation = int(nra.current_generation) + 1
		self.log("Rolled new generation: all networks now at generation: {}".format(int(net.current_generation)), key)

	# @pysnooper.snoop()
	def clone_participant(self, participant_id, targetcondition):
		key = "experiment.py >> clone_participant: "
		participant = self.models.Participant.query.get(participant_id)
		# participant.status = "working"
		participant_nodes = participant.nodes()
		for node in participant_nodes:
			
			# Identify the source network to access replication, decision_index
			sourcenetwork = self.models.ParticleFilter.query.get(node.network_id)
			replication, decision_index = sourcenetwork.replication, sourcenetwork.decision_index

			# get the target network
			targetnetwork = (
				self.models.ParticleFilter.query
				.filter(self.models.ParticleFilter.property3 == repr(replication))
				.filter(self.models.ParticleFilter.property4 == repr(decision_index))
				.filter(self.models.ParticleFilter.property5.contains(targetcondition))
			).one()
			
			# create the node copy in the target network
			clone_node = self.create_node(network = targetnetwork, participant = participant, ignore_duplicates = True)
			self.add_node_to_network(clone_node, targetnetwork)
			
			# gotta overwrite decison_index b/c add_node_to_network gets that from participants previous nodes
			clone_node.decision_index = node.decision_index

			for info in node.infos(type = Meme):
				clone_info = Meme(origin = clone_node, contents = info.contents)
		# participant.status = "approved"
		self.log("Cloned {} nodes from participant {} [C: {}; R: {}] --> {}".format(len(participant_nodes), participant_id, sourcenetwork.condition, sourcenetwork.replication, targetcondition), key)

	def permission_to_recruit(self, generation_to_recruit):
		return self.models.RecruitmentRequestRecord.query.filter(self.models.RecruitmentRequestRecord.property2 == repr(generation_to_recruit)).count() == 1

	def recruit(self):
		"""Recruit participants"""
		key = "experiment.py >> recruit: "

		# estbalish current generation
		current_generation = deepcopy(self.current_generation())

		# how many of this geenration's nodes have been approved?
		num_experimental_participants_approved_this_generation = self.approved_participants(generation = current_generation) 

		# Is this generation complete?
		end_of_generation = self.generation_complete(generation = current_generation, approved_participants = int(num_experimental_participants_approved_this_generation))
		
		participants_required_this_generation = self.num_experimental_participants_per_generation if int(current_generation) > 0 else (self.initial_recruitment_size - self.planned_overflow)
		self.log("Generation in Progress: {}; Experimental participants approved: {}; Required: {}".format(current_generation, num_experimental_participants_approved_this_generation, participants_required_this_generation), key)
		self.log("End of generation: {}".format(end_of_generation), key)

		# Are all experimental generations complete?
		experimental_networks_complete = (current_generation == self.generations - 1) & (end_of_generation)

		# Have we finished recruiting experimental paricipants?
		if experimental_networks_complete:
			# How many overflow nodes are required according to the recruitments that have been issued?
			total_overflow_participants_required = self.overflow_uptake_count_total()
			
			num_approved_overflow_participants = self.approved_participants(overflow = True, generation = current_generation, ignore_final_gen = False)
			self.log("Total overflow participants required: {}; Total overflow participants approved: {}".format(total_overflow_participants_required, num_approved_overflow_participants), key)

			if num_approved_overflow_participants >= total_overflow_participants_required:
				self.log("All experimental networks are full. Overflow is full. Experiment complete: closing recruitment", key)
				self.recruiter.close_recruitment()

			else:
				self.log("All experimental networks are full. Overflow recruits remain live. Waiting on {} more overflow participants.".format(total_overflow_participants_required - num_approved_overflow_participants), key)
				return

		# Or are more generations required? 
		elif end_of_generation:

			# persistent records of experimental varibles
			referee = self.models.Referee.query.one()
			self.models.RecruitmentRequestRecord(origin = referee, generation = int(current_generation) + 1)

			next_generation_required_overflow = self.calculate_required_overrecruitment(generation = current_generation)
			self.log("Required over-recruiment at the next generation is: {}.".format(next_generation_required_overflow), key)

			# If we got here, it's time to roll out a new generation
			next_generation_recruitment = (self.generation_size * (self.num_experiments)) + next_generation_required_overflow
			self.log("Generation complete. Please manually recruit {} participants.".format(next_generation_recruitment), key)
			self.log("Hit /manualcompletegeneration/{}/".format(next_generation_recruitment), key)

			# check that nobody is in the middle of triggering this same recruitment
			if self.permission_to_recruit(generation_to_recruit = int(current_generation) + 1):
				self.log("This is the first time that permission to recruit has been granted.", key)

			# 	# check that the experiment is not paused
			# 	if not referee.paused == repr("live"):
			# 		self.log("Experiment is paused. Blocking recruitment. Rollover executed as normal.", key)
			# 		return 
				
			# 	# record that we;re going to recruit
			# 	# self.models.RecruitmentRequestRecord(origin = referee, generation = int(current_generation) + 1)
			# 	self.log("Permission to recruit generation {} granted.".format(int(current_generation) + 1), key)

			# 	# change persistent state immediately
			# 	self.rollover_generation()
			
			# 	# notify all networks to triage any remaining ps into overflow
			# 	self.finish_generation(current_generation)

			# 	self.recruiter.recruit(n = next_generation_recruitment)
			# 	self.log("Made {} new recruitments. Updated RecruitmentRequestRecords.", key)
			# 	return

			# self.log("Permission to recruit generation {} denied.".format(int(current_generation) + 1), key)
	
	def submission_successful(self, participant):
		key = "experiment.py >> submission_successful: "
		nodes = participant.nodes(type=self.models.Particle)
		if not nodes:
			self.log("Participant {} submitted sucessfully but created no nodes.".format(participant.id), key)
			return
		final_node = max(nodes, key=attrgetter("creation_time"))
		final_network = self.models.ParticleFilter.query.get(final_node.network_id)
		final_network.distribute(final_node, nodes)

		# Gn
		# current_generation = self.current_generation()
		current_generation = participant.nodes()[0].generation
		if int(current_generation) == self.generations - 1:
			if int(self.approved_participants(generation = current_generation)) == int(self.num_experimental_participants_per_generation):
				if "OVF" in participant.nodes()[0].property5:
					self.tag_final_gen_overflows(participant.nodes())



	#@pysnooper.snoop()
	def bonus(self, participant):
		"""Calculate a participants bonus."""
		infos = participant.infos()
		
		# self.log("{}".format(infos), "--**bonus infos-->>")

		totalbonus = 0

		for info in infos:
			if info.type == "meme":
				contents = json.loads(info.contents)
				if contents["is_practice"] == False:
					totalbonus += (contents["current_bonus"] / 1000.)

		totalbonus = round(totalbonus, 2)

		if totalbonus > self.bonus_max:
			totalbonus = self.bonus_max
		return totalbonus
	
	def tag_final_gen_overflows(self, participant_nodes):
		"""
		"""
		for node in participant_nodes:
			node.property5 = node.property5 + ":FGSX"

	def tag_failed_overflow(self, participant_nodes):
		"""
		EDGE CASE:
			- an overfolow node fials after the generation has rolled over
			- a recruitment is made to replace the overflow node
			- but calculate_required_overrecruitment has alreeady replaced to overflow slot
			- so: some nodes are gonna be failed, without an overflow tag
			- this func tags these nodes with OVF "Post Rollover Failure" 
			- so that they can be counted
		"""
		for node in participant_nodes:
			node.property5 = node.property5 + "OVFPRF"
   
	def fail_participant(self, participant):
		"""Fail all the nodes of a participant."""
		self.log("Participant {} failed".format(participant.id), "experiment.py >> fail_participant: ")
		participant_nodes = Node.query.filter_by(
			participant_id=participant.id, failed=False
		).all()

		if participant_nodes:
			for node in participant_nodes:
				node.fail()
			if int(participant_nodes[0].property2) != int(self.current_generation()):
				self.tag_failed_overflow(participant_nodes)

	def attention_check(self, participant=None):
		"""Check a participant paid attention."""
		key = "experiment.py >> data_check: "
		infos = participant.infos()

		if not infos:
			return False
		
		passed =  np.any([info.passed for info in infos if info.type == 'comprehensiontest'])

		if not passed:
			self.log("Participant {} failed".format(participant.id), key)

		return passed

	# @pysnooper.snoop()
	def data_check(self, participant):
		"""Check a participants data."""
		key = "experiment.py >> data_check: "
		nodes = Node.query.filter_by(participant_id=participant.id).all()

		if not nodes:
			return False

		if len(nodes) != self.num_practice_networks_per_experiment + self.num_experimental_networks_per_experiment:
			self.log("Error: self.models.Participant has {} nodes. Data check failed"
				  .format(len(nodes)), key)
			return False

		nets = [n.network_id for n in nodes]
		if len(nets) != len(set(nets)):
			self.log("Error: self.models.Participant participated in the same network \
				   multiple times. Data check failed", key)
			return False

		return True

	def is_complete(self):
		required_participants = (self.generations * (self.generation_size - 1) * sum(self.condition_counts.values())) + max([self.planned_overflow, int(self.models.NetworkRandomAttributes.query.first().overflow_pool)]) + self.initial_recruitment_size
		completed_participants = self.models.Participant.query.filter_by(status="approved", failed = False).count()
		return completed_participants >= required_participants

	def custom_summary(self):
		key = "experiment.py >> custom_summary: "
		status_counts = (
			db.session.query(
			func.count(self.models.Particle.participant_id.distinct()).label('count'), self.models.ParticleFilter.property5, self.models.ParticleFilter.property3, self.models.Particle.property5, self.models.Particle.property2, self.models.Participant.status)
			.join(self.models.Participant)
			.join(self.models.ParticleFilter)
			.group_by(self.models.ParticleFilter.property5, self.models.ParticleFilter.property3, self.models.Particle.property5, self.models.Particle.property2, self.models.Participant.status)  
			.filter_by(failed = False)
		)
		fullsummary = {}
		for (status, generation, pcondition, replication, condition, count) in [c[::-1] for c in status_counts]:
			ovf_status = "OVF" if "OVF" in pcondition else "EXP"
			label = "{} | R:{} | G:{} | P:{} | {}".format(condition, replication, generation, ovf_status, status)
			fullsummary[label] = count

		for row in collections.OrderedDict(sorted(fullsummary.items())):
			self.log("{}: {}".format(str(row), str(fullsummary[row])), key)

		return fullsummary

	#@pysnooper.snoop()
	def getnet(self, network_id):
		net = self.models.Network.query.filter_by(id = network_id).one()
		return net

	def is_overrecruited(self, waiting_count):
		return False

extra_routes = Blueprint(
	'extra_routes',
	__name__,
	template_folder='templates',
	static_folder='static')

@extra_routes.route("/network/<network_id>/getnet/", methods=["GET"])
def getnet(network_id):
	try:
		exp = UWPFWP(db.session)

		net = exp.getnet(network_id)

		return Response(json.dumps({"network":{"property4":net.__json__()["property4"],"property5": net.__json__()["property5"], "property3": net.__json__()["property3"]}}), status=200, mimetype="application/json")

	except Exception:
		db.logger.exception('Error fetching network info')
		return Response(status=403, mimetype='application/json')

@extra_routes.route("/random_attributes/<int:network_id>/<int:node_generation>/<int:node_slot>", methods=["GET"])
# @pysnooper.snoop()
def get_random_atttributes(network_id, node_generation, node_slot):
	# logger.info("--->>> generation: {}, {}".format(generation, type(generation)))

	exp = UWPFWP(db.session)

	# get the network for this id
	net = exp.getnet(network_id)

	# if we're at generation zero, just get color payout and button order
	if (node_generation == 0):

		# grab the attributes for this netowrk
		network_attributes = exp.models.NetworkRandomAttributes.query.filter_by(network_id = network_id).one()

		# load detils
		data = json.loads(network_attributes.details)

		payout_colors, button_orders = data["payout_color"], data["button_order"]

		# Whcih color is incentivised for this node?
		node_payout = payout_colors[str(node_slot)]

		# Button order randomisation
		node_button_order = button_orders[str(node_slot)]

		# practice + fixed + random
		# [.65, .46, .35, .54] // [.49, .51, .51, .49] // [.48, .52, .51, .49]
		# ks = [12, 5, 0, 7]  + [8, 12, 4, 0] + [12, 0, 4, 8]
		# assert exp.num_random_order_experimental_networks_per_experiment == 8
		# ks = [12, 5, 2, 9] + [2, 5, 7, 10] + [0,4,8,12]

		# n = 12

		# i = int(net.decision_index)

		# return Response(json.dumps({"k":ks[i], "n":12, "b":-1, "button_order":node_button_order, "node_utility":node_payout}), status=200, mimetype="application/json")
		
		return Response(json.dumps({"k":-1, "n":-1, "b":-1, "button_order":node_button_order, "node_utility":node_payout}), status=200, mimetype="application/json")

	# @pysnooper.snoop()
	def f(network_id = None, node_slot = None, node_generation = None):

		# establish whether we're dealing with an overflow node or not
		node_type = exp.models.Particle

		# all approveed participnts
		approved_participants = exp.models.Participant.query.filter_by(status="approved", failed = False).all()

		# all approved participant ids
		approved_participant_ids = [p.id for p in approved_participants]

		previous_generation_nodes = (node_type.query.filter_by(failed = False, network_id = net.id).filter(and_(
			node_type.property2 == repr(int(node_generation) - 1),
			not_(node_type.property5.contains("OVF")),
			node_type.participant_id.in_(approved_participant_ids)))
		.all())

		if len(previous_generation_nodes) > int(exp.generation_size):
			previous_generation_nodes = sorted(previous_generation_nodes[:int(exp.generation_size)], key=lambda x: x.creation_time, reverse=True)

		# grab the attributes for this netowrk
		network_attributes = exp.models.NetworkRandomAttributes.query.filter_by(network_id = network_id).one()

		# load detils
		data = json.loads(network_attributes.details)

		# isolate the three data fields
		parentschedule, payout_colors, button_orders = data["parentschedule"][str(node_generation)], data["payout_color"], data["button_order"]

		# estblish the incentivised colors for this node
		node_payout = payout_colors[str(node_slot)]

		previous_generation_memes = np.array([node.infos(type = Meme)[0] for node in previous_generation_nodes if node.infos(type = Meme)])

		previous_generation_memes_contents = np.array([json.loads(node.infos(type = Meme)[0].contents) for node in previous_generation_nodes if node.infos(type = Meme)])

		previous_generation_choices = np.array([json.loads(node.infos(type = Meme)[0].contents)["choice"] for node in previous_generation_nodes if node.infos(type = Meme)])

		previous_generation_utilities = np.array([payout_colors[str(node.slot)] for node in previous_generation_nodes if node.infos(type = Meme)])
		
		# make a list of whether each parent node chose blue or not
		chose_blue = previous_generation_choices == "blue"

		# count how many parents selected their incentivised color
		# k = (previous_generation_choices == previous_generation_utilities).sum()
		k = sum(np.array([json.loads(node.infos(type = Meme)[0].contents)["choice"] == payout_colors[str(node.slot)] for node in previous_generation_nodes if node.infos(type = Meme)]))

		# count the number who did choose blue
		# this is the nunmbr of current gen participants whose social information was "someone chose blue"
		# b = sum(chose_blue)
		b = sum(np.array([json.loads(node.infos(type = Meme)[0].contents)["choice"] == "blue" for node in previous_generation_nodes if node.infos(type = Meme)]))

		# count the generation size and check it liens up with the exp
		n = exp.generation_size
		assert n == len(chose_blue)

		return Response(json.dumps({"k":int(k), "n":int(n), "b":int(b), "button_order":button_orders[str(node_slot)], "node_utility":node_payout}), status=200, mimetype="application/json")

	try:
		return f(network_id = network_id, node_generation = node_generation, node_slot = node_slot)

	except Exception:
		db.logger.exception('Error fetching network info')
		return Response(status=403, mimetype='application/json')

def fake_node(participant, exp):
	# execute the request
	network = exp.get_network_for_participant(participant=participant)
	node = exp.create_node(participant=participant, network=network)
	exp.add_node_to_network(node=node, network=network)
	return node

# @extra_routes.route("/debugbot/<int:participant_id>/", methods=["POST"])
# def debugbot(participant_id):
# 	try:
# 		import models
# 		exp = UWPFWP(db.session)
# 		participant = models.Participant.query.filter_by(id=participant_id).one()

# 		for practice_trial in range(exp.num_practice_networks_per_experiment):
# 			node = fake_node(participant, exp)
# 			if practice_trial == 0:
# 				comprehensiontest = models.ComprehensionTest(origin = node, contents = json.dumps({"q1":"1","q2":"1","q3":"1"}))
# 			contents = {"is_practice": True, "current_bonus": 50}
# 			Meme(origin = node, contents = json.dumps(contents))

# 		for test_trial in range(exp.num_random_order_experimental_networks_per_experiment):
# 			node = fake_node(participant, exp)
# 			contents = {"is_practice": False, "current_bonus": 50}
# 			Meme(origin = node, contents = json.dumps(contents))
		
# 		db.session.commit()
# 		exp.log("Faked Info structure for participant {}".format(participant_id), "experiment.py >> /debugbot: ")
# 		return Response(json.dumps({"status": "Success!"}), status=200, mimetype="application/json")

# 	except Exception:
# 		db.logger.exception('Error fetching node info')
# 		return Response(status=403, mimetype='application/json')

@extra_routes.route("/pause", methods=["GET"])
def pause():
	"""This prevents recruitment
	
	If you need to use this:
	- Be calm, this works.
	- 1) Press pause
	- 2) Wait for the genration to complte and rollover. 
		 This will always happen eventually, b/c the required HITs have already been posted.
	- 3) See exp log for recruied over recruiment.
	- 4) Click resume/ -- this will simply unblock recruitment.
	- 5) Click the recrutibutton/ with exp.generationsize * nconditions * replications + requiredoverflow.
	"""
	try:
		import models
		exp = UWPFWP(db.session)
		referee =  models.Referee.query.one()
		referee.paused = "paused"
		db.session.commit()
		exp.log("Paused the experiment.", "experiment.py >> /pause: ")
		return Response(json.dumps({"status": "Success!"}), status=200, mimetype="application/json")

	except Exception:
		db.logger.exception('Error pausing')
		return Response(status=403, mimetype='application/json')


@extra_routes.route("/resume", methods=["GET"])
def resume():
	try:
		import models
		exp = UWPFWP(db.session)
		referee =  models.Referee.query.one()
		referee.paused = "live"
		db.session.commit()
		exp.log("Resumed the experiment.", "experiment.py >> /resume: ")
		return Response(json.dumps({"status": "Success!"}), status=200, mimetype="application/json")

	except Exception:
		db.logger.exception('Error pausing')
		return Response(status=403, mimetype='application/json')

@extra_routes.route("/experimentrecruit", methods=["GET"])
def experimentrecruit():
	try:
		import models
		exp = UWPFWP(db.session)
		exp.recruit()
		exp.log("Clicked experiment.recruit().", "experiment.py >> /experimentrecruit: ")
		return Response(json.dumps({"status": "Success!"}), status=200, mimetype="application/json")

	except Exception:
		db.logger.exception('Error pausing')
		return Response(status=403, mimetype='application/json')

@extra_routes.route("/recruitbutton/<int:nparticipants>/", methods=["GET"])
def recruitbutton(nparticipants):
	try:
		import models
		exp = UWPFWP(db.session)

		exp.recruiter.recruit(n = nparticipants)
		# exp.save()

		exp.log("Made {} recruitments.".format(nparticipants), "experimnt.py >> **--Recruitbutton Hit: ")
		
		return Response(json.dumps({"status": "Success!"}), status=200, mimetype="application/json")

	except Exception:
		db.logger.exception('Error fetching node info')
		return Response(status=403, mimetype='application/json')

@extra_routes.route("/customsummary", methods=["GET"])
def customsummary():
	try:
		exp = UWPFWP(db.session)

		summary = exp.custom_summary()
		
		return Response(json.dumps({"status": "Success!", "summary": summary}), status=200, mimetype="application/json")

	except Exception:
		db.logger.exception('Error fetching node info')
		return Response(status=403, mimetype='application/json')

@extra_routes.route("/rolloverbutton/", methods=["GET"])
def rolloverbutton():
	try:
		import models
		exp = UWPFWP(db.session)

		# rollover generation
		exp.rollover_generation()
		exp.save()
	
		# # notify all networks to triage any remaining ps into overflow
		# exp.finish_generation(exp.current_generation())

		exp.log("Pushed rollover and finish.", "experiment.py >> /rolloverbutton: ")
		
		return Response(json.dumps({"status": "Success!"}), status=200, mimetype="application/json")

	except Exception:
		db.logger.exception('Error rolling over')
		return Response(status=403, mimetype='application/json')

@extra_routes.route("/pinbutton/<int:generation>/", methods=["GET"])
def pinbutton(generation):
	try:
		import models
		exp = UWPFWP(db.session)

		# rollover generation
		exp.pin_generation(generation)
		exp.save()
	
		exp.log("Manually pinned generation at: {}.".format(generation), "experiment.py >> /pinbutton: ")
		return Response(json.dumps({"status": "Success!"}), status=200, mimetype="application/json")

	except Exception:
		db.logger.exception('Error pinning generation')
		return Response(status=403, mimetype='application/json')

@extra_routes.route("/finishgenerationbutton/<int:current_generation>/", methods=["GET"])
def finishgenerationbutton(current_generation):
	try:
		import models
		exp = UWPFWP(db.session)
	
		# notify all networks to triage any remaining ps into overflow
		exp.finish_generation(current_generation)

		exp.log("Pushed rollover and finish.", "experiment.py >> /rolloverbutton: ")
		
		return Response(json.dumps({"status": "Success!"}), status=200, mimetype="application/json")

	except Exception:
		db.logger.exception('Error rolling over')
		return Response(status=403, mimetype='application/json')

@extra_routes.route("/manualcompletegeneration/<int:num_recruits>/", methods=["GET"])
def manualcompletegeneration(num_recruits):
	try:
		import models
		exp = UWPFWP(db.session)

		exp.log("Pushed manualcompletegeneration.", "experiment.py >> /manualcompletegeneration: ")

		current_generation = int(exp.current_generation())

		# rollover generation
		exp.rollover_generation()
		exp.save()
	
		# notify all networks to triage any remaining ps into overflow
		exp.finish_generation(current_generation)
		
		# recruit more people
		exp.recruiter.recruit(n = num_recruits)
		
		return Response(json.dumps({"status": "Success!"}), status=200, mimetype="application/json")

	except Exception:
		db.logger.exception('Error rolling over')
		return Response(status=403, mimetype='application/json')

@extra_routes.route("/getcurrentgeneration/", methods=["GET"])
def getcurrentgeneration():
	try:
		import models
		exp = UWPFWP(db.session)
	
		# notify all networks to triage any remaining ps into overflow
		g = exp.current_generation()

		exp.log("exp.current_generation: {}; network generations: {}".format(g, [net.current_generation for net in exp.networks()]), "experiment.py >> /rolloverbutton: ")
		
		return Response(json.dumps({"status": "Success!"}), status=200, mimetype="application/json")

	except Exception:
		db.logger.exception('Error rolling over')
		return Response(status=403, mimetype='application/json')