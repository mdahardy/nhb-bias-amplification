import json
import pdb
from operator import attrgetter
import random
# import pysnooper
import statistics
import pandas as pd

from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql.expression import cast
from sqlalchemy import Boolean, Integer
from dallinger.nodes import Source, Agent
from dallinger.models import Info, Node
from dallinger.models import Network, Participant
from dallinger import models
from dallinger.config import get_config
from datetime import datetime, time
from pytz import timezone
from collections import Counter
from ast import literal_eval

from operator import attrgetter
import sys
import random
import json
import numpy as np
from math import lgamma

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
from collections import defaultdict
import pysnooper
import gevent
import numpy as np

import logging

logger = logging.getLogger(__file__)

config = get_config()
config.load()
LOG_LEVELS = [
    logging.DEBUG,
    logging.INFO,
    logging.WARNING,
    logging.ERROR,
    logging.CRITICAL,
]
LOG_LEVEL = LOG_LEVELS[config.get("loglevel")]
logging.basicConfig(format="%(asctime)s %(message)s", level=LOG_LEVEL)


class ExperimentSupervisor:
    """Recruitment Decision-maker"""

    def __init__(self, experiment):
        super(ExperimentSupervisor, self).__init__()
        self.exp = experiment
        self.tasks = []

    def records(self):
        return SupervisorRecords.query.count()

    def create_records(self):
        SupervisorRecords(
            network=AcceleratedParticleFilter.query.first()
        )
        self.exp.session.commit()

    def record_experiment_completion(self):
        records = SupervisorRecords.query.one()
        details = json.loads(records.details).copy()
        details["complete"] = 1
        records.details = json.dumps(details)
        self.log("Recorded completion of the experiment.")

    def experiment_is_complete(self):
        """Returns whether the exp has been completed
        and the on_finish_experiment processing has already
        been performed.
        """
        records = SupervisorRecords.query.one()
        details = json.loads(records.details).copy()
        return bool(int(details["complete"]))

    def paused(self):
        """Is the experiment paused?"""
        return SupervisorRecords.query.one().paused == repr("paused")

    def log(self, message):
        """Log messages"""
        s = ">>>> {} {}".format(
            "models.py >> ExperimentSupervisor > ", message
        )
        logger.info(s)

    ###################################
    ## Generation structured control ##
    ###################################
    def current_generation(self):
        """What is the current generation?"""
        return int(SupervisorRecords.query.one().property2)

    def network_exclusions(self, net, generation):
        """At some generations 
        some networks don't need to be counted.
        """
        skipyoke = (
            self.exp.yoke
            and net.condition
            not in self.exp.yoking_structure().values()
            and generation == self.exp.first_generation
        )
        return skipyoke

    def generation_progress(self, generation):
        """Is this generation complete"""
        summary = {
            "ps_approved": 0,
            "ps_remaining": 0,
            "nets_complete": 0,
            "nets_incomplete": 0,
        }
        for net in self.exp.networks():
            if self.network_exclusions(net, generation):
                continue

            if int(net.generation_size) == 0:
                continue
            report = net.generation_progress(generation=generation)
            summary["ps_approved"] += report["approved"]
            summary["ps_remaining"] += report["remaining"]
            summary["nets_complete"] += report["remaining"] == 0
            summary["nets_incomplete"] += report["remaining"] > 0
        return summary

    def generation_complete(self, state):
        """Is this generation complete"""
        generation = state["generation"]
        for net in self.exp.networks():
            if self.network_exclusions(net, generation):
                continue
            if int(net.generation_size) == 0:
                continue
            if not net.generation_complete(generation=generation):
                return False
        self.log("Generation {} is complete.".format(generation))
        return True

    def experiment_complete(self):
        """Is this experiment complete"""
        return False

    def experiment_state(self):
        """Establish the experiment state"""
        return {
            "paused": self.paused(),
            "generation": self.current_generation(),
            "complete": self.experiment_complete(),
        }

    def rollover_supervisor(self):
        """Inform the supervisor records 
        that we are moving to a new generaiton"""
        records = SupervisorRecords.query.one()
        records.current_generation = (
            int(records.current_generation) + 1
        )
        self.exp.session.commit()
        self.log(
            "SupervisorRecords is now at generation: {}".format(
                int(records.current_generation)
            )
        )

    def rollover_networks(self):
        """Inform the networks that we are 
        moving to a new generation"""
        for net in self.exp.networks():
            net.current_generation = int(net.current_generation) + 1
        self.exp.session.commit()
        msg = "All networks now at generation: {}".format(
            int(net.current_generation)
        )
        self.log(msg)

    def rollover_generation(self):
        """Update the database (nets and supervisorrecords)
        to know the new current generation"""
        self.log("Rolling over to the next generation.")
        self.rollover_supervisor()
        self.rollover_networks()

    ###################################
    ####### SupervisionTask API #######
    ###################################
    def add_task(self, task):
        """Initialise and Add a task 
        to the wokrload"""
        _task = task(experiment=self.exp)
        self.tasks.append(_task)

    def start_experiment(self):
        """Called when the experiment
        first begins"""
        state = self.experiment_state()
        for task in self.tasks:
            task.on_experiment_start(state)

    def finish_experiment(self, state):
        """Called when the experiment
        ends"""
        for task in self.tasks:
            task.on_experiment_finish(state)

        self.record_experiment_completion()
        self.exp.session.commit()

    def start_generation(self, state):
        """Called when a generation
        begins"""
        for task in self.tasks:
            task.on_generation_start(state)

    def finish_generation(self, state):
        """Called when a generation
        ends"""
        for task in self.tasks:
            task.on_generation_finish(state)
        self.rollover_generation()

    def iteration(self, state):
        """Called every iteration
        of the supervision loop"""
        self.log(
            "New Iteration ** Generation {}".format(
                state["generation"]
            )
        )
        self.log(
            "Progress: {}".format(
                self.generation_progress(state["generation"])
            )
        )
        for task in self.tasks:
            task.on_iteration(state)

    ###################################
    ### Thread & Schedule Structure ##
    ###################################
    def policy(self):
        state = self.experiment_state()
        if self.generation_complete(state):
            if state["generation"] == self.exp.generations:
                if not self.experiment_is_complete():
                    self.log("Finishing the experiment.")
                    self.finish_experiment(state)
                else:
                    return True
            else:
                self.finish_generation(state)
                self.start_generation(state)

        else:
            self.iteration(state)

    def run(self):
        """"""
        # first iteration
        if not self.records():
            self.create_records()
            self.start_experiment()

        complete = self.policy()
        if complete:
            self.log("Supervision complete. Doing nothing.")


class SupervisionTask:
    """docstring for SupervisionTask"""

    def __init__(self, experiment):
        super(SupervisionTask, self).__init__()
        self.exp = experiment

    def _key(self):
        """Log messages"""
        return "models.py >> SupervisionTask > {}: ".format(
            self.__class__.__name__
        )

    def log(self, message):
        """Log messages"""
        s = ">>>> {} {}".format(self._key(), message)
        logger.info(s)

    def on_experiment_start(self, experiment_state):
        """Called by the supervisor
         when the experiment starts"""
        pass

    def on_experiment_finish(self, experiment_state):
        """Called by the supervisor
         when the experiment starts"""
        pass

    def on_generation_start(self, experiment_state):
        """Called by the supervisor
         when a new generation begins"""
        pass

    def on_generation_finish(self, experiment_state):
        """Called by the supervisor
         when a new generation ends"""
        pass

    def on_iteration(self, experiment_state):
        """Called by the supervisor
         when a new generation ends"""
        pass


class Yoking(SupervisionTask):
    """Yoking Thread. 

    Connects gen 2 nodes from one network
    up to gen one nodes of another network.
    Used to implement the same initialisation
    across conditions.

    Begins by creating dummy gen 1 nodes in yoked
    networks at the start of the experiment. On
    each iteration of the supervision thread,
    this class checks whether ther are any new
    approved non-overflow nodes in the seeding 
    networks waiting to be connected up to any
    networks yoked to that seed. 

    Clones infos created by seeding nodes. Creates 
    copies with the yoked node as origin.

    Should run after OverflowTriage in the
    supervision loop. Only runs as gen 1
    and if self.exp.yoke == True. Yoked nodes
    have no participant_ids.
    """

    def __init__(self, experiment):
        super(Yoking, self).__init__(experiment)

    def yoking_complete(self):
        """"""
        yoked = (
            self.exp.session.query(CloneRecord.destination_id)
            .distinct()
            .count()
        )

        target_conditions = [
            target_condition
            for (
                target_condition,
                seed_conditon,
            ) in self.exp.yoking_structure().items()
            if target_condition != seed_conditon
        ]

        total_replications = sum(
            [
                replications
                for (
                    condition,
                    replications,
                ) in self.exp.condition_counts.items()
                if condition in target_conditions
            ]
        )

        return yoked == total_replications * self.exp.generation_size

    def clone(self, origin, destination):
        ignored_types = [
            CloneRecord,
            BiasReport,
            ComprehensionTest
        ]
        for info in origin.infos():
            infotype = info.__class__
            if infotype in ignored_types:
                continue

            infotype(
                origin=destination, contents=info.contents
            )

        destination.participant_id = origin.participant_id
        destination.randomisation_profile = (
            origin.randomisation_profile
        )
        CloneRecord(origin=origin, destination=destination)

    def queue(self):
        cloned = [
            x[0]
            for x in self.exp.session.query(CloneRecord.origin_id)
            .distinct()
            .all()
        ]

        self.log("previously cloned: {}".format(len(cloned)))

        waiting = (
            self.exp.session.query(Particle)
            .join(
                Participant, Participant.id == Particle.participant_id
            )
            .filter(
                Particle.property5.in_(
                    self.exp.yoking_structure().values()
                )
            )
            .filter(Particle.failed == False)
            .filter(~Particle.id.in_(cloned))
            .filter(Participant.status == "approved")
            .filter(Particle.overflow == 0)
            .all()
        )
        self.log("waiting: {}".format(len(waiting)))

        remaining = (
            self.exp.initial_recruitment_size
            - self.exp.planned_overflow
            - len(cloned)
            - len(waiting)
        )
        self.log("remaining: {}".format(remaining))

        return {
            "cloned": cloned,
            "waiting": waiting,
            "remaining": remaining,
        }

    def get_seeding_nodes_waiting(self, queue):
        """Get all nodes ready to seed another network"""
        seed_nodes_waiting = defaultdict(list)
        for node in queue["waiting"]:
            if node.overflow == 1:
                continue
            seed_nodes_waiting[
                (node.condition, node.network_id)
            ].append(node)
        return seed_nodes_waiting

    def get_already_yoked_nodes_query(self, queue):
        """Get all nodes alread yoked up"""
        return (
            self.exp.session.query(CloneRecord.destination_id)
            .distinct()
            .all()
            if len(queue["cloned"]) > 0
            else []
        )

    def get_already_yoked_nodes(self, queue):
        """Get all nodes alread yoked up"""

        if len(queue["cloned"]) == 0:
            return []

        clonde_records = (
            CloneRecord
            .query
            .all()
        )
        return list(set([
            int(c.destination_id)
            for c in clonde_records
        ]))

    def get_target_nodes_waiting(self, excluded_nodes):
        particles =  (
            Particle
            .query
            .filter(
                Particle.generation == self.exp.first_generation
            )
            .filter(
                Particle.overflow == 0
            )
            .filter(
                not_(
                    Particle
                    .property5
                    .in_(self.exp.yoking_structure().values())
                )
            )
            .all()
        )
        target_nodes_waiting = [
            particle for particle in particles
            if particle.id not in excluded_nodes
        ]
        return target_nodes_waiting

    def get_network_id(self, condition, replication):
        """Find the id of the network 
        corresponding to: (condition, replciation) """
        return (
            AcceleratedParticleFilter.query.filter(
                AcceleratedParticleFilter.condition == condition
            )
            .filter(
                AcceleratedParticleFilter.replication == replication
            )
            .one()
        ).id

    def yoke_seed_to_target(
        self,
        seeding_condition,
        target_condition,
        replication,
        seed_nodes_waiting,
        target_nodes_already_yoked,
        target_nodes_waiting,
    ):
        seeding_network_id = self.get_network_id(
            seeding_condition, replication
        )
        target_network_id = self.get_network_id(
            target_condition, replication
        )
        seed_nodes = seed_nodes_waiting[
            (seeding_condition, seeding_network_id)
        ]
        if len(seed_nodes) == 0:
            return

        # self.log("Yoking condition {} replication {} to seeding condition {} replication {}".format(target_condition, replication, seeding_condition, replication))
        # self.log("# Seed nodes waiting to be yoked in this condition:replication: {}".format(len(seed_nodes)))
        target_nodes_waiting_this_net = [
            node
            for node in target_nodes_waiting
            if node.network_id == target_network_id
        ]

        # self.log("seed_nodes: {}".format([(n.id, n.condition, n.network_id, n.failed) for n in seed_nodes]))
        # self.log("# Target nodes waiting to be yoked in this condition:replication: {}".format(len(target_nodes_waiting_this_net)))
        # self.log("target_nodes_waiting_this_net: {}".format(target_nodes_waiting_this_net))
        assert len(seed_nodes) <= len(target_nodes_waiting_this_net)
        target_nodes = random.sample(
            target_nodes_waiting_this_net, len(seed_nodes)
        )
        # self.log("target_nodes: {}".format(target_nodes))
        for i, seed_node in enumerate(seed_nodes):
            self.clone(origin=seed_node, destination=target_nodes[i])

    ###################################
    ## Main Recruitment Loop Control ##
    ###################################
    def yoke(self, queue):
        if len(queue["waiting"]) == 0:
            return

        k = 0
        seed_nodes_waiting = self.get_seeding_nodes_waiting(queue)
        target_nodes_already_yoked = self.get_already_yoked_nodes(
            queue
        )
        yoking_structure = self.exp.yoking_structure()
        target_nodes_waiting = self.get_target_nodes_waiting(
            excluded_nodes=target_nodes_already_yoked
        )

        attested_seed_conditions = [
            c for (c, idx) in seed_nodes_waiting.keys()
        ]
        for (
            target_condition,
            seeding_condition,
        ) in yoking_structure.items():
            if target_condition == seeding_condition:
                continue

            if not (seeding_condition in attested_seed_conditions):
                continue

            k += 1
            for replication in range(
                self.exp.condition_counts[target_condition]
            ):
                self.yoke_seed_to_target(
                    seeding_condition=seeding_condition,
                    target_condition=target_condition,
                    replication=replication,
                    seed_nodes_waiting=seed_nodes_waiting,
                    target_nodes_already_yoked=target_nodes_already_yoked,
                    target_nodes_waiting=target_nodes_waiting,
                )
        self.exp.session.commit()
        self.log("Yoked {} conditions".format(k))

    def sign_off(self):
        """Inform the supervisor that yoking has been completed."""
        s = SupervisorRecords.query.one()
        s.yoking_complete = 1
        self.exp.session.commit()

    def proceed_with_yoking(self, state):
        return (
            state["generation"] == self.exp.first_generation
        ) & self.exp.yoke

    ###################################
    ## Supervision Interface Methds. ##
    ###################################
    def on_experiment_start(self, experiment_state):
        """At setup create dummy nodes for yoked conditions"""
        targets = [
            condition
            for condition in self.exp.conditions
            if condition not in self.exp.yoking_structure().values()
        ]

        target_networks = (
            AcceleratedParticleFilter.query.filter(
                AcceleratedParticleFilter.condition.in_(targets)
            )
            .filter_by(failed=False)
            .all()
        )

        for net in target_networks:
            for k in range(self.exp.generation_size):
                Particle(
                    network=net,
                    participant=None,
                    contents=None,
                    details=None,
                    randomisation_profile=None,
                    generation=1,
                    condition=net.condition,
                    overflow=0,
                )
        self.exp.session.commit()

    def on_iteration(self, experiment_state):
        """Single-threaded supervisor managing yoking"""
        self.log("YOKER")
        if not self.proceed_with_yoking(state=experiment_state):
            return
        try:
            current_queue = self.queue()
            self.yoke(current_queue)
            if current_queue["remaining"] == 0:
                if self.yoking_complete():
                    self.sign_off()
                    return

        except Exception as e:
            self.log("YOKER EXCEPTION: {}".format(e))

    def on_generation_finish(self, experiment_state):
        self.on_iteration(experiment_state)


class Bonusing(SupervisionTask):
    """docstring for BonusStructure"""

    def __init__(self, experiment):
        super(Bonusing, self).__init__(experiment)

    def get_trial_bonus(self, trial):
        reward = trial["correct"] * (
            0.8 * min([10.0 / max([trial["num_swaps"], 1]), 1]) ** 2
            + 0.2
        )
        assert 0 <= reward <= 1
        return reward

    def get_trial_bonuses(self, trials):
        """Bonuses for each trial"""
        return [
            self.exp.bonus_payment * self.get_trial_bonus(trial)
            for trial in trials
            if not trial["is_practice"]
        ]

    def get_trials(self, participant):
        """Return all sorting trials
         and demonstration trials
         """
        return [
            json.loads(trial.contents)
            for trial in sorted(
                filter(
                    lambda info: (info.property1 == "finished")
                    & (info.type == "event"),
                    participant.infos(),
                ),
                key=lambda x: x.creation_time,
                reverse=False,
            )
        ]

    def get_swaps(self, trials):
        """How many swaps were made in total?"""
        return sum(
            [
                int(trial["num_swaps"])
                for trial in trials
                if ~(trial["is_practice"])
            ]
        )

    def too_few_swaps(self, trials):
        """Did this particiipant make enough swaps?"
        """
        total_swaps = self.get_swaps(trials)
        return total_swaps <= (
            self.exp.public_properties["num_trials"]
            - self.exp.practice_trials
        )

    def calculatebonus(self, participant):
        """Reward function from swaps & sucesses to a bonus"""
        trials = self.get_trials(participant)
        trial_bonuses = self.get_trial_bonuses(trials)
        self.log(
            "Trial Bonuses (p = {}): {}".format(
                participant.id, [round(b, 2) for b in trial_bonuses]
            )
        )
        reward = round(np.mean(trial_bonuses), 2)
        if reward > self.exp.bonus_payment:
            reward = self.exp.bonus_payment
        if reward < 0:
            reward = 0
        return reward

    def record_bonus(self, origin, bonus, pseudo):
        """Keep a record 
        that a descendant bonus has been awarded.
        Note that the participant AWARDED the bonnus is the origin 
        of the descendantbonus. The descendant who performed well is
        property2 of the descendantbonus."""
        _record = DescendantBonus(
            origin=origin, bonus=bonus, pseudo=pseudo
        )

    def pay_bonus(self, origin, bonus):
        """Pay the cultural parent."""
        participant = Participant.query.get(origin.participant_id)
        if not participant:
            return
        self.exp.recruiter.reward_bonus(
            participant.assignment_id,
            bonus,
            self.exp.transmission_bonus_reason,
        )
        return True

    def execute_ancestor_bonus(
        self, ancestor_id, bonusamount, pseudo=0
    ):
        """"""
        if bonusamount < 0.01:
            return False

        ancestor_nodes = Particle.query.filter_by(
            participant_id=ancestor_id
        ).all()

        # when yoking, one p can have multiple nodes
        # b/c their participant_id gets tagged onto clones
        # always use the node that wasn't a clone
        if len(ancestor_nodes) > 1:
            ancestor_node = [
                node
                for node in ancestor_nodes
                if node.condition
                in self.exp.yoking_structure().values()
            ][0]

        else:
            ancestor_node = ancestor_nodes[0]

        prior_bonuses = DescendantBonus.query.filter_by(
            origin_id=ancestor_id
        ).one_or_none()

        if ancestor_node and (not prior_bonuses):
            payment = (
                bonusamount * self.exp.transmission_bonus_coeficient
            ) + self.exp.descendant_bonus_constant
            self.record_bonus(
                origin=ancestor_node, bonus=payment, pseudo=pseudo
            )
            self.pay_bonus(origin=ancestor_node, bonus=payment)

            return True

        return False

    def get_generation_selected_nodes(self, generation):
        return (
            db.session.query(Particle)
            .join(
                Participant, Participant.id == Particle.participant_id
            )
            .filter(~Particle.failed)
            .filter(Particle.overflow == 0)
            .filter(Particle.generation == generation)
            .filter(Participant.status == "approved")
            .all()
        )

    def get_generation_oveflow_nodes(self, generation):
        return (
            db.session.query(Particle)
            .join(
                Participant, Participant.id == Particle.participant_id
            )
            .filter(~Particle.failed)
            .filter(Particle.overflow == 1)
            .filter(Particle.generation == generation)
            .filter(Participant.status == "approved")
            .all()
        )

    def get_generation_all_nodes(self, generation):
        query = (
            db.session.query(Particle)
            .join(
                Participant, Participant.id == Particle.participant_id
            )
            .filter(~Particle.failed)
            .filter(Particle.generation == generation)
            .filter(Participant.status == "approved")
        )

        # remove any cloned nodes in G1 when yoking
        if (generation == self.exp.first_generation) & (
            self.exp.yoke
        ):
            query = query.filter(
                Particle.condition.in_(
                    self.exp.yoking_structure().values()
                )
            )
        return query.all()

    def get_participant_bonus(self, participant_id):
        """get bonus from participant table"""
        bonus = Participant.query.get(participant_id).bonus
        if bonus >= self.exp.completion_bonus:
            return bonus - self.exp.completion_bonus
        return bonus

    def get_mean_descendant_bonuses(self, descendant_data):
        all_bonuses = defaultdict(list)
        for dataset in descendant_data.values():
            for (ancestor, count) in dataset["parents"].items():
                if count:
                    all_bonuses[int(ancestor)].append(
                        dataset["bonus"]
                    )
        return {
            ancestor: statistics.mean(bonuses)
            for (ancestor, bonuses) in all_bonuses.items()
        }

    def was_yoked(self, node):
        return (
            (self.exp.yoke)
            & (node.generation == self.exp.first_generation + 1)
            & (
                node.condition
                not in self.exp.yoking_structure().values()
            )
        )

    def get_descendant_data(self, descendant_nodes):
        """Reutrn the parent selecitons and bonus 
        amounts for all descendant_nodes"""
        return {
            int(descendant.participant_id): {
                "parents": json.loads(descendant.details)["parents"],
                "bonus": self.get_participant_bonus(
                    descendant.participant_id
                ),
            }
            for descendant in descendant_nodes
            if not self.was_yoked(descendant)
        }

    ###################################
    ## Supervision Interface Methds. ##
    ###################################
    def on_generation_finish(self, experiment_state):
        """Calculate and send bonuses to 
        ancestors and non ancestors"""
        descendant_generation = experiment_state["generation"]
        if descendant_generation == self.exp.first_generation:
            return

        descendants = self.get_generation_all_nodes(
            generation=descendant_generation
        )
        descendant_data = self.get_descendant_data(
            descendant_nodes=descendants
        )
        descendant_bonuses = self.get_mean_descendant_bonuses(
            descendant_data
        )

        all_previous_generation_nodes = self.get_generation_all_nodes(
            generation=descendant_generation - 1
        )
        n_ancestors = 0
        total = 0
        actually_executed = 0
        not_executed = 0
        for node in all_previous_generation_nodes:
            is_ancestor = bool(
                int(node.participant_id) in descendant_bonuses
            )
            bonusamount = (
                descendant_bonuses[int(node.participant_id)]
                if is_ancestor
                else self.get_participant_bonus(
                    participant_id=node.participant_id
                )
            )
            executed = self.execute_ancestor_bonus(
                ancestor_id=node.participant_id,
                pseudo=0 if is_ancestor else 1,
                bonusamount=bonusamount,
            )
            if executed:
                actually_executed += 1
                n_ancestors += int(is_ancestor)
                total += (
                    float(bonusamount)
                    * self.exp.transmission_bonus_coeficient
                ) + self.exp.descendant_bonus_constant
            else:
                not_executed += 1

        self.log(
            "Awarded {} ancestry bonuses (ancestors = {}) totalling ${}. Skipped {}.".format(
                actually_executed, n_ancestors, total, not_executed
            )
        )

    def on_experiment_finish(self, experiment_state):
        """Bonus the final genration"""
        self.on_generation_finish(experiment_state)

class Resampling(SupervisionTask):
    """docstring for BonusStructure"""

    def __init__(self, experiment):
        super(Resampling, self).__init__(experiment)

    def recode(self, series):
        """label encode the values
        in an iterable"""
        return series.map(
            dict(zip(sorted(series.unique()),
            range(series.nunique())))
        )

    def reverse_coding(self, iterable):
        """Returns a map from the
        code createed by recode to 
        the original values. e.g. 
        gettin gback the original participant
        ids from the coded indices"""
        series = pd.Series(iterable)
        return dict(zip(range(series.nunique()), sorted(series.unique())))

    def get_decision_data(self, decisions):
        """
        Extract decisions, participant_ids,
        problem_ids in vector form
        from a set of decision infos
        """
        data = {}
        data['choice_data'] = [
            json.loads(decision.contents)
            for decision in decisions
        ]
        data['original_chose_green'] = pd.Series([
            int(cd['choice'] == "green")
            for cd in data['choice_data']
        ])
        data['chose_green'] = self.recode(
            data['original_chose_green']
        )
        data['original_stimulus_identifier'] = pd.Series([
            cd['stimulus_id']
            for cd in data['choice_data']
        ])
        data['stimulus_identifier'] = self.recode(
            data['original_stimulus_identifier']
        )
        data['original_participant_identifier'] = pd.Series([
            # use participant id rather than node if b/c yoking
            int(Particle.query.get(int(d.origin_id)).participant_id)
            for d in decisions
        ])
        data['participant_identifier'] = self.recode(
            data['original_participant_identifier']
        )
        data['n_participants'] = (
            data['participant_identifier']
            .nunique()
        )
        data['n_problems'] = (
            data['stimulus_identifier']
            .nunique()
        )
        data['n_decisions'] = (
            len(data['chose_green'])
        )
        return data

    # @pysnooper.snoop()
    def irt(self, decisions):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        from jax import random as jaxrandom

        def model():
            """
            Specify the IRT model
            in numpyro
            """
            data = self.get_decision_data(decisions)
            with numpyro.plate('participants', data['n_participants']):
                b = numpyro.sample("b", dist.Normal(0, 3))

            with numpyro.plate('problems', data['n_problems']):
                d = numpyro.sample("d", dist.Normal(0, 3))
            
            with numpyro.plate("data", data['n_decisions']):
                p_idx = data['participant_identifier'].values
                s_idx = data['stimulus_identifier'].values
                binary_decisions = data['chose_green'].values
                linear_combinations = (
                    b[p_idx] 
                    - d[s_idx]
                )
                numpyro.sample(
                    'obs',
                    dist.Bernoulli(
                        logits=linear_combinations
                    ), 
                    obs=binary_decisions
                )

        def format_irt_posterior_samples(samples, decisions):
            data = self.get_decision_data(decisions)
            
            # here i'm taking extra care to ensure that
            # the ordering of columns in the mcmc samples for b
            # aligns with the ordering of participant identifiers
            # The columns in the mcmc samples reflect the coded participant identifiers
            # these are 0 - n, mapped to the sorted original ids 
            # we can use this to be sure of the order alignment by
            # looping over range(num_participants) and using i to index
            # into the reversed mapping, to avoid relying on the ordering
            # of the mapping dict items
            original_participant_identifiers = self.reverse_coding(
                data['original_participant_identifier']
            )
            num_unique_participants = len(
                original_participant_identifiers
            )
            bias_columns = [
                original_participant_identifiers[i]
                for i in range(num_unique_participants)
            ]
            bias = pd.DataFrame(
                samples['b'],
                columns=bias_columns
            )

            # same approach for the difficulties
            original_stimulus_identifiers = (
                self.reverse_coding(
                    data['original_stimulus_identifier']
                )
            )
            num_unique_problems = len(
                original_stimulus_identifiers
            )
            difficulty_columns = [
                original_stimulus_identifiers[i]
                for i in range(num_unique_problems)
            ]
            difficulties = pd.DataFrame(
                samples['d'],
                columns=difficulty_columns
            )
            return {'b': bias, 'd':difficulties}

        def save_posterior_summary(formatted_samples):
            """Save a summery of the posterior
            over bias coefficients our to the node
            table for every participant"""

            def save_node_summary(series, extra_details):
                nodes = Particle.query.filter(
                    Particle.participant_id == int(series.name)
                ).all()
                series_summary = (
                    series
                    .describe()
                    .to_json()
                )
                # one participant id can have multiplenodes b/c yoking
                for node in nodes:
                    details = node.details.copy()
                    details['bias_posterior_summary'] = series_summary
                    for (k, v) in extra_details.items():
                        details[k] = v
                    node.details = details
    
            difficulties = formatted_samples['d'].describe().to_json()
            formatted_samples['b'].apply(
                save_node_summary, 
                extra_details={
                    'difficulties_posterior_summary': difficulties
                }, 
                axis=0
            )

        nuts_kernel = NUTS(model)
        self.log("Kernel constructed. Starting MCMC")
        # self.log("NOTE: -- -- -- Incomplete sampler, change to 2500 samples, 8 chains, half warmup -- --- --")
        mcmc = MCMC(nuts_kernel, num_warmup=1250, num_samples=2500, num_chains = 8, progress_bar=False)
        rng_key = jaxrandom.PRNGKey(0)
        mcmc.run(rng_key)
        samples = mcmc.get_samples()
        formatted_samples = format_irt_posterior_samples(samples, decisions)
        save_posterior_summary(formatted_samples)
        self.log(
            "numpyro sbf model has been fit."
        )
        return formatted_samples


    def record_weights(
            self,
            weights,
            decisions
        ):
        """extract weights and record
        in the info table"""
        for decision in decisions:
            participant_id = Particle.query.get(decision.origin_id).participant_id
            decision.importance_weight = weights[participant_id].iloc[0]

    def calculate_q(self, bias_samples, difficulty_samples, y, integrate=True):
        '''calculate q(y) given
        samples of b and d'''
        from jax.scipy.special import expit
        biased_model = bias_samples.values - difficulty_samples.values
        q = expit(biased_model)

        np.log(np.mean(q))
        if y:
            return np.log(np.mean(q))
        else:
            return np.log(np.mean(1 - q))

    def calculate_p(self, difficulty_samples, y, integrate=True):
        '''calculate p(y) given
        samples of b and d'''
        from jax.scipy.special import expit
        unbiased_model = 0 - difficulty_samples
        p = expit(unbiased_model)
        if y:
            return np.log(np.mean(p))
        else:
            return np.log(np.mean(1 - p))
   
    def calculate_log_importance_weights(self, bias_samples, difficulty_samples, y):
        """N1 GMIS Scheme"""
        q_y = self.calculate_q(
            bias_samples,
            difficulty_samples,
            y
        )
        p_y = self.calculate_p(
            difficulty_samples,
            y
        )
        return p_y - q_y

    def normalize_importance_weights(self, unnormalized_weights):
        """Normalize log importance weights along rows"""
        from jax.scipy.special import logsumexp
        normalizing_constants = logsumexp(unnormalized_weights.values, axis=1, keepdims=True)
        log_normalized_weights_samples = unnormalized_weights.values - normalizing_constants
        return log_normalized_weights_samples

    def calculate_importance_weights_for_problem(
        self, stimulus_id, network_id, decisions, 
        posterior_samples
        ):
        """
        For a given problem, 
        sample importance weights for a set of
        decisions made by participants
        facing that problem
        """
        difficulty_samples = posterior_samples['d'][stimulus_id]
        log_weights_samples_by_participant = {}
        for decision in decisions:
            y = int(json.loads(decision.contents)['choice'] == 'green')
            participant_id = (
                Particle
                .query
                .get(int(decision.origin_id))
                .participant_id
                )
            bias_samples = posterior_samples['b'][
                participant_id
            ]
            log_importance_weights = self.calculate_log_importance_weights(
                bias_samples,
                difficulty_samples,
                y
            )
            if participant_id in log_weights_samples_by_participant:
                self.log(f"Warning. Participant_id {participant_id} is already in the results dictionary. Overwriting previous entry.")
            log_weights_samples_by_participant[participant_id] = log_importance_weights

        results = pd.DataFrame(
            log_weights_samples_by_participant, index=[0]
        )
        log_normalized_weights = self.normalize_importance_weights(
            results
        )
        resampling_probabilities = np.exp(log_normalized_weights)
        resampling_probabilities[resampling_probabilities > 1] = 1
        resampling_probabilities[resampling_probabilities < 0] = 0
        return pd.DataFrame(
            resampling_probabilities,
            columns=results.columns
        )

    def sample_weights_for_network(self, network, formatted_irt_posrterior_samples, decisions):
        """
        Sample importance weights for decisions
        made by participants in this network
        for every problem (stimulus_id)
        """
        self.log(f"Sampling importance weights for network {network.id} ({network.condition})")
        network_decisions = [
            decision for decision in decisions
            if decision.network_id == network.id
        ]
        data = self.get_decision_data(decisions)
        for stimulus_id in data['original_stimulus_identifier'].unique():
            stimulus_decisions = [
                decision for decision
                in network_decisions
                if decision.stimulus_id
                == stimulus_id
            ]
            if not stimulus_decisions:
                self.log(f"No decisions for stimulus {stimulus_id}")
                continue
            self.log(f"Sampling weights for {len(stimulus_decisions)} decisions on stimulus {stimulus_id}")
            sampled_importance_weights = self.calculate_importance_weights_for_problem(
                stimulus_id=stimulus_id,
                network_id=network.id,
                decisions=stimulus_decisions,
                posterior_samples=formatted_irt_posrterior_samples
            )
            self.record_weights(
                weights=sampled_importance_weights,
                decisions=stimulus_decisions
            )
        self.exp.session.commit()

    def decisions_to_model(self, generation):
        """Return the decision infos that should 
        be used to fit the participant model. THe decisions
        in the set will determine the participants that 
        go into the model."""
        decision_query = (
            db.session.query(Decision)
            .join(
                Particle,
                Particle.id 
                == Decision.origin_id
            )
            .filter(
                Decision.generation 
                == generation
            )
            .filter(
                Particle.condition 
                .contains('SOC')
            )
            .filter(
                Particle.overflow
                == 0
            )
            .filter(
                Decision.is_practice
                == 0
            )
            .filter(not_(Particle.failed))
            .filter(not_(Decision.failed))
        )

        # Watch out for yking!
        # if first gen, only model one set of "participants" per matchin yoked group
        if generation == self.exp.first_generation:
            yoking_structure = self.exp.yoking_structure()
            distinct_source_conditions = set(
                yoking_structure
                .values()
            )
            target_conditions = yoking_structure.keys()
            modelling_groups = {}

            # make sure every unique source has exactly one 
            # group of yoked participants to model
            for source_condition in distinct_source_conditions:
                if source_condition in modelling_groups:
                    continue
                
                for target_condition in target_conditions:
                    # dont choose asocial
                    if 'ASO' in target_condition:
                        continue
                    
                    if yoking_structure[target_condition] == source_condition:
                        modelling_groups[source_condition] = target_condition
                        break
            self.log(
                f"First generation [only modelling one condition per yoked source: {modelling_groups}]"
            )
            # subset the decision data down to the groups
            # that should be modelled (i.e. avoid modelling clones)
            decision_query = decision_query.filter(
                Particle.condition.in_(
                    list(modelling_groups.values())
                ) 
            )

        decisions = decision_query.all()
        if len(decisions) == 0:
            self.log("No decisions to model and resample. Doing nothing.")
            return
        decisions_summary = Counter(getattr(decision, 'origin_id') for decision in decisions)
        self.log("Decisions to model: {}, {}".format(len(decisions), decisions_summary))
        return decisions

    def decisions_to_resample(self, generation):
        """Return the set of decisions that should be
        subject to resampling for the next generation.
        i.e. all decisions made by approved social participants"""
        decisions = (
            db.session.query(Decision)
            .join(
                Particle,
                Particle.id 
                == Decision.origin_id
            )
            .filter(
                Decision.generation 
                == generation
            )
            .filter(
                Particle.condition 
                .contains('SOC')
            )
            .filter(
                Particle.overflow
                == 0
            )
            .filter(
                Decision.is_practice
                == 0
            )
            .filter(not_(Particle.failed))
            .filter(not_(Decision.failed))
        ).all()

        if len(decisions) == 0:
            self.log("No decisions to model and resample. Doing nothing.")
            return
        decisions_summary = Counter(getattr(decision, 'origin_id') for decision in decisions)
        self.log("Decisions to resample: {}, {}".format(len(decisions), decisions_summary))
        return decisions

    def fit_participant_model(self, generation):
        decisions = self.decisions_to_model(generation)
        posterior_samples = self.irt(decisions)
        return posterior_samples

    def use_participant_model_for_resampling(self, posterior_samples, generation):
        """"""
        decisions = self.decisions_to_resample(generation)
        networks = (
            AcceleratedParticleFilter
            .query
            .filter(
                not_(
                    AcceleratedParticleFilter
                    .failed
                )
            )
        ).all()
        num_nets = 0
        for net in networks:
            if not ('SOC' in net.condition):
                continue

            if not ('W-R' in net.condition):
                continue
            num_nets += 1
            self.sample_weights_for_network(
                network=net,
                formatted_irt_posrterior_samples=posterior_samples,
                decisions=decisions
            )
        self.log(
            "Sampled importance weights for {} networks."
            .format(num_nets)
        )
        pass

    def on_generation_finish(self, experiment_state):
        """"""
        g = experiment_state["generation"]
        self.log(f"Hello from the resampler! gen = {g}")
        posterior_samples = self.fit_participant_model(generation=g)
        self.use_participant_model_for_resampling(
            posterior_samples, 
            generation=g
        )

    def on_iteration(self, experiment_state):
        pass


class TimeKeeping(SupervisionTask):
    """Recruitment Hours:
    - Start Curfew (prevent recruitment) = 14:30pm PDT = 21:30pm UTC = 5:30pm EDT
    - End Curfew (allow recruitment) = 10am EDT = 14:00pm UTC = 7am PDT
    
    Old curfew:
    - End Curfew (allow recruitment) = 8am EDT = 12pm UTC = 5am PDT
    - Start Curfew (prevent recruitment) = 6pm PDT = 1am UTC = 9pm EDT
    """

    def __init__(self, experiment):
        super(TimeKeeping, self).__init__(experiment)
        self.curfew_start = time(21, 30)  # UTC
        self.curfew_end = time(14, 0)  # UTC
        # self.curfew_start = time(19, 40) # UTC
        # self.curfew_end = time(19, 45) # UTC
        self.autoresume = True

    def experiment_started(self):
        return True if SupervisorRecords.query.count() else False

    def is_paused(self, experiment_state):
        """Is experiment paused according to 
        the supervisor?"""
        return experiment_state["paused"]

    def is_valid_time(self):
        UTC = timezone("UTC")
        now = datetime.now(UTC)
        self.log("Time now (UTC) is: {}".format(now.time()))
        if self.curfew_start < self.curfew_end:
            in_curfew_window = (now.time() >= self.curfew_start) and (
                now.time() <= self.curfew_end
            )
        else:
            in_curfew_window = (now.time() >= self.curfew_start) or (
                now.time() <= self.curfew_end
            )
        return False if in_curfew_window else True

    def click_pause(self):
        records = SupervisorRecords.query.one()
        records.paused = "paused"
        db.session.commit()
        self.log("Clicked pause.")

    def click_resume(self):
        records = SupervisorRecords.query.one()
        records.paused = "live"
        db.session.commit()
        self.log("Clicked resume.")

    ###################################
    ## Supervision Interface Methds. ##
    ###################################
    def on_iteration(self, experiment_state):
        """Checks the curfew requirments,
        clicks pause and resume"""
        if not self.experiment_started():
            return False

        paused = self.is_paused(experiment_state)
        valid = self.is_valid_time()
        self.log("Paused: {}".format(paused))
        self.log("Valid time: {}".format(valid))
        if paused & valid and self.autoresume:
            self.click_resume()
        if (not paused) & (not valid):
            self.click_pause()
        return False


class Recruitment(SupervisionTask):
    """Recruitment Decision making"""

    def __init__(self, experiment):
        super(Recruitment, self).__init__(experiment)

    def stage_pending_recruitment(self):
        """Update state records to 
        reflect a pending new generation."""
        records = SupervisorRecords.query.one()
        records.recruitment_pending = 1
        self.exp.session.commit()
        self.log("Staged pending recruitment.")

    def resolve_pending_recruitment(self):
        """Update state records to 
        reflect a resolved new generation."""
        records = SupervisorRecords.query.one()
        records.recruitment_pending = 0
        self.exp.session.commit()

    def recruitment_pending(self):
        return bool(SupervisorRecords.query.one().recruitment_pending)

    def overflow_uptake_count(self, generation):
        """How many participants were assigned to the
        overflow this generation across all networks?"""
        return sum(
            [
                net.overflow_uptake(generation=generation)
                for net in AcceleratedParticleFilter.query.all()
            ]
        )

    def calculate_required_overrecruitment(self, new_generation):
        """How many additional recruitments are required?"""
        return min(
            [
                self.exp.planned_overflow,
                self.overflow_uptake_count(
                    generation=new_generation - 1
                ),
            ]
        )

    def recruit_new_generation(self, finishing_generation):
        """Calculate and make the required recruitment"""
        required_overflow = self.calculate_required_overrecruitment(
            new_generation=finishing_generation + 1
        )
        self.log(
            "Required overflow refill at the next generation is: {}.".format(
                required_overflow
            )
        )
        sample_size = (
            self.exp.generation_size * (self.exp.num_experiments)
        ) + required_overflow
        self.exp.recruiter.recruit(n=sample_size)
        self.log("Made {} new recruitments".format(sample_size))

    ###################################
    ## Supervision Interface Methds. ##
    ###################################
    def on_generation_start(self, experiment_state):
        """Stage or execute a recruitment"""
        # final generation
        if experiment_state["generation"] == self.exp.generations:
            self.log("Final generation, not recruiting.")
            return

        if experiment_state["paused"]:
            self.log("Experiment is paused. Blocking recruitment.")
            self.stage_pending_recruitment()
            return

        self.recruit_new_generation(
            finishing_generation=experiment_state["generation"]
        )

    def on_iteration(self, experiment_state):
        """Excecute a pending recruitment"""
        if not self.recruitment_pending():
            return

        self.log("Recruitment pending.")
        if experiment_state["paused"]:
            self.log("Experiment remains paused. Not recruiting.")
            return

        self.log("Experiment is live. Resolving pending recruitment.")
        self.recruit_new_generation(
            finishing_generation=experiment_state["generation"] - 1
        )
        self.resolve_pending_recruitment()


class OverflowTriage(SupervisionTask):
    """Recruitment curfew"""

    def __init__(self, experiment):
        super(OverflowTriage, self).__init__(experiment)

    def network_exclusions(self, net, generation):
        """At some generations 
        some networks don't need to be counted.
        """
        skipyoke = (
            self.exp.yoke
            and net.condition
            not in self.exp.yoking_structure().values()
            and generation == self.exp.first_generation
        )
        return skipyoke

    ###################################
    ## Supervision Interface Methds. ##
    ###################################
    def on_iteration(self, experiment_state):
        """"""
        generation = experiment_state["generation"]
        nets = self.exp.networks()
        for net in nets:
            if self.network_exclusions(net, generation):
                continue
            if int(net.generation_size) == 0:
                continue
            net.select_nodes(generation=generation)
        self.log(
            "Overflow triage decisions in {} networks completed.".format(
                len(nets)
            )
        )

    def on_generation_finish(self, experiment_state):
        """Treat as normal iteration"""
        self.on_iteration(experiment_state)

    def on_experiment_finish(self, experiment_state):
        """Treat as normal iteration"""
        self.on_iteration(experiment_state)


class Checks(SupervisionTask):
    """docstring for TestExperiment"""

    def __init__(self, experiment):
        super(Checks, self).__init__(experiment)

    def network_exclusions(self, net, generation):
        """At some generations 
        some networks don't need to be counted.
        """
        skipyoke = (
            self.exp.yoke
            and net.condition
            not in self.exp.yoking_structure().values()
            and generation == self.exp.first_generation
        )
        return skipyoke

    def test_generation_size(self, net, generation):
        particles = (
            db.session.query(Particle)
            .join(
                Participant, Participant.id == Particle.participant_id
            )
            .filter(Particle.network_id == net.id)
            .filter(not_(Particle.failed))
            .filter(Particle.overflow == 0)
            .filter(Particle.generation == int(generation))
            .filter(Participant.status == "approved")
        )

        return particles.count() == net.generation_size

    def test_network_structure(self):
        """Test that all networks are the correct shape"""
        for net in self.exp.networks():
            for generation in self.exp.generations:
                self.test_generation_size(net, generation)

    ###################################
    ## Supervision Interface Methds. ##
    ###################################
    def on_iteration(self, experiment_state):
        """"""
        generation = experiment_state["generation"]
        nets = self.exp.networks()
        for net in nets:
            if self.network_exclusions(net, generation):
                continue
            self.test_generation_size(net, generation)


class NetworkTriage(object):
    """docstring for NetworkTriage"""

    def __init__(self, experiment):
        super(NetworkTriage, self).__init__()
        self.exp = experiment

    def log(self, message, key="models.py >> NetworkTriage: "):
        """Log messages through the expeirments logger"""
        s = ">>>> {} {}".format(key, message)
        logger.info(s)

    def triage(self, priorities):
        """Sample from the network proprity queue"""
        valid_nets = ([
                net for net in self.exp.networks()
                if self.exp.replication_mask[int(net.replication)]
        ])
        if not priorities:
            self.log("No priority Queue. Returning random network.")
            return random.choice(valid_nets)

        # self.log("Priority Queue: {}".format({net.id: priority for net, priority in priorities.items()}))
        return random.choice(
            [
                net
                for net in priorities.keys()
                if (
                    (np.isclose(priorities[net], max(priorities.values())))
                    and (net in valid_nets))
            ]
        )

    # @pysnooper.snoop()
    def get_network_for_new_participant(self, participant):
        """Construct and sample the network priority que"""
        valid_nets = ([
                net for net in self.exp.networks()
                if self.exp.replication_mask[int(net.replication)]
        ])
        network_priorities = dict(
            [
                (net, net.priority())
                for net in AcceleratedParticleFilter.query.all()
                if net in valid_nets
            ]
        )

        if (
            self.exp.current_generation() == self.exp.first_generation
        ) and self.exp.yoke:
            network_priorities = {
                net: priority
                for net, priority in network_priorities.items()
                if net.condition
                in self.exp.yoking_structure().values()
            }

        return self.triage(network_priorities)

    def get_network_for_participant(self, participant):
        """Find a network for a participant."""
        participant_nodes = participant.nodes()
        if participant_nodes:
            self.log(
                "Participant {} is requesting a network but already created nodes.".format(
                    participant.id
                )
            )
            self.log(
                "Nodes for participant {}: {}.".format(
                    participant.id, participant_nodes
                )
            )
            return None

        chosen_network = self.get_network_for_new_participant(
            participant
        )
        if chosen_network is not None:
            self.log(
                "Assigned participant {} to network: {}; Condition: {}; Replication: {}; Generation: {}".format(
                    participant.id,
                    chosen_network.id,
                    chosen_network.condition,
                    chosen_network.replication,
                    chosen_network.current_generation,
                )
            )
        else:
            self.log(
                "Participant {} requested a network but was assigned None.".format(
                    participant.id
                )
            )
        return chosen_network


class AcceleratedParticleFilter(Network):
    """Discrete fixed size generations with Overflow"""

    __mapper_args__ = {"polymorphic_identity": "particlefilter"}

    def __init__(
        self,
        generation_size,
        generations,
        replication,
        condition,
        yoke,
    ):
        """Endow the network with some persistent properties."""
        self.current_generation = 1
        self.generation_size = generation_size
        self.max_size = (
            generations * generation_size + 2
        )  # add one to account for initial_source; one for random_attributes
        self.replication = replication
        self.condition = condition
        self.completion_probability = 0.85
        self._yoke(yoke_condition=yoke)

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
    def yoke_condition(self):
        """Make property4 yoke_condition."""
        return self.property4

    @yoke_condition.setter
    def yoke_condition(self, yoke_condition):
        """Make yoke_condition settable."""
        self.property4 = yoke_condition

    @yoke_condition.expression
    def yoke_condition(self):
        """Make yoke_condition queryable."""
        return self.property4

    @hybrid_property
    def condition(self):
        """Make property5 condition."""
        return self.property5

    @condition.setter
    def condition(self, condition):
        """Make condition settable."""
        self.property5 = condition

    @condition.expression
    def condition(self):
        """Make condition queryable."""
        return self.property5

    def log(self, text, key=None):
        """Print a string to the logs."""
        if key is None:
            key = "models.py >> ParticleFilter (ID:{};C:{};R:{};G:{}) >> ".format(
                self.id,
                self.condition,
                self.replication,
                self.current_generation,
            )
        s = ">>>> {} {}".format(key, text)
        print(s)
        sys.stdout.flush()
        logger.debug(s)

    def _yoke(self, yoke_condition):
        self.yoke_condition = yoke_condition

    @property
    def randomisation_quota(self):
        return {0: self.generation_size}

    def roster(self, generation, ignore_id=None):
        """How many people are workig / confirmed in each randomisation profile?"""
        counts = (
            db.session.query(
                func.count(Particle.participant_id.distinct()).label(
                    "count"
                ),
                Particle.condition,
                Participant.status,
                Particle.randomisation_profile,
                Particle.overflow,
            )
            .join(
                Participant, Participant.id == Particle.participant_id
            )
            .group_by(
                Particle.condition,
                Participant.status,
                Particle.randomisation_profile,
                Particle.overflow,
            )
            .filter(Particle.network_id == self.id)
            .filter(Particle.generation == int(generation))
            .filter(~Particle.failed)
            .filter(~Participant.failed)
        )

        if ignore_id is not None:
            counts = counts.filter(
                Particle.participant_id != ignore_id
            )

        fields = [
            "count",
            "node_condition",
            "participant_status",
            "node_profile",
            "node_overflow",
        ]

        return [dict(zip(fields, record)) for record in counts.all()]

    def randomisation_profile_working(self, roster, profile):
        """How many participants remain working?"""
        finished = ["approved", "bad_data", "did_not_attend"]
        return sum(
            [
                record["count"]
                for record in roster
                if record["node_profile"] == profile
                and record["participant_status"] not in finished
            ]
        )

    def randomisation_profile_num_approved(self, roster, profile):
        """How many participants have completed this profile?"""
        return sum(
            [
                record["count"]
                for record in roster
                if record["node_profile"] == profile
                and record["participant_status"] == "approved"
            ]
        )

    def randomisation_profile_approved_shortfall(
        self, roster, profile
    ):
        """How many participants remain working?"""
        target = self.randomisation_quota[profile]
        completed = self.randomisation_profile_num_approved(
            roster, profile
        )
        return target - completed

    def randomisation_profile_selected_shortfall(
        self, roster, profile
    ):
        """How many more ps must be selected to makw quota?"""
        target = self.randomisation_quota[profile]
        completed = self.randomisation_profile_num_selected(
            roster, profile
        )
        return target - completed

    def randomisation_profile_quota_approved(self, roster, profile):
        """Has this randomisation profile got all the approved ps
        it needs?"""
        return (
            self.randomisation_profile_num_approved(roster, profile)
            >= self.randomisation_quota[profile]
        )

    def randomisation_profile_num_selected(self, roster, profile):
        """How many nodes have already been selected for transmission?"""
        return sum(
            [
                record["count"]
                for record in roster
                if record["node_profile"] == profile
                and record["participant_status"] == "approved"
                and record["node_overflow"] == 0
            ]
        )

    def randomisation_profile_quota_selected(self, roster, profile):
        """Has this randomisation profile got all the selected nodes
        it needs?"""
        return (
            self.randomisation_profile_num_selected(roster, profile)
            >= self.randomisation_quota[profile]
        )

    def randomisation_profile_empty(self, roster, profile):
        """Has this randomisation been started?"""
        for record in roster:
            if record["node_profile"] == profile:
                return False
        return True

    def working(self, roster):
        """w_i for all i"""
        return {
            profile: self.randomisation_profile_working(
                roster, profile
            )
            for profile in self.randomisation_quota.keys()
        }

    def shortfall(self, roster):
        """t_i - c_i"""
        return {
            profile: self.randomisation_profile_approved_shortfall(
                roster, profile
            )
            for profile in self.randomisation_quota.keys()
        }

    # @pysnooper.snoop()
    def generation_complete(self, roster=None, generation=None):
        """Is this generation finished?"""
        if int(self.generation_size) == 0:
            return True

        if roster is None:
            assert generation is not None
            roster = self.roster(generation=generation)

        for profile in self.randomisation_quota.keys():
            if not self.randomisation_profile_quota_approved(
                roster=roster, profile=profile
            ):
                return False
        return True

    # @pysnooper.snoop()
    def generation_processing_complete(
        self, roster=None, generation=None
    ):
        """Is this generation finished?"""
        if int(self.generation_size) == 0:
            return True

        if roster is None:
            assert generation is not None
            roster = self.roster(generation=generation)

        for profile in self.randomisation_quota.keys():
            if not self.randomisation_profile_quota_selected(
                roster=roster, profile=profile
            ):
                return False
        return True

    def generation_progress(self, generation):
        """How many ps are approved and remaining?"""
        if int(self.generation_size) == 0:
            return {
                "approved": 0,
                "remaining": 0
            }
            
        roster = self.roster(generation=generation)
        report = [
            {
                "target": target,
                "approved": min(
                    [
                        self.randomisation_profile_num_approved(
                            roster, profile
                        ),
                        target,
                    ]
                ),
            }
            for (profile, target) in self.randomisation_quota.items()
        ]
        return {
            "approved": sum(
                [record["approved"] for record in report]
            ),
            "remaining": sum(
                [
                    record["target"] - record["approved"]
                    for record in report
                ]
            ),
        }

    # @pysnooper.snoop()
    def overflow_uptake(self, roster=None, generation=None):
        if int(self.generation_size) == 0:
            return 0
        if roster is None:
            if generation is None:
                generation = self.current_generation
            roster = self.roster(generation=generation)

        return sum(
            [
                record["count"]
                for record in roster
                if bool(record["node_overflow"])
            ]
        )

    def priority(self):
        """Global priority of the network"""
        if int(self.generation_size) == 0:
            return -np.inf

        roster = self.roster(int(self.current_generation))
        weights = self.compute_weights(roster=roster)

        # if wieghts is empty, no rcs have shortfall
        if not weights:
            return -np.inf

        return max(weights.values())

    # @pysnooper.snoop()
    def compute_weights(self, roster):
        """Compute priority weights using the beta-binomial formulation"""

        def logsumexp(v):
            v = np.array(v)
            maxval = np.max(v)
            d = v - maxval
            sums = np.exp(d).sum()
            return maxval + np.log(sums)

        def log_bb(n, k, a, b):
            """
            https://en.wikipedia.org/wiki/Beta-binomial_distribution
            https://stackoverflow.com/questions/26935127/beta-binomial-function-in-python
            """
            return (
                lgamma(n + 1)
                + lgamma(k + a)
                + lgamma(n - k + b)
                + lgamma(a + b)
                - (
                    lgamma(k + 1)
                    + lgamma(n - k + 1)
                    + lgamma(a)
                    + lgamma(b)
                    + lgamma(n + a + b)
                )
            )

        def log_bb_cdf(working, shortfall, a, b):
            """logarithm of cdf for beta-binomial distributoin 0 up to k - 1"""
            if working == 0 or working < shortfall:
                return np.log(1)
            return logsumexp(
                [
                    log_bb(n=working, k=j, a=a, b=b)
                    for j in range(shortfall)
                ]
            )

        working, shortfall = (
            self.working(roster),
            self.shortfall(roster),
        )
        weights = {
            profile: log_bb_cdf(
                shortfall=shortfall[profile],
                working=working[profile],
                a=10,
                b=1,
            )
            for profile in self.randomisation_quota.keys()
            if shortfall[profile] > 0
        }

        return weights

    def assign_randomisation_profile(self, participant):
        roster = self.roster(
            ignore_id=participant.id,
            generation=self.current_generation,
        )

        # no nodes yet created
        if not roster:
            return random.choice(
                list(self.randomisation_quota.keys())
            )

        # the generation is over
        if self.generation_complete(roster=roster):
            return random.choice(
                list(self.randomisation_quota.keys())
            )

        priorities = self.compute_weights(roster)
        if not priorities:
            return random.choice(
                list(self.randomisation_quota.keys())
            )

        return random.choice(
            [
                profile
                for profile in priorities.keys()
                if np.isclose(
                    priorities[profile], max(priorities.values())
                )
            ]
        )

    def select_nodes(self, generation):
        """Flip some nodes out of the OVF"""
        if int(self.generation_size) == 0:
            return

        roster = self.roster(generation)
        # self.log("Roster: {}".format(roster))

        for randomisation_profile in self.randomisation_quota.keys():
            if self.randomisation_profile_quota_selected(
                roster, randomisation_profile
            ):
                continue

            if self.randomisation_profile_empty(
                roster, randomisation_profile
            ):
                continue

            # self.log("RP {} is not complete".format(randomisation_profile))
            potentials = (
                db.session.query(Particle)
                .join(
                    Participant,
                    Participant.id == Particle.participant_id,
                )
                .filter(Particle.network_id == self.id)
                .filter(Particle.failed == False)
                .filter(Particle.generation == generation)
                .filter(Participant.status == "approved")
                .filter(Particle.overflow == 1)
                .filter(
                    Particle.randomisation_profile
                    == randomisation_profile
                )
                .all()
            )

            # self.log("# Candidate nodes: {}".format(len(potentials)))
            if len(potentials) == 0:
                continue

            current_shortfall = self.randomisation_profile_selected_shortfall(
                roster, randomisation_profile
            )
            k = min([len(potentials), current_shortfall])
            chosen_nodes = random.sample(potentials, k)
            for node in chosen_nodes:
                # self.log("Lifting node: {}".format(node))
                node.overflow = 0
        db.session.commit()

    def processing_complete(self, generation):
        if int(self.generation_size) == 0:
            return True
        roster = self.roster(generation)
        for profile in self.randomisation_quota.keys():
            if not self.randomisation_profile_quota_selected(
                roster=roster, profile=profile
            ):
                return False
        return True

    def create_node(self, participant, randomisation_profile=None):
        if randomisation_profile is None:
            randomisation_profile = self.assign_randomisation_profile(
                participant
            )
        new_node = Particle(
            network=self,
            participant=participant,
            randomisation_profile=randomisation_profile,
            generation=self.current_generation,
            condition=self.condition,
            overflow=1,
        )
        db.session.commit()
        return new_node

    def add_node(self, node):
        """Link to parents"""

        # special treatment for the first generation
        if int(node.generation) == 1:
            return

        # find the node's parents in the DB
        # if this is a yoked generation
        # then parent particles wont have approved participant ids
        # therefore the query is different
        if (int(node.generation) == 2) and (
            self.yoke_condition != self.condition
        ):
            parents_query = (
                db.session.query(Particle)
                .filter(Particle.network_id == self.id)
                .filter(Particle.failed == False)
                .filter(Particle.overflow == 0)
                .filter(
                    Particle.generation == int(node.generation) - 1
                )
            )

        # this is the main parent query
        # for all except yoked generations
        else:
            parents_query = (
                db.session.query(Particle)
                .join(
                    Participant,
                    Participant.id == Particle.participant_id,
                )
                .filter(Particle.network_id == self.id)
                .filter(Particle.failed == False)
                .filter(Particle.overflow == 0)
                .filter(
                    Particle.generation == int(node.generation) - 1
                )
                .filter(Participant.status == "approved")
            )

        parents = parents_query.all()
        assert len(parents) >= self.generation_size
        if parents:
            for parent in parents:
                parent.connect(whom=node)

    def calculate_full(self):
        return False

    # @pysnooper.snoop()
    def network_is_complete(self, final_generation):
        """"""
        return (
            self.current_generation
            >= final_generation
            & self.generation_complete(generation=final_generation)
        )


class Particle(Node):
    """The Rogers Agent."""

    __mapper_args__ = {"polymorphic_identity": "particle"}

    @hybrid_property
    def overflow(self):
        """Convert property1 to genertion."""
        return int(self.property1)

    @overflow.setter
    def overflow(self, overflow):
        """Make overflow settable."""
        self.property1 = repr(overflow)

    @overflow.expression
    def overflow(self):
        """Make overflow queryable."""
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
    def randomisation_profile(self):
        """Convert property3 to randomisation_profile."""
        return int(self.property3)

    @randomisation_profile.setter
    def randomisation_profile(self, randomisation_profile):
        """Mark randomisation_profile settable."""
        self.property3 = repr(randomisation_profile)

    @randomisation_profile.expression
    def randomisation_profile(self):
        """Make randomisation_profile queryable."""
        return cast(self.property3, Integer)

    @hybrid_property
    def condition(self):
        """Make property5 condition."""
        return self.property5

    @condition.setter
    def condition(self, condition):
        """Make condition settable."""
        self.property5 = condition

    @condition.expression
    def condition(self):
        """Make condition queryable."""
        return self.property5

    def __init__(
        self,
        network,
        participant=None,
        contents=None,
        details=None,
        randomisation_profile=None,
        generation=None,
        condition=None,
        overflow=0,
    ):
        """Create a node."""
        # check the network hasn't failed
        if network.failed:
            raise ValueError(
                "Cannot create node in {} as it has failed".format(
                    network
                )
            )
        # check the participant hasn't failed
        if participant is not None and participant.failed:
            raise ValueError(
                "{} cannot create a node as it has failed".format(
                    participant
                )
            )

        # check the participant is working
        if (
            participant is not None
            and participant.status != "working"
        ):
            raise ValueError(
                "{} cannot create a node as they are not working".format(
                    participant
                )
            )

        self.network = network
        self.network_id = network.id
        self.generation = generation
        self.overflow = overflow
        network.calculate_full()

        if participant is not None:
            self.participant = participant
            self.participant_id = participant.id

        self.condition = condition
        self.randomisation_profile = randomisation_profile


class Decision(Info):
    """An Info that represents one decision trial."""

    @declared_attr
    def __mapper_args__(cls):
        """The name of the source is derived from its class name."""
        return {"polymorphic_identity": cls.__name__.lower()}

    @hybrid_property
    def stimulus_id(self):
        """Convert property1 to genertion."""
        return self.property1

    @stimulus_id.setter
    def stimulus_id(self, stimulus_id):
        """Make stimulus_id settable."""
        self.property1 = stimulus_id

    @stimulus_id.expression
    def stimulus_id(self):
        """Make stimulus_id queryable."""
        return self.property1

    @hybrid_property
    def choice(self):
        """Convert property2 to genertion."""
        return int(self.property2)

    @choice.setter
    def choice(self, choice):
        """Make choice settable."""
        self.property2 = repr(choice)

    @choice.expression
    def choice(self):
        """Make choice queryable."""
        return cast(self.property2, Integer)

    @hybrid_property
    def importance_weight(self):
        """Convert property3 to genertion."""
        return float(self.property3)

    @importance_weight.setter
    def importance_weight(self, importance_weight):
        """Make importance_weight settable."""
        self.property3 = repr(importance_weight)

    @importance_weight.expression
    def importance_weight(self):
        """Make importance_weight queryable."""
        return cast(self.property3, Float)

    @hybrid_property
    def generation(self):
        """Convert property4 to genertion."""
        return int(self.property4)

    @generation.setter
    def generation(self, generation):
        """Make generation settable."""
        self.property4 = repr(generation)

    @generation.expression
    def generation(self):
        """Make generation queryable."""
        return cast(self.property4, Integer)

    @hybrid_property
    def is_practice(self):
        """Convert property5 to genertion."""
        return int(self.property5)

    @is_practice.setter
    def is_practice(self, is_practice):
        """Make is_practice settable."""
        self.property5 = repr(is_practice)

    @is_practice.expression
    def is_practice(self):
        """Make is_practice queryable."""
        return cast(self.property5, Integer)

    def parse_data(self):
        data = json.loads(
            self.contents
        )
        self.stimulus_id = data['stimulus_id']
        self.choice = int(
            data["choice"] == "green"
        )
        self.importance_weight = 1.
        self.is_practice = int(data['is_practice'])

    def __init__(
        self,
        origin,
        contents=None,
        details=None
    ):
        self.origin = origin
        self.origin_id = origin.id
        self.network_id = origin.network_id
        self.network = origin.network
        self.generation = origin.generation
        self.contents = contents
        self.parse_data()

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
        return int(self.property1)

    @passed.setter
    def passed(self, p):
        """Assign passed to property1."""
        self.property1 = int(p)

    @passed.expression
    def passed(self):
        """Retrieve passed via property1."""
        return cast(self.property1, Integer)

    def evaluate_answers(self):
        return int(all([q == "1" for q in self.questions.values()]))

    def __init__(self, origin, contents=None, details = None, initialparametrisation = None):
        self.origin = origin
        self.origin_id = origin.id
        self.network_id = origin.network_id
        self.network = origin.network
        self.questions = json.loads(contents)
        self.passed = self.evaluate_answers()
        self.contents = contents

class BiasReport(Info):

    @declared_attr
    def __mapper_args__(cls):
        """The name of the source is derived from its class name."""
        return {
            "polymorphic_identity": cls.__name__.lower()
        }

class SupervisorRecords(Node):
    """The experiment's referee.
    The referee can pause and resume the experiment.
    """

    @declared_attr
    def __mapper_args__(cls):
        """The name of the source is derived from its class name."""
        return {"polymorphic_identity": cls.__name__.lower()}

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

    @hybrid_property
    def current_generation(self):
        """Make property2 current_generation."""
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
    def recruitment_pending(self):
        """Make property3 recruitment_pending."""
        return int(self.property3)

    @recruitment_pending.setter
    def recruitment_pending(self, recruitment_pending):
        """Make recruitment_pending settable."""
        self.property3 = repr(recruitment_pending)

    @recruitment_pending.expression
    def recruitment_pending(self):
        """Make recruitment_pending queryable."""
        return cast(self.property3, Integer)

    @hybrid_property
    def yoking_complete(self):
        """Make property4 yoking_complete."""
        return int(self.property4)

    @yoking_complete.setter
    def yoking_complete(self, yoking_complete):
        """Make yoking_complete settable."""
        self.property4 = repr(yoking_complete)

    @yoking_complete.expression
    def yoking_complete(self):
        """Make yoking_complete queryable."""
        return cast(self.property4, Integer)

    @hybrid_property
    def rollover_pending(self):
        """Make property5 rollover_pending."""
        return int(self.property5)

    @rollover_pending.setter
    def rollover_pending(self, rollover_pending):
        """Make rollover_pending settable."""
        self.property5 = repr(rollover_pending)

    @rollover_pending.expression
    def rollover_pending(self):
        """Make rollover_pending queryable."""
        return cast(self.property5, Integer)

    def __init__(self, network, details=None):
        super(SupervisorRecords, self).__init__(network)
        self.paused = "live"
        self.current_generation = 1
        self.recruitment_pending = 0
        self.yoking_complete = 0
        self.rolled_over = 0
        self.details = json.dumps({"complete": 0})


class CloneRecord(Info):
    """An Info that records a cloned node."""

    @declared_attr
    def __mapper_args__(cls):
        """The name of the source is derived from its class name."""
        return {"polymorphic_identity": cls.__name__.lower()}

    @hybrid_property
    def destination_id(self):
        """Use property2 to store the technology's destination_id."""
        try:
            return int(self.property2)
        except TypeError:
            return None

    @destination_id.setter
    def destination_id(self, idx):
        """Assign destination_id to property2."""
        self.property2 = idx

    @destination_id.expression
    def destination_id(self):
        """Retrieve destination_id via property2."""
        return cast(self.property2, Integer)

    @hybrid_property
    def origin_condition(self):
        """Make property3 origin_condition."""
        return self.property3

    @origin_condition.setter
    def origin_condition(self, origin_condition):
        """Make origin_condition settable."""
        self.property3 = origin_condition

    @origin_condition.expression
    def origin_condition(self):
        """Make origin_condition queryable."""
        return self.property3

    @hybrid_property
    def destination_condition(self):
        """Make property4 destination_condition."""
        return self.property4

    @destination_condition.setter
    def destination_condition(self, destination_condition):
        """Make destination_condition settable."""
        self.property4 = destination_condition

    @destination_condition.expression
    def destination_condition(self):
        """Make destination_condition queryable."""
        return self.property4

    def __init__(
        self, origin, destination, contents=None, details=None
    ):
        self.origin = origin
        self.origin_id = origin.id
        self.origin_condition = origin.condition
        self.network_id = origin.network_id
        self.network = origin.network
        self.destination_id = destination.id
        self.destination_condition = destination.condition

class YokeDict(dict):
    def __missing__(self, key):
        return key
