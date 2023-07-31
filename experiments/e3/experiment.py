from dallinger.networks import Chain
from dallinger.experiment import Experiment
from dallinger import db
from flask import Blueprint, Response
from sqlalchemy import and_, not_, func
from sqlalchemy.sql.expression import true
import numpy as np
import datetime
import json
import random
from operator import attrgetter
from collections import Counter

import pandas as pd
from dallinger.bots import (
    HighPerformanceBotBase,
)
from requests.exceptions import RequestException
import urllib
import requests
from collections import defaultdict
import gevent

import logging

logger = logging.getLogger(__file__)

DEBUG = False

class UWPFWPE3(Experiment):
    """UWPFWP Experiment E3

    - button order is simple randomized in the getsocialinformation route
    - assign utility colors to networks

    """

    @property
    def public_properties(self):
        return {
            "num_replications_per_condition": 5,
            "num_experimental_trials_per_experiment": 2 if DEBUG else 16,
            "num_practice_trials_per_experiment": 1 if DEBUG else 2,
            "generations": 1,
            "generation_size": 1,
            "planned_overflow": 0,
            "yoke": True,
            "first_generation": 1,
            "restart": False,
            "restart_data": None,
            "partial_restart": False,
            "generation_size_table": None,
            "bonus_max": 1.6 # 1.6 if they get everything right ... 
        }

    def __init__(self, session=None):
        super(UWPFWPE3, self).__init__(session)
        from . import models

        self.models = models
        self.set_known_classes()

        # These variables are potentially needed on every invocation
        self.set_params()

        # This is used for data checking and debugging
        self.calculate_info_structure()

        # setup is only needed when launching the experiment
        if session and not self.networks():
            self.setup()
        self.save()

    def set_known_classes(self):
        self.known_classes[
            "particle"
        ] = self.models.Particle
        self.known_classes[
            "acceleratedparticlefilter"
        ] = (
            self.models.AcceleratedParticleFilter
        )
        self.known_classes[
            "decision"
        ] = self.models.Decision
        self.known_classes[
            "comprehensiontest"
        ] = self.models.ComprehensionTest

        self.known_classes[
            "biasReport"
        ] = self.models.BiasReport

    def _initial_recruitment_size(self):
        if not self.yoke:
            if (
                self.restart
                & self.partial_restart
            ):
                data = pd.read_csv(
                    self.generation_size_table
                )
                return (
                    data.drop_duplicates().outstanding.sum()
                    + self.planned_overflow
                )

            return (
                self.generation_size
                * self.num_experiments
            ) + self.planned_overflow

        G0 = len(
            set(
                self.yoking_structure().values()
            )
        )
        return (
            self.generation_size
            * G0
            * self.num_replications_per_condition
        ) + self.planned_overflow

    def set_public_properties(self):
        for (
            key,
            value,
        ) in self.public_properties.items():
            setattr(self, key, value)

    def parse_condition_string(self, s):
        """Return a dictionary of the 
        condition details"""
        keys = [
            "social_condition",
            "sample_type",
            "resampling_condition",
            "randomization_color"
        ]
        values = s.split(":")
        return dict(zip(
            keys,
            values
        ))

    def set_conditions(self):
        self.conditions = [
            'ASO:oversampling:N-R:B',
            # 'ASO:oversampling:N-R:G',
            # 'ASO:resampling:N-R:B',
            # 'ASO:resampling:N-R:G',
        ]

        self.condition_counts = dict(
            zip(
                self.conditions,
                [
                    self.num_replications_per_condition
                ]
                * len(self.conditions),
            )
        )

    def set_initial_recruitment_size(self):
        self.initial_recruitment_size = (
            self._initial_recruitment_size()
        )


    def set_proportions(self):
        practice_trial_proportions = (
            [0.53, 0.47]
            if not DEBUG
            else [0.53]
        )
        random_order_experimental_trial_proportions = (
            [
                0.48,
                0.52,
                0.51,
                0.49,
                0.48,
                0.52,
                0.51,
                0.49,
                0.48,
                0.52,
                0.51,
                0.49,
                0.48,
                0.52,
                0.51,
                0.49
            ]
            if not DEBUG
            else [0.48, .52]
        )
        self.stimulus_proportions = dict(
            zip(
                range(
                    0,self.num_trials_per_participant,
                ),
                practice_trial_proportions
                + random_order_experimental_trial_proportions,
            )
        )

    def create_stimuli(self):
        self.stimuli  = {
            f'stimulus-{i}': {
                'proportion_green': self.stimulus_proportions[i],
                'is_practice': i < self.num_practice_trials_per_experiment,
                'stimulus_id': f'stimulus-{i}',
                'bot_difficulty': np.random.normal(0, 4)
            } for i in range(self.num_trials_per_participant)
        }

    def set_derived_quantities(self):
        self.live_replications = sum(
            self.replication_mask.values()
        )
        self.num_experiments = (
            len(
                set(
                    self.condition_counts.keys()
                )
            )
            * self.live_replications
        )
        self.num_experimental_participants_per_generation = (
            self.generation_size
            * self.num_experiments
        )
        self.num_trials_per_participant = (
            self.num_practice_trials_per_experiment
            + self.num_experimental_trials_per_experiment
        )

    def set_replication_mask(self):
        """Mask off any replications 
        we do not wish to collect data
        in"""
        self.replication_mask = {
            replication: True
            for replication in range(
                self.num_replications_per_condition
            )
        }

    def restart_data_network_alignment(self):
        data = pd.read_csv(self.restart_data)
        conditions = self.session.query(
            self.models.AcceleratedParticleFilter.condition.distinct().label(
                "condition"
            )
        ).all()
        replications = self.session.query(
            self.models.AcceleratedParticleFilter.replication.distinct().label(
                "replication"
            )
        ).all()
        restart_sources = data.groupby(
            ["condition", "replication"]
        )["participant_id"].nunique()

        all_sources_complete = (
            restart_sources
            >= self.generation_size
        ).all()
        if not all_sources_complete:
            return False
        for condition in conditions:
            for replication in replications:
                treatment = (
                    condition.condition,
                    replication.replication,
                )
                complete = (
                    restart_sources.loc[
                        treatment
                    ]
                    >= self.generation_size
                )
                if not complete:
                    return False
        self.log(
            "Restart data is aligned with network conditions and replications.",
            "experiment.py >> restart_data_network_alignment: ",
        )

        # all participants should have
        # shuffle, submit advice, swap
        participant_data_complete = (
            data.groupby("participant_id")[
                "event_type"
            ].nunique()
            >= 2
        )
        if not participant_data_complete.all():
            return False
        self.log(
            "Restart data has a complete inventory of event types for every participant.",
            "experiment.py >> restart_data_network_alignment: ",
        )
        return True

    def check_asserts(self):
        # invalid combinations
        assert ~(self.restart & self.yoke)
        assert ~(
            self.partial_restart
            & (self.generations > 1)
        )
        assert ~(
            self.partial_restart
            & (~self.restart)
        )
        assert ~(
            self.yoke
            & (
                False
                in self.replication_mask.values()
            )
        )

    def set_params(self):
        """
        Notes:
        - A condition is a manipulation
        - An experiment is a single replication of a condition
        - Information does not flow between experiments (excepting yoking)
        """
        self.set_public_properties()
        self.set_conditions()
        self.set_replication_mask()
        self.set_derived_quantities()
        self.set_initial_recruitment_size()
        self.set_proportions()
        self.create_stimuli()
        self.check_asserts()

    # @pysnooper.snoop()
    def yoking_structure(self):
        """{condition: yoked_to}
            - Defaults to {condition: condition} if not self.yoke
            - this means that all nodes have a valid yoke_condition
            - which is useful becasue yoke_condition is checked sometimes in AcceleratedParticleFilter
            - and AcceleratedParticleFilter doens;t have easy access to self.yoke
        """
        structure = self.models.YokeDict()
        if self.yoke:
            sources = [
                condition for condition
                in self.conditions
                if self.parse_condition_string(condition)
                ["social_condition"] == "ASO"
            ]

            matching_criteria = [
                "randomization_color",
                "sample_type"
            ]

            for condition in self.conditions:
                condition_details = self.parse_condition_string(
                    condition
                )
                for source in sources:
                    source_details = self.parse_condition_string(
                        source
                    )
                    for criterion in matching_criteria:
                        matched = (
                            source_details[criterion] 
                            == condition_details[criterion] 
                        )
                        if not matched:
                            break
                    if matched:
                        structure.update({
                            condition:source
                        })
                        break
        return structure

    def current_generation(self):
        """What is the current generation?"""
        try:
            return int(
                self
                .models
                .SupervisorRecords
                .query
                .one()
                .property2
            )
        except:
            return self.first_generation

    def calculate_info_structure(self):
        """"""
        self.correct_counts = {}

    def get_generation_size(
        self, condition, replication
    ):
        """How many ps to recruit every generation in this condition
        replication combo"""
        if not self.replication_mask[
            int(replication)
        ]:
            return 0

        if not self.restart:
            return self.generation_size

        # restarting seeded by generation t - 1
        # partial restart means we're collecting the remainder of
        # an already begun generation t
        # so we need to lookup network-specific generation
        # sizes from self.generation_size_table
        if self.partial_restart:
            data = pd.read_csv(
                self.generation_size_table
            )
            lookup = data.set_index(
                ["condition", "replication"]
            )
            outstanding = lookup.loc[
                (condition, replication)
            ].outstanding
            assert (
                outstanding
                <= self.generation_size
            )
            self.log(
                "Partial Recruitment: {} {} {}".format(
                    condition,
                    replication,
                    outstanding,
                ),
                "experiment.py >> get_generation_size",
            )
            return int(outstanding)
        return self.generation_size

    def create_network(
        self, condition, replication, yoke_to
    ):
        """"""
        network_exists =  (
            self
            .models
            .AcceleratedParticleFilter
            .query
            .filter(
                self
                .models
                .AcceleratedParticleFilter
                .condition == condition
            )
            .filter(
                self
                .models
                .AcceleratedParticleFilter
                .replication == replication
            )
        ).all()

        if network_exists:
            self.log(
                f"Preventing duplication of network for condition {condition} replication {replication}"
            )
            return

        generation_size = self.get_generation_size(
            condition, replication
        )
        net = self.models.AcceleratedParticleFilter(
            generations=self.generations,
            generation_size=generation_size,
            replication=replication,
            condition=condition,
            yoke=yoke_to,
        )
        self.session.add(net)
        return net

    def create_node(self, network, participant):
        """Make a new node for participants."""
        return network.create_node(participant)

    def setup(self):
        """First time setup."""
        for (
            condition,
            replications,
        ) in self.condition_counts.items():
            for replication in range(
                replications
            ):
                _network = self.create_network(
                    condition=condition,
                    replication=replication,
                    yoke_to=self.yoking_structure()[
                        condition
                    ],
                )
        self.session.commit()

    def supervisor(self):
        experiment_supervisor = self.models.ExperimentSupervisor(
            experiment=self
        )
        tasks = (
            [
                self.models.OverflowTriage,
                self.models.Yoking,
                self.models.Resampling,
                self.models.Recruitment,
            ]
            if self.yoke
            else [
                self.models.OverflowTriage,
                self.models.Resampling,
                self.models.Recruitment,
            ]
        )
        for task in tasks:
            experiment_supervisor.add_task(task)
        experiment_supervisor.run()

    @property
    def clock_tasks(self):
        return [self.supervisor]

    def get_network_for_participant(
        self, participant
    ):
        """Find a network for a participant."""
        router = self.models.NetworkTriage(
            experiment=self
        )
        return router.get_network_for_participant(
            participant
        )

    def format_social_data(self, parents):
        """turn decision infos into a pandas df"""
        raw_data = [] 
        for parent in parents:
            decisions = parent.infos(
                type=self.models.Decision
            )
            for decision in decisions:
                contents = json.loads(
                    decision.contents
                )
                raw_data.append({
                    "origin_id" : decision.origin_id,
                    "importance_weight" : float(decision.importance_weight),
                    "stimulus_id" : contents['stimulus_id'],
                    "is_practice" : bool(contents['is_practice']),
                    "sample_type" : contents['sample_type'],
                    "trial_order" : contents['trial_order'] if 'trial_order' in contents else contents['trial'],
                    "chose_green": int(contents["choice"] == "green"),
                    "condition": parent.condition,
                    "generation": parent.generation,
                    "overflow": parent.overflow,
                })
        result = pd.DataFrame(raw_data)
        if DEBUG:

            # correct num ps
            assert result.origin_id.nunique() == self.generation_size
            
            # correct num decisions
            assert result.shape[0] == (
                self.generation_size 
                * (self.num_experimental_trials_per_experiment + self.num_practice_trials_per_experiment)
            )

            # correct number of decisions per trial
            assert result.trial_order.value_counts().max() == self.generation_size

            # no overflow
            assert ~(result.overflow).all()

            # no condition mismatches
            assert result.condition.nunique() == 1

            # 

        return result

    def save_social_data(self, node, social_data, name):
        """Add json of social info pandas df
        into node details column."""
        details = node.details.copy()
        details[name] = social_data.to_json()
        node.details = details
        self.session.commit() 

    def resampling(self, structured_social_data, node):
        """Resample social data using imporance wedights"""

        def resample(d):
            # mo resampling for practice trials
            if d.is_practice.any():
                if DEBUG:
                    assert d.is_practice.all()
                return d.drop(columns=['stimulus_id'])
            
            # weiighteed resampleing with replacement for main trials
            return d.drop(columns=['stimulus_id']).sample(
                n=self.generation_size,
                weights=d.importance_weight,
                replace=True
            )
        
        # weighted sampling of rows of df with replacement
        resampled_data = (
            structured_social_data
            .groupby('stimulus_id')
            .apply(resample)
        ).reset_index().drop(columns=['level_1'])

        # keep a record of the resampling selections
        self.save_social_data(
            node=node, 
            social_data=resampled_data,
            name="resampled_social_data"
        )

        # calculate k_chose_green among resampled
        # data for each problem
        results = (
            resampled_data
            .groupby('stimulus_id')["chose_green"]
            .sum()
        ).to_dict()
        return results

    # @pysnooper.snoop()
    def transmit_resampled_social_information(self, node, parents):
        structured_data = self.format_social_data(parents)
        self.save_social_data(
            node=node,
            social_data=structured_data,
            name="original_social_data"
        )
        results = self.resampling(structured_data, node)
        return results

    def transmit_veridical_social_information(self, node, parents):
        """pass on decisions from parents"""
        self.save_social_data(
            node=node,
            social_data=self.format_social_data(parents),
            name="original_social_data"
        )
        data = defaultdict(int)
        for parent in parents:
            decisions = parent.infos(
                type=self.models.Decision
            )
            for decision in decisions:
                contents = json.loads(
                    decision.contents
                )
                stimulus_id = contents["stimulus_id"]
                
                data[stimulus_id] += int(
                    contents["choice"] == "green"
                )
        return data

    def transmit_social_information(self, node):
        """send infos from parents to children"""
        parents = node.neighbors(
            direction="from"
        )
        if not parents:
            return

        if 'ASO' in node.condition:
            return {}

        if ('W-R' in node.condition) and ('SOC' in node.condition):
            data = self.transmit_resampled_social_information(node, parents)
        
        else:
            data = self.transmit_veridical_social_information(node, parents)     
        
        return data


    def sample_stimuli_schedule(self, node):
        """sample a schedule of stimuli for
        a new partivipnt"""
        stimuli = self.stimuli.copy()
        practice_trial_indices = list(
            range(
                0,
                self.num_practice_trials_per_experiment,
            )
        )
        experimental_stimuli_indices = list(
            range(
                self.num_practice_trials_per_experiment,
                self.num_practice_trials_per_experiment + self.num_experimental_trials_per_experiment,
            )
        )
        
        # set schedule for practrice trials 
        random.shuffle(
           practice_trial_indices
        ) 

        for i, practice_trial_order in enumerate(practice_trial_indices):
            stimuli[f'stimulus-{practice_trial_order}']['trial_order'] = i
        
        # set schedule for experimental trials (random order)
        random.shuffle(
            experimental_stimuli_indices
        ) 
        for i, experimental_stimulus in enumerate(experimental_stimuli_indices):
            stimuli[f'stimulus-{experimental_stimulus}']['trial_order'] = self.num_practice_trials_per_experiment + i
        
        # add social information
        if int(node.generation) > self.first_generation:
            social_information = self.transmit_social_information(node)
            for stimulus_id in social_information:
                stimuli[stimulus_id]['k_chose_green'] = (
                    social_information[stimulus_id]
                )
        return [v for k,v in stimuli.items()]

    def sample_bot_bias(self, node):
        if node.condition.endswith('B'):
            return np.random.normal(-100, 1)
        if node.condition.endswith('G'):
            return np.random.normal(100, 1)
        self.log("Bot isnt blue or green")

    def sample_bot_decision(self, node, stimulus):
        from jax.scipy.special import expit
        details = node.details.copy()
        bias = float(details['bot_bias'])
        difficulty = float(stimulus['bot_difficulty'])
        p = expit(bias - difficulty)
        return 'green' if random.random() < p else 'blue'

    def add_node_to_network(
        self, node, network
    ):
        """Random sampling of parents from previous generation"""
        network.add_node(node)

    def recruit_on_fail(self, participant):
        nodes = participant.nodes(failed="all")
        if not nodes:
            self.recruiter.recruit(n=1)
            return

        try:
            node_generation = int(
                nodes[0].generation
            )
        except:
            self.recruiter.recruit(n=1)
            return

        if (
            node_generation
            != self.current_generation()
        ):
            self.log(
                "Node generation is not equal to the current generation. Preventing re-recruitment of failed overflow.",
                "experiment.py >> recruit_on_fail: ",
            )
            return
        self.recruiter.recruit(n=1)

    def fail_participant(self, participant):
        """Fail all the nodes of a participant."""
        self.log(
            "Participant {} failed".format(
                participant.id
            ),
            "experiment.py >> fail_participant: ",
        )
        participant_nodes = self.models.Node.query.filter_by(
            participant_id=participant.id,
            failed=False,
        ).all()

        if not participant_nodes:
            self.log(
                "Participant {} created no nodes".format(
                    participant.id
                ),
                "experiment.py >> fail_participant: ",
            )
            return

        for node in participant_nodes:
            node.fail()

    def data_check(self, participant):
        """Check a participants data."""
        key = "experiment.py >> data_check (p = {}): ".format(
            participant.id
        )

        particles = participant.nodes(
            type=self.models.Particle
        )
        if not particles:
            self.log(
                "Failed data_check: participant did not "
                + "create a particle.",
                key,
            )
            return False

        infos = participant.infos()
        if not infos:
            self.log(
                "Failed data_check: participant created "
                + "no infos.",
                key,
            )
            return False

        decisions = [info for info in infos if info.type == 'decision']
        if len(decisions) < self.num_practice_trials_per_experiment:
            self.log(
                "Failed data_check: participant created "
                + f"too few decisions ({len(decisions)}).",
                key,
            )
            return False

        return True

    def bonus(self, participant):
        """Calculate a bonus for participant.
        Paid after data_check but *before* attention check"""
        infos = participant.infos()
        totalbonus = 0
        for info in infos:
            if info.type == "decision":
                contents = json.loads(info.contents)
                if contents["is_practice"] == False:
                    totalbonus += contents["current_bonus_dollars"]

        totalbonus = round(totalbonus, 2)

        if totalbonus > self.bonus_max:
            totalbonus = self.bonus_max
        return totalbonus
    
    # @pysnooper.snoop()
    def attention_check(self, participant=None):
        """Check a participant paid attention."""
        key = "experiment.py >> attention_check: "
        infos = participant.infos()

        if not infos:
            return False

        # comprehension test
        results = [info.passed for info in infos if info.type == 'comprehensiontest']
        passed = np.any(results)

        if not passed:
            self.log("Participant {} failed the comprehnsion test".format(participant.id), key)
            return False

        decisions = [info for info in infos if info.type == 'decision']
        if len(decisions) < self.num_trials_per_participant:
            self.log(
                "Failed data_check: participant created "
                + f"too few decisions ({len(decisions)}).",
                key,
            )
            return False
        return True

    def is_complete(self):
        if (
            not self.models.SupervisorRecords.query.count()
        ):
            return False
        for net in self.networks():
            if not net.network_is_complete(
                final_generation=self.generations
                if self.first_generation == 1
                else self.generations - 1
            ):
                return False
        return True


extra_routes = Blueprint(
    "extra_routes",
    __name__,
    template_folder="templates",
    static_folder="static",
)


@extra_routes.route(
    "/recruitbutton/<int:nparticipants>/",
    methods=["GET"],
)
def recruitbutton(nparticipants):
    try:
        from . import models

        exp = UWPFWPE3(db.session)

        exp.recruiter.recruit(n=nparticipants)
        exp.log(
            "Made {} additional recruitments.".format(
                nparticipants
            ),
            "experiment.py >> /recruitbutton",
        )
        return Response(
            json.dumps({"status": "Success!"}),
            status=200,
            mimetype="application/json",
        )

    except Exception:
        db.logger.exception(
            "Error fetching node info"
        )
        return Response(
            status=403,
            mimetype="application/json",
        )


@extra_routes.route("/pause", methods=["GET"])
def pause():
    """"""
    try:
        from . import models

        exp = UWPFWPE3(db.session)
        supervisor = (
            models.SupervisorRecords.query.one()
        )
        supervisor.paused = "paused"
        db.session.commit()
        exp.log(
            "Paused the experiment.",
            "experiment.py >> /pause: ",
        )
        return Response(
            json.dumps({"status": "Success!"}),
            status=200,
            mimetype="application/json",
        )

    except Exception:
        db.logger.exception("Error pausing")
        return Response(
            status=403,
            mimetype="application/json",
        )


@extra_routes.route("/resume", methods=["GET"])
def resume():
    try:
        from . import models

        exp = UWPFWPE3(db.session)
        supervisor = (
            models.SupervisorRecords.query.one()
        )
        supervisor.paused = "live"
        db.session.commit()
        exp.log(
            "Resumed the experiment.",
            "experiment.py >> /resume: ",
        )
        return Response(
            json.dumps({"status": "Success!"}),
            status=200,
            mimetype="application/json",
        )

    except Exception:
        db.logger.exception("Error resuming")
        return Response(
            status=403,
            mimetype="application/json",
        )


@extra_routes.route(
    "/dallingerdownload/<int:node_id>/",
    methods=["GET"],
)
def dallingerdownload(node_id):
    try:
        from . import models

        exp = UWPFWPE3(db.session)
        node = models.Particle.query.get(
            node_id
        )
        if not node:
            return Response(
                status=403,
                mimetype="application/json",
            )
        network = models.AcceleratedParticleFilter.query.get(
            node.network_id
        )
        stimuli = exp.sample_stimuli_schedule(node)
        condition = exp.parse_condition_string(
            network.condition
        )
        data = {
            "network_id": node.network_id,
            "generation_number": node.generation,
            "stimuli": stimuli,
            "condition_string": network.condition,
            "randomization_color": condition["randomization_color"],
            "sample_type": condition["sample_type"],
            "social_condition": condition["social_condition"],
            "resampling_condition": condition["resampling_condition"],
            "button_order": int(
                random.random() < 0.5
            ),
            "participant_id": node.participant_id,
            "condition_replication": network.replication,
            "n": int(
                exp.generation_size
            ),
        }
        exp.log("{}".format(data), "Data >> : ")
        return Response(
            json.dumps(data),
            status=200,
            mimetype="application/json",
        )

    except Exception:
        db.logger.exception(
            "Error fetching social info"
        )
        return Response(
            status=403,
            mimetype="application/json",
        )


@extra_routes.route(
    "/debugbot/<int:node_id>/", 
    methods=["GET"]
)
def debugbot(node_id):
    try:
        from . import models
        import json
        exp = UWPFWPE3(db.session)
        node = models.Particle.query.get(node_id)
        network = models.AcceleratedParticleFilter.query.get(
            node.network_id
        )
        stimuli = exp.sample_stimuli_schedule(node)
        condition = exp.parse_condition_string(
            network.condition
        )
        
        exp.log("* * * * * * * * * * * * * * * * * * * * * * * * *", "experiment.py >> botroute")
        exp.log("* * * * * * * * * * NEW BOT * * * * * * * * * * *", "experiment.py >> botroute")
        exp.log("* * * * * * * * * * * * * * * * * * * * * * * * *", "experiment.py >> botroute")
        exp.log("                                                 ", "experiment.py >> botroute")

        for i in range(exp.num_trials_per_participant):
            relevant_stimulus = [s for s in stimuli if s['trial_order'] == i][0]
            contents = {
                # "choice":"green" if random.random() < .5 else "blue",
                "choice": exp.sample_bot_decision(node, relevant_stimulus),
                "sample_type": condition['sample_type'],
            }
            contents.update(relevant_stimulus)
            models.Decision(
                origin=node,
                contents=json.dumps(contents)
            )
        db.session.commit()
        return Response(json.dumps({"status": "Success!"}), status=200, mimetype="application/json")

    except Exception:
        db.logger.exception('Error fetching node info')
        return Response(status=403, mimetype='application/json')

class Bot(HighPerformanceBotBase):
    """Bot tasks for experiment participation"""

    def run_experiment(self, **kwargs):
        """Runs the phases of interacting with the experiment
        including signup, participation, signoff, and recording completion.
        """
        self.sign_up()
        self.participate()
        if self.sign_off():
            self.complete_experiment("worker_complete")
        else:
            self.complete_experiment("worker_failed")

    def stochastic_sleep(self):
        # delay = max(1.0 / random.expovariate(0.5), 60.0)
        delay = random.gammavariate(1,10)
        gevent.sleep(delay)

    def post(self, url, data = None):
        try:
            result = requests.post(url, data) if data is not None else requests.post(url)
            result.raise_for_status()
            return result
        except RequestException:
            return False

    def get(self, url, data = None):
        try:
            result = requests.get(url, data) if data is not None else requests.get(url)
            result.raise_for_status()
            return result
        except RequestException:
            return False

    def repeat(self, url, success_func, action_func, max_attempts = 1, data = None):
        failed_attempts = 0
        while failed_attempts < max_attempts:
            result = action_func(url = url, data = data)
            if not result:
                failed_attempts += 1
                self.stochastic_sleep()
                continue
            
            else:
                response = json.loads(result.text)
                if success_func(response):
                    return response
                else:
                    self.stochastic_sleep()
                    continue
        
        return False

    def get_node(self):
        """Obtain a node ina  network from dallinger"""
        url = "{host}/node/{participant_id}".format(host=self.host, participant_id=self.participant_id)
        response = self.repeat(url = url, action_func = self.post, success_func = lambda r: r["node"]['id'])
        if not response:
            self.complete_experiment("worker_failed")
        else:
            self.node_id = int(response["node"]["id"])
            exp = UWPFWPE3(db.session)
            node = exp.models.Particle.query.get(self.node_id)
            details = node.details.copy()
            details['bot_bias'] = exp.sample_bot_bias(node)
            node.details = details
            exp.session.commit()
    
    def debugbot(self):
        url = "{host}/debugbot/{node_id}".format(host=self.host, node_id=self.node_id)
        response = self.repeat(url = url, action_func = self.get, success_func = lambda r: r["status"] == "Success!")
        if not response:
            self.complete_experiment("worker_failed")

    def botchecks(self):
        url = "{host}/botchecks/{participant_id}".format(host=self.host, participant_id=self.participant_id)
        response = self.repeat(url = url, action_func = self.get, success_func = lambda r: r["status"] == "checks performed!")
        if not response:
            self.complete_experiment("worker_failed")        

    def complete_experiment(self, status):
        """Record worker completion status to the experiment server.
        This is done using a GET request to the /worker_complete
        or /worker_failed endpoints.
        """
        self.log("Bot player completing experiment. Status: {}".format(status))
        while True:
            url = "{host}/{status}?participant_id={participant_id}".format(
                host=self.host, participant_id=self.participant_id, status=status
            )
            try:
                result = requests.post(url)
                result.raise_for_status()
            except RequestException:
                self.stochastic_sleep()
                continue
            return result

    def timeline(self):
        return [self.stochastic_sleep, self.get_node, self.debugbot]

    def participate(self):
        """experiment logic"""
        timeline = self.timeline()
        for task in timeline:
            task()