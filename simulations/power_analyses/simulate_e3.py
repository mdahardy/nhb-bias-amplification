import networkx as nx
import random
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random as jaxrandom
import sys
import json
import os
import copy

ORACLE_PARAMETERS = {
    'asocial':{
        'absolute_bias': 0.2582982,
        'bias_sd': 0.05304466,
        'green_dummies': {
            '0.48': 0.1371524,
            '0.49': 0.3467926,
            '0.51': 0.9927337,
            '0.52': 1.2856189
        },
        'intercept': -0.5834531
    },

    'social':{
        'absolute_bias':0.3698188,
        'bias_sd': 0.4683244,
        'green_dummies': {
            '0.48': -0.96784148,
            '0.49': -0.62858374,
            '0.51': 0.02834432,
            '0.52': 0.12247819
        },
        'intercept': -0.4441746,
        'vote_coefficient':0.2332352
    }
}

class ExperimentSimulator:
    def __init__(self, num_replications):
        self.num_replications = num_replications
        self.num_participants_per_condition = 8 * 2 * num_replications 
        self.estimated_biases = {}
        self.estimated_difficulties = {}
        self.stimuli = {0:'0.48',1:'0.48',2:'0.48',3:'0.48',4:'0.49',5:'0.49',6:'0.49',7:'0.49',8:'0.51',9:'0.51',10: '0.51',11: '0.51',12: '0.52',13: '0.52',14: '0.52',15: '0.52',}
        self.choices = {}
        self.social_observations = {}
        self.conditions = ['asocial','oversampling','resampling']
        self.true_biases = {}

    def get_participant_id(self, condition, replication, color, participant_index):
        # All counting is zero-indexed
        condition_index = self.conditions.index(condition)
        num_conditions = len(self.conditions)
        color_index = 0 if color == 'blue' else 1
        return condition_index * self.num_replications * 2 * 8 + replication * 2 * 8 + color_index * 8 + participant_index

    def get_participant_details(self, participant_id):
        num_conditions = len(self.conditions)
        condition = participant_id // ( self.num_replications * 2 * 8)
        condition = self.conditions[condition]
        participant_id %=  self.num_replications * 2 * 8

        replication = participant_id // (2 * 8)
        participant_id %= 2 * 8

        color_id = participant_id // 8
        color = 'blue' if color_id == 0 else 'green'

        participant = participant_id % 8

        return condition, replication, color, participant


    def initialize_participants(self):
        for condition in ['asocial','oversampling','resampling']:
            self.true_biases[condition] = {}
            for replication in range(self.num_replications):
                self.true_biases[condition][replication] = {}
                for color in ['blue','green']:
                    self.true_biases[condition][replication][color] = {}
                    for i in range(8):
                        params = ORACLE_PARAMETERS['asocial'] if condition == 'asocial' else ORACLE_PARAMETERS['social']
                        raw_bias = random.normalvariate(params['absolute_bias'],params['bias_sd'])
                        self.true_biases[condition][replication][color][i] = raw_bias if color == 'green' else raw_bias * -1

    def simulate_choice(self,stimulus_id,green_bias,observations,params,decision_type):
        logit_accumulator = params['intercept']
        logit_accumulator += params['green_dummies'][self.stimuli[stimulus_id]]
        logit_accumulator += green_bias
        if decision_type == 'social':
            logit_accumulator += params['vote_coefficient'] * observations
        prob_green = 1 / (1 + math.exp(-logit_accumulator))
        choice = random.random()
        if choice < prob_green:
            return 1
        else:
            return 0

    def sample_choices(self, condition):
        decision_type = 'asocial' if condition == 'asocial' else 'social'
        decision_parameters = ORACLE_PARAMETERS[decision_type]
        # Loop over keys of stimuli dict and stimulate choice for each stimulus
        choices = {}
        for replication in range(self.num_replications):
            choices[replication] = {}
            for color in ['blue','green']:
                choices[replication][color] = {}
                for stimulus_id in range(len(self.stimuli)):
                    choices[replication][color][stimulus_id] = {}
                    for participant_index in range(8):
                        green_bias = self.true_biases[condition][replication][color][participant_index]
                        observations = -1 if condition == 'asocial' else self.get_social_observations(participant_index,condition,replication,color,stimulus_id)
                        choice = self.simulate_choice(stimulus_id,green_bias,observations,decision_parameters,decision_type)
                        choices[replication][color][stimulus_id][participant_index] = choice
        self.choices[condition] = choices
    
    def get_q(self,normalized_bias,difficulty,choice):
        logit_accumulator = normalized_bias - difficulty
        prob_green = 1 / (1 + math.exp(-logit_accumulator))
        if choice == 1:
            return prob_green * normalized_bias
        return (1 - prob_green) * normalized_bias

    def get_p(self,difficulty,choice):
        logit_accumulator = - difficulty
        prob_green = 1 / (1 + math.exp(-logit_accumulator))
        if choice == 1:
            return prob_green
        return 1 - prob_green

    def get_weight_for_choice(self,replication,color,stimulus,participant_index):
        difficulty = self.estimated_difficulties[stimulus]
        # Print all inputs to the function
        bias = self.estimated_biases['asocial'][replication][color][participant_index]
        choice = self.choices['asocial'][replication][color][stimulus][participant_index]
        p = self.get_p(difficulty,choice)
        q = self.get_q(bias,difficulty,choice)
        return p / q

    def get_weights_for_choice(self,participant_id,condition,replication,color,stimulus,oversampled_choice_indices):
        weights = []
        for observed_index in oversampled_choice_indices:
            weight = self.get_weight_for_choice(replication,color,stimulus,observed_index)
            weights.append(weight)
        # Normalize the weights
        weights = [weight / sum(weights) for weight in weights]
        return weights


    def resample_choices(self,participant_id,condition,replication,color,stimulus,oversampled_choice_indices):
        weights = self.get_weights_for_choice(participant_id,condition,replication,color,stimulus,oversampled_choice_indices)
        # Sample 8 choices with replacement from observations using the weights
        choice_indices = random.choices(oversampled_choice_indices,weights,k=8)
        return choice_indices

    def oversample_choices(self,replication,color,stimulus):
        incoming_biases = [self.estimated_biases['asocial'][replication][color][node] for node in range(8)]
        # Assign weights to each node proportional to magnitude of bias
        weights = [abs(bias) for bias in incoming_biases]
        # Normalize the weights
        weights = [weight / sum(weights) for weight in weights]
        # Sample 8 choices with replacement from observations using the weights. Just get the indices, not the actual choices
        index_vector = list(range(len(incoming_biases)))
        choice_indices = random.choices(index_vector,weights,k=8)
        return choice_indices

    def get_social_observations(self,participant_index,condition,replication,color,stimulus):
        choice_indices = self.oversample_choices(replication,color,stimulus)
        if (condition == 'resampling'):
            choice_indices = self.resample_choices(participant_index,condition,replication,color,stimulus,choice_indices)
        observed_choices = [self.choices['asocial'][replication][color][stimulus][node] for node in choice_indices]
        observed_choices_sum = sum(observed_choices)
        self.social_observations.setdefault(condition, {}).setdefault(replication, {}).setdefault(color, {}).setdefault(stimulus, {})[participant_index] = observed_choices_sum
        return observed_choices_sum

    def get_correct_choice(self, stimulus):
        green_proportion = float(self.stimuli[stimulus])
        if green_proportion > 0.5:
            return 'green'
        return 'blue'

    def irt(self):

        def model():
            with numpyro.plate('participants', self.num_participants_per_condition):
                b = numpyro.sample("b", dist.Normal(0, 3))

            with numpyro.plate('problems', len(self.stimuli)):
                d = numpyro.sample("d", dist.Normal(0, 3))

            choices_df = self.choices_to_df(True)
            
            with numpyro.plate("data", self.num_participants_per_condition * len(self.stimuli)):
                p_idx = choices_df['participant_id'].values
                s_idx = choices_df['stimulus'].values
                binary_decisions = choices_df['chose_green'].values
                linear_combinations = (b[p_idx] - d[s_idx])
                numpyro.sample('obs', dist.Bernoulli(logits=linear_combinations), obs=binary_decisions)

        def format_irt_posterior_samples(samples):
            bias = {}
            difficulties = {}
            for participant_id in range(self.num_participants_per_condition):
                bias[participant_id] = float(np.mean(samples['b'][:,participant_id]))
            for stimulus in self.stimuli:
                difficulties[stimulus] = float(np.mean(samples['d'][:,stimulus]))
            return {'b': bias, 'd':difficulties}

        def save_posterior_estimates(samples):
            for participant_id in samples['b']:
                # condition, replication, color, participant. All conditions are asocial
                _, replication, color, participant_index = self.get_participant_details(participant_id)
                self.estimated_biases.setdefault('asocial', {}).setdefault(replication, {}).setdefault(color, {})[participant_index] = samples['b'][participant_id]
            # Save difficulties to self.estimated_difficulties
            for stimulus in samples['d']:
                self.estimated_difficulties[stimulus] = samples['d'][stimulus]

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=1250, num_samples=2500, num_chains = 8, progress_bar=False)
        rng_key = jaxrandom.PRNGKey(0)
        mcmc.run(rng_key)
        samples = mcmc.get_samples()
        formatted_samples = format_irt_posterior_samples(samples)
        save_posterior_estimates(formatted_samples)

    def choices_to_df(self,asocial_only):
        # Converts the choices dict to a pandas dataframe. Column names: stimulus, color, participant_id, chose_green, chose_correct, iteration
        columns = ['stimulus','color','participant_id','chose_green','chose_correct', 'green_proportion','k_chose_green','chose_bias','condition','replication','true_bias','estimated_bias']
        data = []  # Create an empty list to store rows
        conditions = ['asocial'] if asocial_only else self.conditions
        participant_id = 0
        for condition in conditions:
            for replication in range(self.num_replications):
                for color in ['blue','green']:
                    for stimulus in self.choices[condition][replication][color]:
                        for participant_index in range(8): 
                            chose_green = self.choices[condition][replication][color][stimulus][participant_index]
                            choice = 'green' if chose_green == 1 else 'blue'
                            correct_choice = self.get_correct_choice(stimulus)
                            chose_correct = 1 if choice == correct_choice else 0
                            k_chose_green = -1 if condition == 'asocial' else self.social_observations[condition][replication][color][stimulus][participant_index]
                            chose_bias = choice == color
                            participant_id = self.get_participant_id(condition, replication, color, participant_index)
                            true_bias = self.true_biases[condition][replication][color][participant_index]
                            estimated_bias = self.estimated_biases['asocial'][replication][color][participant_index] if condition == 'asocial' and asocial_only == False else 0
                            row = [stimulus, color, participant_id, chose_green, chose_correct, self.stimuli[stimulus], k_chose_green,chose_bias,condition,replication,true_bias,estimated_bias]
                            data.append(row)
        df = pd.DataFrame(data,columns=columns)
        return df

    def simulate_experiment(self):
        # Initialize participants
        self.initialize_participants()
        # Simulate asocial generation
        self.sample_choices('asocial')
        # Fit IRT model
        self.irt()
        # Simulate oversampling condition
        self.sample_choices('oversampling')
        # # Simulate resampling condition
        self.sample_choices('resampling')

# -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- 
# SCRIPT STARTS HERE
# -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- 

# Setup
replications_vector = [20,25,30,35]
num_simulations = 100

# Loop through 1:num_simulations and create a network for each
for num_replications in replications_vector:
    # Initialize dfs
    dfs = {}
    for i in range(num_simulations):
        print(f'Running simulation {i+1} of {num_simulations} for {num_replications} replications')
        simulator = ExperimentSimulator(num_replications)
        simulator.simulate_experiment()
        # Convert to dataframe
        df_result = simulator.choices_to_df(False)
        # Add simulation number as column
        df_result['simulation'] = i
        # Add to dfs
        dfs[i] = df_result

    # Concatenate all the dataframes
    df = pd.concat(dfs)

    # Group df by condition and randomization color and print mean of chose_green
    print(df.groupby(['condition','color'])['chose_green'].mean())
    print("-- -- -- -- -- --")
    print(df.groupby(['condition','color'])['k_chose_green'].mean()/8)

    directory = '../../data/e3_power_analysis/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + f'num_replications={num_replications}-num_simulations={num_simulations}.csv'
    df.to_csv(filename,index=False)


# Tuesday: Power analysis + send to Tom
# Wednesday: Launch experiment + network writeup
# Thursday: Continue expeirment + writing session
# Friday: Experiment writeup
# Saturday: Writing session
# Sunday: Writing session
# To Tom on Sunday night