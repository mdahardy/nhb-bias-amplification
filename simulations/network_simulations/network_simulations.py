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

# def generate_uniform_random_network(num_nodes, in_edges):
#     graph = nx.DiGraph()

#     # Add nodes to the graph
#     for i in range(num_nodes):
#         graph.add_node(i)

#     # Add incoming edges to each node
#     for i in range(num_nodes):
#         incoming_nodes = random.sample(range(num_nodes), in_edges + 1)  # Sample one extra node to handle self-loops
#         if i in incoming_nodes:
#             incoming_nodes.remove(i)  # Remove self-loop if present
#         else:
#             incoming_nodes.pop()  # Remove the extra node if not needed

#         for j in incoming_nodes:
#             graph.add_edge(j, i)

#     return graph

def generate_random_network(num_nodes, in_edges, inequality_factor, uniform=False):
    graph = nx.DiGraph()

    # Add nodes to the graph
    for i in range(num_nodes):
        graph.add_node(i)

    if uniform:
        # Use a uniform distribution for outgoing edges
        probabilities = np.ones(num_nodes) / num_nodes
    else:
        # Generate probabilities for the number of outgoing edges for each node using a power-law distribution
        probabilities = np.random.power(inequality_factor, num_nodes)
        probabilities /= probabilities.sum()

    # Add incoming edges to each node
    for i in range(num_nodes):
        available_nodes = list(range(num_nodes))
        available_nodes.remove(i)  # Remove the current node to avoid self-loops

        # Calculate the probabilities of the available nodes
        avail_probs = probabilities[available_nodes]
        avail_probs /= avail_probs.sum()

        for _ in range(in_edges):
            # Sample a node based on the renormalized probability distribution
            sampled_node = np.random.choice(available_nodes, p=avail_probs)

            # Add the edge to the graph
            graph.add_edge(sampled_node, i)

            # Remove the sampled node from the available nodes and renormalize probabilities
            available_nodes.remove(sampled_node)
            avail_probs = probabilities[available_nodes]
            avail_probs /= avail_probs.sum()

    return graph

def print_incoming_edges(graph):
    for node in graph.nodes():
        in_degree = graph.in_degree(node)
        print(f"Node {node}: {in_degree} incoming edges")

def print_outgoing_edges_sorted(graph):
    outgoing_edges = [(node, graph.out_degree(node)) for node in graph.nodes()]
    sorted_outgoing_edges = sorted(outgoing_edges, key=lambda x: x[1], reverse=True)
    for node, out_degree in sorted_outgoing_edges:
        print(f"Node {node}: {out_degree} outgoing edges")


# graph = generate_random_network(100, 8,10,uniform=False)
# print('-- INCOMING EDGES --')
# print_incoming_edges(graph)
# print('-- OUTGOING EDGES --')
# print_outgoing_edges_sorted(graph)
# stop()

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

class NetworkSimulation:
    def __init__(self, networks, num_participants,num_iterations,alpha,resampling,simulation_num):
        self.num_replications = num_replications
        self.num_participants = num_participants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.generated_networks = networks
        self.green_biases = {'blue':{},'green':{}}
        # One for each iteration
        self.estimated_biases = {}
        # One for each iteration
        self.estimated_difficulties = {}
        self.stimuli = {
            0: '0.48',
            1: '0.48',
            2: '0.48',
            3: '0.48',
            4: '0.49',
            5: '0.49',
            6: '0.49',
            7: '0.49',
            8: '0.51',
            9: '0.51',
            10: '0.51',
            11: '0.51',
            12: '0.52',
            13: '0.52',
            14: '0.52',
            15: '0.52',
        }
        self.choices = {}
        self.social_observations = {}
        self.resampling = resampling
        self.simulation_num = simulation_num

    def get_participant_id(self,participant_index,color):
        if color == 'green':
            return participant_index + self.num_participants
        return participant_index

    def get_participant_index_and_color(self,participant_id):
        if participant_id >= self.num_participants:
            return participant_id - self.num_participants, 'green'
        return participant_id, 'blue'

    def initialize_participants(self):
        for color in ['blue','green']:
            for i in range(self.num_participants):
                raw_bias = random.normalvariate(ORACLE_PARAMETERS['asocial']['absolute_bias'],ORACLE_PARAMETERS['asocial']['bias_sd'])
                self.green_biases[color][i] = raw_bias if color == 'green' else raw_bias * -1

    def simulate_choice(self,stimulus_id,green_bias,observations,params,decision_type):
        logit_accumulator = params['intercept']
        logit_accumulator += params['green_dummies'][self.stimuli[stimulus_id]]
        logit_accumulator += green_bias
        if decision_type == 'social':
            logit_accumulator += params['vote_coefficient'] * observations
        # Probability of choosing green
        prob_green = 1 / (1 + math.exp(-logit_accumulator))
        # Sample choice
        choice = random.random()
        if choice < prob_green:
            return 1
        else:
            return 0


    def sample_choices(self,iteration):
        decision_type = 'asocial' if iteration == 0 else 'social'
        decision_parameters = ORACLE_PARAMETERS[decision_type]
        # Loop over keys of stimuli dict and stimulate choice for each stimulus
        choices = {}
        for stimulus_id in range(len(self.stimuli)):
            stimulus = self.stimuli[stimulus_id]
            choices[stimulus_id] = {}
            for color in ['blue','green']:
                choices[stimulus_id][color] = {}
                for participant_id in range(self.num_participants):
                    green_bias = self.green_biases[color][participant_id]
                    observations = 0 if iteration == 0 else self.get_social_observations(participant_id,iteration,color,stimulus_id)
                    choice = self.simulate_choice(stimulus_id,green_bias,observations,decision_parameters,decision_type)
                    choices[stimulus_id][color][participant_id] = choice
        self.choices[iteration] = choices

    def get_irt_choice_probability(self,bias,difficulty,choice):
        logit_accumulator = bias - difficulty
        prob_green = 1 / (1 + math.exp(-logit_accumulator))
        if choice == 1:
            return prob_green
        return 1 - prob_green

    def get_weight_for_choice(self,observed_participant_index,stimulus,choice,bias,iteration,color):
        difficulty = self.estimated_difficulties[iteration][stimulus]
        bias = self.estimated_biases[iteration][color][observed_participant_index]
        p = self.get_irt_choice_probability(0,difficulty,choice)
        q = self.get_irt_choice_probability(bias,difficulty,choice)
        return p / q

    def get_weights_for_choice(self,participant_id,iteration,stimulus,original_observations,participant_ids_of_observations,color):
        weights = []
        for index, observation in enumerate(original_observations):
            bias = self.green_biases[color][participant_ids_of_observations[observation]]
            weight = self.get_weight_for_choice(participant_ids_of_observations[index],stimulus,observation,bias,iteration-1,color)
            weights.append(weight)
        # Normalize the weights
        weights = [weight / sum(weights) for weight in weights]
        return weights


    def resample_choices(self,participant_id,iteration,stimulus,original_observations,participant_ids_of_observations,color):
        # Get the weights for the choices
        weights = self.get_weights_for_choice(participant_id,iteration,stimulus,original_observations,participant_ids_of_observations,color)
        # Sample 8 choices with replacement from observations using the weights
        choices = random.choices(original_observations,weights,k=8)
        return choices


    def get_social_observations(self,participant_id,iteration,color,stimulus):
        network = self.generated_networks[color]
        # Get the indices of incoming nodes to current node
        incoming_nodes = list(network.predecessors(participant_id))
        # Get the choices of the incoming nodes
        incoming_choices = [self.choices[iteration-1][stimulus][color][node] for node in incoming_nodes]
        if (self.resampling):
            # Resample the choices
            incoming_choices = self.resample_choices(participant_id,iteration,stimulus,incoming_choices,incoming_nodes,color)

        # Add the incoming choices to the social observations
        if iteration not in self.social_observations:
            self.social_observations[iteration] = {}
        if stimulus not in self.social_observations[iteration]:
            self.social_observations[iteration][stimulus] = {}
        if color not in self.social_observations[iteration][stimulus]:
            self.social_observations[iteration][stimulus][color] = {}

        # Put observed choices into self.observations
        self.social_observations[iteration][stimulus][color][participant_id] = incoming_choices
        # Sum the choices
        return sum(incoming_choices)

    def get_correct_choice(self, stimulus):
        green_proportion = float(self.stimuli[stimulus])
        if green_proportion > 0.5:
            return 'green'
        return 'blue'

    def irt(self,iteration):

        def model():
            # Specify the priors
            total_num_participants = self.num_participants * 2
            with numpyro.plate('participants', total_num_participants):
                b = numpyro.sample("b", dist.Normal(0, 3))

            with numpyro.plate('problems', len(self.stimuli)):
                d = numpyro.sample("d", dist.Normal(0, 3))

            choices_df = self.choices_to_df(iteration)
            
            with numpyro.plate("data", total_num_participants * len(self.stimuli)):
                p_idx = choices_df['participant_id'].values
                s_idx = choices_df['stimulus'].values
                binary_decisions = choices_df['chose_green'].values
                linear_combinations = (b[p_idx] - d[s_idx])
                numpyro.sample('obs', dist.Bernoulli(logits=linear_combinations), obs=binary_decisions)

        def format_irt_posterior_samples(samples):
            # I want this to return a dict with keys:
            # 'b', with nested keys for each participant and the value being the mean of the posterior samples
            # and 'd', with nested keys for each stimulus and the value being the mean of the posterior samples
            bias = {}
            difficulties = {}
            for participant_id in range(self.num_participants * 2):
                bias[participant_id] = float(np.mean(samples['b'][:,participant_id]))
            for stimulus in self.stimuli:
                difficulties[stimulus] = float(np.mean(samples['d'][:,stimulus]))
            return {'b': bias, 'd':difficulties}

        def save_posterior_estimates(samples,iteration):
            # Save biases to self.estimated_biases
            self.estimated_biases[iteration] = {}
            for participant_id in samples['b']:
                participant_index,participant_color = self.get_participant_index_and_color(participant_id)
                if participant_color not in self.estimated_biases[iteration]:
                    self.estimated_biases[iteration][participant_color] = {}
                self.estimated_biases[iteration][participant_color][participant_index] = samples['b'][participant_id]
            # Save difficulties to self.estimated_difficulties
            self.estimated_difficulties[iteration] = {}
            for stimulus in samples['d']:
                self.estimated_difficulties[iteration][stimulus] = samples['d'][stimulus]

        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_warmup=1250, num_samples=2500, num_chains = 8, progress_bar=False)
        rng_key = jaxrandom.PRNGKey(0)
        mcmc.run(rng_key)
        samples = mcmc.get_samples()
        formatted_samples = format_irt_posterior_samples(samples)
        save_posterior_estimates(formatted_samples,iteration)

    def choices_to_df(self,iteration='all'):
        # Converts the choices dict to a pandas dataframe. Column names: stimulus, color, participant_id, chose_green, chose_correct, iteration
        columns = ['stimulus','color','participant_id','chose_green','chose_correct','iteration', 'green_proportion','resampling','k_chose_green','chose_bias','simulation_num']
        data = []  # Create an empty list to store rows
        iterations = range(self.num_iterations) if iteration == 'all' else [iteration]
        for iteration in iterations:
            for stimulus in self.choices[iteration]:
                for color in self.choices[iteration][stimulus]:
                    for participant_index in self.choices[iteration][stimulus][color]: 
                        chose_green = self.choices[iteration][stimulus][color][participant_index]
                        # get chose_correct
                        choice = 'green' if chose_green == 1 else 'blue'
                        correct_choice = self.get_correct_choice(stimulus)
                        chose_correct = 1 if choice == correct_choice else 0
                        k_chose_green = 0 if iteration == 0 else sum(self.social_observations[iteration][stimulus][color][participant_index])
                        chose_bias = choice == color
                        participant_id = self.get_participant_id(participant_index, color)
                        # Append to df using concat
                        row = [stimulus, color, participant_id, chose_green, chose_correct, iteration, self.stimuli[stimulus], self.resampling, k_chose_green,chose_bias,self.simulation_num]
                        data.append(row)
        df = pd.DataFrame(data,columns=columns)
        return df

    def save_graph(self):
        # Loop through keys of self.generated_networks (blue/green)
        for color in self.generated_networks:
            network = self.generated_networks[color]
            # Filename: network_size={num_participants}-alpha={alpha}-resampling={resampling}-simulation_num={simulation_num}.graphml
            filename = f'network_size={self.num_participants}-alpha={self.alpha}-resampling={self.resampling}-simulation_num={self.simulation_num}.graphml'
            directory = '../../data/network_simulations/graphs/{}/'.format(color)
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = directory + filename
            nx.write_graphml(network, filename)

    def run_social_simulations(self):
        for i in range(1, self.num_iterations):
            # self.generate_networks(i)
            if self.resampling:
                self.irt(i-1)
            self.sample_choices(i)

        

# -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- 
# PARAMETERS
# -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- 

# Setup
numpyro.set_host_device_count(8)
if len(sys.argv) < 5:
    print("Usage: python network_simulations.py <network_size> <alpha>")
    sys.exit(1)
num_participants_in_network = int(sys.argv[1])
alpha = float(sys.argv[2])
num_replications = int(sys.argv[3])
num_iterations = int(sys.argv[4])

plot_results = False
save_results = True

# Initialize dataframe to store results
dfs = {}

# Loop through 1:num_replications and create a network for each
for i in range(num_replications):
    print(f'Running simulation {i+1} of {num_replications}')
    # Generate network
    green_network = generate_random_network(num_participants_in_network,8,alpha)
    blue_network = generate_random_network(num_participants_in_network,8,alpha)
    networks = {'green':green_network,'blue':blue_network}
    # initial networks
    network_no_resampling = NetworkSimulation(networks,num_participants_in_network,num_iterations,alpha,False, i)
    # Initialize participants
    network_no_resampling.initialize_participants()
    # Sample initial choices
    network_no_resampling.sample_choices(0)
    # Copy network_no_resampling to network_with_resampling
    network_with_resampling = copy.deepcopy(network_no_resampling)
    # Change resampling to True
    network_with_resampling.resampling = True
    # Simulate social observations
    network_no_resampling.run_social_simulations()
    network_with_resampling.run_social_simulations()
    # Convert to dfs
    no_resampling_df = network_no_resampling.choices_to_df()
    with_resampling_df = network_with_resampling.choices_to_df()
    # Add to dfs with i-resampling and i-control
    dfs[f'{i}-control'] = no_resampling_df
    dfs[f'{i}-resampling'] = with_resampling_df
    # Save graphs
    if save_results:
        network_no_resampling.save_graph()

# Concatenate all the dataframes
df = pd.concat(dfs)

if save_results:
    directory = '../../data/network_simulations/results/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Filename: network_size={}-alpha={}-num_replications={}.csv
    filename = directory + f'network_size={num_participants_in_network}-alpha={alpha}-num_replications={num_replications}.csv'
    df.to_csv(filename,index=False)

if (plot_results):
    # # Group df by iteration and color and get mean of the response column
    chose_green_grouped = df.groupby(['iteration','color','resampling'])['chose_green'].mean().reset_index()
    chose_correct_grouped = df.groupby(['iteration','color','resampling'])['chose_correct'].mean().reset_index()
    custom_palette = {'blue': 'blue', 'green': 'green'}
    
    # Plot the results. Make color of line determined by color, and line type by resampling
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    sns.lineplot(data=chose_green_grouped,x='iteration',y='chose_green',hue='color',style='resampling',palette=custom_palette,ax=ax[0])
    sns.lineplot(data=chose_correct_grouped,x='iteration',y='chose_correct',hue='color',style='resampling',palette=custom_palette,ax=ax[1])
    plt.show()

    # Plot the chose_bias results
    all_bias = df.groupby(['iteration','resampling'])['chose_bias'].mean().reset_index()
    all_correct = df.groupby(['iteration','resampling'])['chose_correct'].mean().reset_index()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    sns.lineplot(data=all_bias,x='iteration',y='chose_bias',hue='resampling',style='resampling',ax=ax[0])
    sns.lineplot(data=all_correct,x='iteration',y='chose_correct',hue='resampling',style='resampling',ax=ax[1])
    plt.show()

    # Plot k_chose_green by iteration and color, ignoring iteration 0
    k_chose_green_grouped = df[df['iteration'] > 0].groupby(['iteration','color','resampling'])['k_chose_green'].mean().reset_index()
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    sns.lineplot(data=k_chose_green_grouped,x='iteration',y='k_chose_green',hue='color',style='resampling',palette=custom_palette,ax=ax)
    plt.show()