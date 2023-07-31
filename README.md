# Bias amplification in social networks

Pre-registrations, experiment demos, data, analyses, and simulations for <b>Experimental evidence for bias amplification in social networks and a mitigation framework</b> by [Matt Hardy](https://matthardy.org/), [Bill Thompson](https://billdthompson.github.io/), [Peaks Krafft](https://www.arts.ac.uk/creative-computing-institute/people/peaks-krafft), and [Tom Griffiths](https://cocosci.princeton.edu).

## Pre-registration links

* Experiment 1: https://osf.io/yth5r
* Experiment 2: https://osf.io/87me6
* Experiment 3: https://osf.io/8s7y2

## Experiment demo

Perform the experiment as they were given to participants at the link below. Note that the participant interface for the social-resampling condition in Experiments 2 and 3 were identical to the social-motivated condition.

[Experiment demo](https://bias-amplification.netlify.app)

## Experiment code

Code for all experiments is available in the experiments/ directory. Experiments were run using Dallinger. To test an experiment locally, navigate to the experiment directory, and then run: 

```
$ pip install -r requirements.txt
$ dallinger debug --verbose
```

You may want to reduce the sample sizes in the relevant `experiment.py` file (e.g., reducing `generation_size` and `planned_overflow`) as each recruitment request will open a new browser window when debugging. Also note that installing Dallinger can be buggy and there are often version conficts between required packages.  More documentation is given on their [website](https://dallinger.readthedocs.io/en/latest).

## Data

Data from all experiments can be found in the data/ directory:

* Experiment 1: [experiment_1.csv](data/experiment_1.csv)
* Experiment 2: [experiment_2.csv](data/experiment_2.csv)
* Experiment 3: [experiment_3.csv](data/experiment_3.csv)

Data for the power analyses for Experiments 2 and 3 along with data from the network simulations can be found in the data/simulated_data/ directory. Note that this data must but unzipped in order to be used in the corresponding scripts.

## Analyses

Scripts for running the experiment analyses are in the experiment_analyses/ directory. Simulation scripts for the power analyses and network simulations are in the simulations/ directory.