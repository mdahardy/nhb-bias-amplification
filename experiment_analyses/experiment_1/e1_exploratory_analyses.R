### --- Script for running the exploratory (non-preregistered) analyses for experiment 1 --- ###

library(lme4)
library(dplyr)
library(lubridate)
source('utils.R') # helper functions

### --- --- --- --- --- --- --- --- --- ###
### --- Variables for model fitting --- ###
### --- --- --- --- --- --- --- --- --- ###

# Directory to write the results (matched and nonmatched described below)
matched_directory = 'stats/e1_exploratory/matched'
nonmatched_directory = 'stats/e1_exploratory/nonmatched'

# Dependent variables to investigate
dependent_variables = c('chose_correct','chose_bias')

# glmer or lmer, depending on type of dependent_variables
regression_type = 'glmer'

# The formula for the glmer that all models share
common_model_formula = '+(1|condition_replication)'

# Addition to common_formula for pre reg models
addition_for_prereg = '+ (1 | participant_id)'

### --- --- RUN EXPLORATORY ANALYSES --- --- --- ###
# How can social-bias participants be both more accurate and more biased than asocial-control participants?
# Investigate models predicting accuracy when data is partitioned into trials
# where the true color matched their bias (matched trials) and didn't (nonmatched trials)
### --- --- --- --- --- --- --- --- --- --- --- --- ###

# Load experiment 1 data
e1_data = load_e1_data()
matched_data = subset(e1_data,matched==T)
nonmatched_data = subset(e1_data,matched==F)

# First get condition comparisons for matched trials
matched_comparisons = matched_data %>%
  compare_conditions(dependent_variables,regression_type,common_model_formula,addition_for_prereg) %>%
  mutate(matched=T)

# Now get condition comparisons for unmatched trials
nonmatched_comparisons = nonmatched_data %>%
  compare_conditions(dependent_variables,regression_type,common_model_formula,addition_for_prereg) %>%
  mutate(matched=F)

# Write test statistics, p values, and condition means for direct use in the paper
print('-- -- -- Writing statistical tests -- -- --')
write_tests(matched_comparisons,matched_directory,F)
write_tests(nonmatched_comparisons,nonmatched_directory,F)
print('-- -- -- Writing means -- -- --')
write_means(matched_data,matched_directory,dependent_variables)
write_means(nonmatched_data,nonmatched_directory,dependent_variables)

# Print condition means for both matched and non-matched trials
print('-- -- -- Means -- -- --')
e1_data %>%
  group_by(condition,matched) %>%
  summarise(accuracy = mean(chose_correct))

# Print comparisons
comparisons = rbind(matched_comparisons,nonmatched_comparisons)
print('-- -- -- Condition comparisons -- -- --')
print(comparisons)

# -- -- Get bonus and hourly wage -- --

# Load data with overflow participants
e1_earnings_data = '../../data/earnings_data/e1.csv' %>% 
  read.csv() %>%
  select(c('creation_time_participant','end_time','participant_id','is_overflow','generation','condition','is_practice','chose_correct')) %>%
  mutate(
    is_cloned = generation == 0 & grepl('SOC',condition,fixed=T)
  ) %>%
  subset(is_cloned==F)

# Need to adjust this to allow for cloned
ok_participants = e1_earnings_data %>%
  group_by(participant_id) %>%
  summarize(n=n()) %>%
  subset(n==10)

e1_earnings_data = subset(e1_earnings_data,participant_id %in% ok_participants$participant_id)

e1_earnings_data = e1_earnings_data %>%
  mutate(
    start = ymd_hms(creation_time_participant),
    end = ymd_hms(end_time),
    ms_spend = as.numeric(difftime(end, start, units = "secs") * 1000),
    minutes_spent = (ms_spend / (1000 * 60)),
    hours_spent = minutes_spent/60
  )

# Get worker earnings
participant_earnings = e1_earnings_data %>%
  subset(is_practice == 'False') %>%
  group_by(participant_id) %>%
  summarize(
    minutes_spent = unique(minutes_spent[1]),
    num_correct = sum(chose_correct == 'True'),
    bonus_in_points = (num_correct * 50) + 400,
    bonus_in_cents = bonus_in_points/10,
    total_points = (num_correct * 50) + 650 + 400,
    total_dollars = total_points/1000,
    dollars_per_hour = total_dollars / hours_spent
  )

# Get average of each column
participant_earnings %>%
  colMeans() %>%
  round(2) %>%
  print()
