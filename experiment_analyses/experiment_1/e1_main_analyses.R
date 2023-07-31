### --- Script that runs all bias and accuracy comparisons for all treatments --- ###
### --- Pre-reg available at https://osf.io/yth5r/ --- ###

library(lme4)
library(dplyr)
source('utils.R')

### --- --- --- --- --- --- --- --- --- ### 
### --- Variables for model fitting --- ###
### --- --- --- --- --- --- --- --- --- ### 

# Directory to write the results if write_outcomes is TRUE
results_directory = 'stats/e1'

# Dependent variables to investigate
dependent_variables = c('chose_correct','chose_bias')

# glmer or lmer, depending on type of dependent_variables
regression_type = 'glmer'

# The formula for the glmer that all (unrestricted, restricted, prereg, reported) models share
common_model_formula = '+(1|condition_replication)'

# Addition to common_formula for pre reg models
addition_for_prereg = '+ (1 | participant_id)'      

### --- --- --- --- --- --- --- --- --- --- --- --- ### 
### --- Run all comparisons between treatments --- ###
### --- --- --- --- --- --- --- --- --- --- --- --- ### 

# Load experiment 1 data
e1_data = load_e1_data()

# Run all comparisons between treatments 
# with both chose_bias and chose_correct as dependent variables on:
# 1. Reported models (no participant random intercepts)
# 2. Pre-registered models (with participant random intercepts)
# These include condition random intercepts for social participants
comparisons = e1_data %>% 
  compare_conditions(dependent_variables,regression_type,common_model_formula,addition_for_prereg)

# Write test statistics, p values, and condition means for direct use in the paper
print('-- -- -- Writing statistical tests -- -- --')
write_tests(comparisons,results_directory,T)
print('-- -- -- Writing means -- -- --')
write_means(e1_data,results_directory,dependent_variables)

# Print condition means
print('-- -- -- Condition means -- -- --')
e1_data %>%
  get_condition_means()

# Print statistical test outcomes
print('-- -- -- Condition comparisons -- -- --')
print(comparisons)
