### --- Script for running the pre-registered regressions --- ###

library(lme4)
library(dplyr)
source('../utils.R')

### --- --- --- --- --- --- --- --- --- ### 
### --- Variables for model fitting --- ###
### --- --- --- --- --- --- --- --- --- ### 

# Directory to write the results
results_directory = '../../paper/stats/e3'

# Dependent variables to investigate to 
dependent_variables = c('chose_bias','chose_correct')

# glmer or lmer, depending on type of dependent_variables
regression_type = 'glmer'

addition_for_prereg = '+ (0 + is_social | condition_replication)'

# The formula for the glmer that all (unrestricted, restricted, prereg, reported) models share
common_model_formula ='+ randomization_color+(1|participant_id)'

### --- --- --- --- --- --- --- --- --- --- --- --- ### 
### --- Run all comparisons between treatments --- ###
### --- --- --- --- --- --- --- --- --- --- --- --- ### 

# Load experiment 3 data
e3_data = load_e3_data()

# Run all comparisons between treatments 
# with both chose_bias and chose_correct as dependent variables on:
# 1. Reported models (no condition random intercepts)
# 2. Pre-registered models (w/ condition random intercepts for social participants)
comparisons = e3_data %>%
  compare_conditions(dependent_variables,regression_type,common_model_formula,addition_for_prereg)

# Write test statistics, p values, and condition means for direct use in the paper
write_tests(comparisons,results_directory,T)
print('-- -- -- Writing means -- -- --')
write_means(e3_data,results_directory,dependent_variables)

# Print condition means
print('-- -- -- Condition means -- -- --')
e3_data %>%
  get_condition_means()

# Print comparisons
print('-- -- -- Condition statistics -- -- --')
print(comparisons)

# Betas and CIs
e3_data %>%
  mutate(
    condition = case_when(
      condition == 'asocial' ~ 'asocial_bias',
      condition == 'oversampling' ~ 'social_oversampling',
      condition == 'resampling' ~ 'social_resampling'
    )
  ) %>%
  fit_and_report_betas_and_cis(common_model_formula,dependent_variables,'glmer',results_directory)


# mod = glmer(chose_bias ~ condition + randomization_color + (1|participant_id),data=e3_data,family)





