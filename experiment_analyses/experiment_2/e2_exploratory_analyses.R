### --- Script for running the exploratory (non-preregistered) analyses for experiment 2 --- ###

library(lme4)
library(dplyr)
library(lubridate)
source('utils.R') # helper functions

### --- --- --- --- --- --- --- --- --- ### 
### --- Variables for model fitting --- ###
### --- --- --- --- --- --- --- --- --- ### 

# Directory to write the results
results_directory = 'stats/e2_exploratory'

# Dependent variables to investigate 
dependent_variables = c('social_info_accuracy','social_info_bias')

# glmer or lmer, depending on type of dependent_variables
regression_type = 'lmer'

# The formula for the glmer that all (unrestricted, restricted, prereg, reported) models share
common_model_formula = '+randomization_color+(1|condition_replication)'

# NO addition for pre-reg here
addition_for_prereg = ''

### --- --- RUN EXPLORATORY ANALYSES PART 1 --- --- --- ### 
# Investigate the accuracy and bias of observed social information in the social conditions
### --- --- --- --- --- --- --- --- --- --- --- --- ###   

# Load experiment 2 data
e2_social_data = load_e2_data() %>% subset(is_social==T)

# Get comparisons for social_info_accuracy and social_info_bias for both social conditions
comparisons = e2_social_data %>%
  compare_conditions(dependent_variables,regression_type,common_model_formula,addition_for_prereg)

# Write test statistics, p values, and condition means for direct use in the paper
print('-- -- -- Writing statistical tests -- -- --')
write_tests(comparisons,results_directory,F)
print('-- -- -- Writing means -- -- --')
write_means(e2_social_data,results_directory,dependent_variables)

# Print condition means
print('-- -- -- Condition means -- -- --')
e2_social_data %>%
  group_by(condition) %>%
  summarize(
    social_info_bias = mean(social_info_bias) %>% round(3),
    social_info_accuracy = mean(social_info_accuracy) %>% round(3))

# Print comparisons
print('-- -- -- Condition statistics -- -- --')
print(comparisons)

### --- --- RUN EXPLORATORY ANALYSES PART 2 --- --- --- ### 
# Investigate the proportion of green social information observed in both conditions
### --- --- --- --- --- --- --- --- --- --- --- --- ###   

# Get comparisons for social_info_accuracy and social_info_bias for both social conditions
blue_green_common_formula ='+(1|condition_replication)'

# Run analyses on blue networks and write
blue_social_data = subset(e2_social_data,randomization_color == 'blue')
blue_comparisons = blue_social_data %>%
  compare_conditions('social_info_green',regression_type,blue_green_common_formula,addition_for_prereg)
write_tests(blue_comparisons,results_directory,F,'blue_bias')

# Run analyses on green networks and write
green_social_data = subset(e2_social_data,randomization_color == 'green')
green_comparisons = green_social_data %>%
  compare_conditions('social_info_green',regression_type,blue_green_common_formula,addition_for_prereg)
write_tests(green_comparisons,results_directory,F,'green_bias')


# Get means and write
output_dir = paste(results_directory,'means/social_info_green',sep='/')
for (condition_i in unique(e2_social_data$condition)){
  write_blue_green_means(blue_social_data,'blue',condition_i,output_dir)
  write_blue_green_means(green_social_data,'green',condition_i,output_dir)
}


# Print blue green means
print('-- -- -- Blue/green condition means -- -- --')
e2_social_data %>%
  group_by(condition,randomization_color) %>%
  summarize(
    social_info_green = mean(social_info_green) %>% round(3))

# Print comparisons
print('-- -- -- Blue comparison statistics -- -- --')
print(blue_comparisons)
print('-- -- -- Green comparison statistics -- -- --')
print(green_comparisons)


# -- -- Get bonus and hourly wage -- --

# Load participant data
e2_earnings_data =  '../../data/earnings_data/e2_participant.csv' %>% 
  read.csv() %>%
  subset(status == 'approved') %>%
  select(c('creation_time','end_time','base_pay','bonus')) %>%
  mutate(
    start = ymd_hms(creation_time),
    end = ymd_hms(end_time),
    ms_spend = as.numeric(difftime(end, start, units = "secs") * 1000),
    minutes_spent = (ms_spend / (1000 * 60)),
    hours_spent = minutes_spent/60,
    total_dollars = base_pay + bonus,
    dollars_per_hour = total_dollars / hours_spent
  )

e2_earnings_data %>%
  select(c('bonus','total_dollars','dollars_per_hour','minutes_spent')) %>%
  colMeans() %>%
  round(2) %>%
  print()
