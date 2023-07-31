### --- Script for running the power analysis of our e3 design --- ###
### --- Power analysis consists of all of our preregistered tests --- ###

library(dplyr)
library(stringr)
library(lme4)
library(ggplot2)

# -- -- -- -- -- --
# -- -- -- -- -- --
# Functions for running the power analysis
# -- -- -- -- -- --
# -- -- -- -- -- --

# Loop through number of problems and number of unique networks for each treatment
# Each combo defines a unique set of simulations
# For each combo, perform a power analysis
power_analysis = function(networks,num_simulations){
  all_df = data.frame()
  for (networks_i in networks){
    print(paste('---- Running analysis on',networks_i,'data ----'))
    outcome_df = run_tests(networks_i,num_simulations)
    all_df = rbind(all_df,outcome_df)
    }
  return(all_df)
}


# Clean the data from a given simulation
prepare_data_for_analysis = function(input_data){
  return(
    input_data %>%
      transmute(
        simulation,
        condition,
        condition_replication = replication,
        participant_id,
        chose_utility = as.logical(chose_bias),
        chose_correct = as.logical(chose_correct),
        is_social = condition != 'asocial',
        randomization_color = color,
        aso_soc_correlated = ifelse(condition == 'oversampling','asocial',condition),
        aso_soc_resampling = ifelse(condition == 'resampling','asocial',condition),
        soc_correlated_soc_resampling = ifelse(condition == 'resampling','oversampling',condition),
      ) %>%
      data.frame()
  )
}

# Make a dataframe of columns to append to the power analysis results
make_params_dataframe = function(num_networks,simulation_num){
  return(
    data.frame(
      num_networks = num_networks,
      simulation_num = simulation_num
    )
  )
}

# Loop through each simulation for a given num_problems and num_networks
# For each simulation, run our pre-registered tests
# Add identifying information (i.e., num_problems, num_networks)
run_tests = function(num_networks,num_simulations){
  sim_data_string = paste0('../../data/e3_power_analysis/num_replications=',num_networks,'-num_simulations=',num_simulations,'.csv')
  all_data = read.csv(sim_data_string) %>% prepare_data_for_analysis()
  power_df = data.frame()
  for (simulation_i in 0:(num_simulations - 1)){
    print(paste('Testing on simulation', simulation_i))
    # Generate filename - power_df_num_replications=num_networks-up_to=simulation_i.csv
    filename = paste0('power_df_num_replications=',num_networks,'-up_to=',simulation_i,'.csv')
    # If exists, return it
    if (file.exists(filename)){
      print(paste('File',filename,'already exists. Skipping.'))
      power_df = read.csv(filename)
      if ('X' %in% colnames(filename)){
        power_df = power_df %>% select(-X)
      }
      next
    }
    simulation_data = all_data %>% subset(simulation == simulation_i)
    outcomes = simulation_data %>%
      run_analysis_on_one_simuation(num_networks) %>%
      cbind(get_condition_means(simulation_data)) %>%
      cbind(make_params_dataframe(num_networks,simulation_i))
    # Bind
    power_df = rbind(power_df,outcomes)
    write.csv(power_df,filename,row.names = F)
  }
  return(power_df)
}

# Get bias and accuracy means for each treatment
get_condition_means = function(input_data){
  return(
    data.frame(
      'aso_bias' = mean(subset(input_data,condition == 'asocial')$chose_utility),
      'aso_accuracy' = mean(subset(input_data,condition == 'asocial')$chose_correct),
      'soc_correlated_bias' = mean(subset(input_data,condition == 'oversampling')$chose_utility),
      'soc_correlated_accuracy' = mean(subset(input_data,condition == 'oversampling')$chose_correct),
      'soc_resampling_bias' = mean(subset(input_data,condition == 'resampling')$chose_utility),
      'soc_resampling_accuracy' = mean(subset(input_data,condition == 'resampling')$chose_correct)
    )
  )
}

# Make glmer and glm models for hypothesis tests
# If the model is singular, it keeps simplifying until the model converges
# First it tries the full pre-registered model with all random intercepts
# If it's singular, it removes participant random intercepts
# If it's singular, it adds participant intercepts back in and removes network intercepts
# If it's singular, it uses a normal logistic regression
# Both restricted and unrestricted models for the hypothesis test have to
# be non-singular for the model to pass the check
make_models = function(sim_data,condition_variable,dependent_variable){
  model_strings = c('+ (1|participant_id) + (0 + is_social | condition_replication)',
                    '+ (0 + is_social | condition_replication)',
                    '+ (1|participant_id)',
                    '')
  unrestricted_model_strings = paste(dependent_variable,'~ condition + randomization_color',model_strings)
  restricted_model_strings = paste(dependent_variable,'~ randomization_color +',condition_variable,model_strings)
  model_singular = T
  num_iterations = 0
  while (model_singular){
    num_iterations  = num_iterations + 1
    unrestricted_model_string = unrestricted_model_strings[num_iterations]
    restricted_model_string = restricted_model_strings[num_iterations]
    if (grepl('\\|',unrestricted_model_string)){
      unrestricted_model = glmer(as.formula(unrestricted_model_string),data=sim_data,family='binomial')
      restricted_model = glmer(as.formula(restricted_model_string),data=sim_data,family='binomial')
      model_singular = (isSingular(unrestricted_model) || isSingular(restricted_model))
    } else {
      unrestricted_model = glm(as.formula(unrestricted_model_string),data=sim_data,family='binomial')
      restricted_model = glm(as.formula(restricted_model_string),data=sim_data,family='binomial')
      model_singular = F
    }
  }
  return(list(unrestricted_model,restricted_model,num_iterations))
}
  
  
  # model_strings = '+ (1|participant_id) + (0 + is_social | condition_replication)'
  # unrestricted_model_string = paste(dependent_variable,'~ condition + randomization_color',model_strings)
  # restricted_model_string = paste(dependent_variable,'~ randomization_color +',condition_variable,model_strings)
  # unrestricted_model = glmer(as.formula(unrestricted_model_string),data=sim_data,family='binomial')
  # restricted_model = glmer(as.formula(restricted_model_string),data=sim_data,family='binomial')
  # return(list(unrestricted_model,restricted_model))


# Function to get "coefficient as expected" for a given hypothesis test
coefficient_as_expected = function(data,hypothesis){
  if (hypothesis=='h1'){
    as_expected = mean(subset(data,condition=='oversampling')$chose_utility) - mean(subset(data,condition=='asocial')$chose_utility) > 0
  } else if (hypothesis=='h2'){
    as_expected = mean(subset(data,condition=='oversampling')$chose_utility) - mean(subset(data,condition=='resampling')$chose_utility) > 0
  } else if (hypothesis=='h3'){
    as_expected = mean(subset(data,condition=='resampling')$chose_correct) - mean(subset(data,condition=='asocial')$chose_correct) > 0
  } else if (hypothesis=='h4'){
    as_expected = mean(subset(data,condition=='resampling')$chose_correct) - mean(subset(data,condition=='oversampling')$chose_correct) >= 0
  }
  return(as_expected)
}

# Runs a likelihood ratio test on a model fit using the function above
# with separate variables for each treatment, and ones where the treatments
# are defined by the condition variable (this will have two treatments coded as one)
run_hypothesis_test = function(sim_data,dependent_variable,condition_variable,hypothesis_number){
  current_models = make_models(sim_data,condition_variable,dependent_variable)
  current_anova = anova(current_models[[1]],current_models[[2]],test='Chisq')
  anova_p_value = if (current_models[[3]] == 4) current_anova[[5]][2] else current_anova[2,'Pr(>Chisq)']
  coeff_as_expected = coefficient_as_expected(sim_data,hypothesis_number)
  if (hypothesis_number == 'h4'){
    current_df = data.frame(as_expected = (coeff_as_expected || anova_p_value > 0.05))
  } else {
    current_df = data.frame(as_expected = (coeff_as_expected && anova_p_value <= 0.05))
  }
  current_df = cbind(current_df,data.frame(p_value = anova_p_value))
  colnames(current_df) = paste(hypothesis_number,colnames(current_df),sep="_")
  return(current_df)
}

# Runs each test outlined in our preregistration
# For each, it makes a new column based on the relevant test that is the same as 
# the condition column, but the two conditions of interest are in the same group
# It then uses this column to make the restricted condition for the likelihood ratio tests
run_analysis_on_one_simuation = function(simulation_data,num_networks){
  # H1: Bias higher in social-correlated vs. asocial
  h1 = run_hypothesis_test(simulation_data, 'chose_utility', 'aso_soc_correlated','h1')
  
  # H2: Bias lower in social-resampling vs. social-correlated
  h2 = run_hypothesis_test(simulation_data, 'chose_utility', 'soc_correlated_soc_resampling','h2')

  # # H3: Accuracy higher in social-resampling vs. asocial
  # h3 = run_hypothesis_test(simulation_data, 'chose_correct', 'aso_soc_resampling','h3')

  # H4: Accuracy NOT higher in social-resampling vs. social-correlated
 h4 = run_hypothesis_test(simulation_data, 'chose_correct', 'soc_correlated_soc_resampling','h4')
  
  # Add a column T/F to whether all hypotheses were as expected and return
  all_df = cbind(h1,h2,h4)
  all_df$all_expected = rowSums(all_df[grepl("as_expected",colnames(all_df))]) == 3
  return(all_df)
}


# -- -- -- -- -- --
# -- -- -- -- -- --
# Run power analysis
# -- -- -- -- -- --
# -- -- -- -- -- --

networks = c(20)
num_simulations = 100
power_data = data.frame()
power_data = power_analysis(networks,num_simulations)

power_data %>%
  group_by(num_networks) %>%
  summarize(power = mean(all_expected)) %>%
  mutate( num_participants = num_networks * 2 * 8 * 3) %>%
  ggplot(aes(x=num_participants,y=power)) +
  geom_point() +
  geom_line() +
  ggtitle('Estimated power: No H3')



# d = read.csv(sim_data_string)
# nrow(d)/(960 * 100)


