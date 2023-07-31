### --- Script for running the power analysis of our e2 design --- ###
### --- Power analysis consists of all of our preregistered tests --- ###
### --- Prereg link available at https://osf.io/87me6/ --- ###

library(dplyr)
library(stringr)
library(lme4)

source("utils.R")

# -- -- -- -- -- --
# -- -- -- -- -- --
# Functions for running the power analysis
# -- -- -- -- -- --
# -- -- -- -- -- --

# Loop through number of problems and number of unique networks for each treatment
# Each combo defines a unique set of simulations
# For each combo, perform a power analysis
power_analysis = function(problems,networks){
  first_iteration = T
  for (problems_i in problems){ 
    for (networks_i in networks){
      print(paste('---- Running analysis of',problems_i,networks_i,'data ----'))
      outcome_df = run_tests(problems_i,networks_i)
      all_df = if (first_iteration) outcome_df else rbind(all_df,outcome_df)
      first_iteration = F
    }
  }
  return(all_df)
}

# Clean the data from a given simulation
prepare_data_for_analysis = function(input_data){
  return(
    input_data %>%
      filter(condition != 'ASO:W-U' | resampling_type != 'simple') %>% # Only keep one of the cloned sets
      transmute(
        condition_replication,
        participant_id,
        chose_utility = as.logical(chose_utility),
        chose_correct = as.logical(chose_correct),
        is_social = as.integer(condition == 'ASO:W-U'),
        true_condition = ifelse(condition == 'ASO:W-U','ASO:W-U',paste(condition,resampling_type,sep=':')),
        aso_soc_identity_condition = ifelse(true_condition == 'SOC:W-U:identity','ASO:W-U',true_condition),
        aso_soc_resampling_condition = ifelse(true_condition == 'SOC:W-U:simple','ASO:W-U',true_condition),
        soc_identity_soc_resampling_condition = ifelse(true_condition == 'SOC:W-U:simple','SOC:W-U:identity',true_condition),
        randomization_color = ifelse(randomization_green == 1, 'green','blue')
      ) %>%
      data.frame()
  )
}

# Get the filepath for a simulation
get_sim_data_string = function(num_problems,num_networks,simulation_num){
  return(
    paste0(
      '../data/simulated_data/numProblems=',num_problems,
      '_numNetworks=',num_networks,
      '_iterationCount=',simulation_num,
      '.csv')
    )
}

# Make a dataframe of columns to append to the power analysis results
make_params_dataframe = function(num_problems,num_networks,simulation_num){
  return(
    data.frame(
      num_problems = num_problems,
      num_networks = num_networks,
      simulation_num = simulation_num
    )
  )
}

# Loop through each simulation for a given num_problems and num_networks
# For each simulation, run our pre-registered tests
# Add identifying information (i.e., num_problems, num_networks)
run_tests = function(num_problems,num_networks){
  power_df = data.frame()
  for (simulation_i in 0:99){
    print(paste('Testing on simulation', simulation_i))
    # Load data and do analysis
    sim_data = get_sim_data_string(num_problems,num_networks,simulation_i) %>% 
      read.csv() %>% 
      prepare_data_for_analysis()
    outcomes = sim_data %>% 
      run_analysis_on_one_simuation(num_problems,num_networks,simulation_i) %>%
      cbind(get_condition_means(sim_data)) %>%
      cbind(make_params_dataframe(num_problems,num_networks,simulation_i))
    # Bind
    power_df = rbind(power_df,outcomes)
  }
  return(power_df)
}

# Get bias and accuracy means for each treatment
get_condition_means = function(input_data){
  return(
    data.frame(
      'aso_wu_bias' = mean(subset(input_data,true_condition == 'ASO:W-U')$chose_utility),
      'aso_wu_accuracy' = mean(subset(input_data,true_condition == 'ASO:W-U')$chose_correct),
      'soc_wu_identity_bias' = mean(subset(input_data,true_condition == 'SOC:W-U:identity')$chose_utility),
      'soc_wu_identity_accuracy' = mean(subset(input_data,true_condition == 'SOC:W-U:identity')$chose_correct),
      'soc_wu_resampling_bias' = mean(subset(input_data,true_condition == 'SOC:W-U:simple')$chose_utility),
      'soc_wu_resampling_accuracy' = mean(subset(input_data,true_condition == 'SOC:W-U:simple')$chose_correct)
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
  unrestricted_model_strings = paste(dependent_variable,'~ true_condition + randomization_color',model_strings)
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

# Runs a likelihood ratio test on a model fit using the function above
# with separate variables for each treatment, and ones where the treatments
# are defined by the condition variable (this will have two treatments coded as one)
run_hypothesis_test = function(sim_data,dependent_variable,condition_variable,hypothesis_number){
  current_models = make_models(sim_data,condition_variable,dependent_variable)
  current_anova = anova(current_models[[1]],current_models[[2]],test='Chisq')
  anova_p_value = if (current_models[[3]] == 4) current_anova[[5]][2] else current_anova[2,'Pr(>Chisq)']
  coefficient = coef(summary(current_models[[1]]))[,1][2][[1]]
  if (hypothesis_number == 'h5'){
    current_df = data.frame(as_expected = (coefficient < 0 || anova_p_value > 0.05))
  } else {
    current_df = data.frame(as_expected = (coefficient > 0 && anova_p_value <= 0.05))
  }
  current_df = cbind(current_df,data.frame(
    p_value = anova_p_value,
    coefficient = coefficient,
    model_convergence_iteration = current_models[[3]])
  )
  colnames(current_df) = paste(hypothesis_number,colnames(current_df),sep="_")
  return(current_df)
}

# Runs each test outlined in our preregistration
# For each, it makes a new column based on the relevant test that is the same as 
# the condition column, but the two conditions of interest are in the same group
# It then uses this column to make the restricted condition for the likelihood ratio tests
run_analysis_on_one_simuation = function(simulation_data,num_problems,num_networks,simulation_num){
  # H1: Bias higher in social-identity vs. asocial
  simulation_data$true_condition = factor(simulation_data$true_condition,levels = c('ASO:W-U','SOC:W-U:identity','SOC:W-U:simple'))
  h1 = run_hypothesis_test(simulation_data, 'chose_utility', 'aso_soc_identity_condition','h1')
  # H3: Accuracy higher in social-identity vs. asocial
  h3 = run_hypothesis_test(simulation_data,'chose_correct', 'aso_soc_identity_condition','h3')
  
  # H2: Bias higher in social-identity vs. social-resampling
  simulation_data$true_condition = factor(simulation_data$true_condition,levels = c('SOC:W-U:simple','SOC:W-U:identity','ASO:W-U'))
  h2 = run_hypothesis_test(simulation_data,'chose_utility', 'soc_identity_soc_resampling_condition','h2')
  # H5: Accuracy NOT higher in social-identity vs. social-resampling
  h5 = run_hypothesis_test(simulation_data,'chose_correct', 'soc_identity_soc_resampling_condition','h5')
  
  # H4: Accuracy higher in social-resampling vs. asocial
  simulation_data$true_condition = factor(simulation_data$true_condition,levels = c('ASO:W-U','SOC:W-U:simple','SOC:W-U:identity'))
  h4 = run_hypothesis_test(simulation_data,'chose_correct', 'aso_soc_resampling_condition','h4')
  
  # Add a column T/F to whether all hypotheses were as expected and return
  all_df = cbind(h1,h2,h3,h4,h5)
  all_df$all_expected = rowSums(all_df[grepl("as_expected",colnames(all_df))]) == 5
  all_df$all_models_converged = rowSums(all_df[grepl("model_convergence_iteration",colnames(all_df))]) == 1
  return(all_df)
}


# -- -- -- -- -- --
# -- -- -- -- -- --
# Run power analysis
# -- -- -- -- -- --
# -- -- -- -- -- --

networks = 14
problems = 16
power_data = power_analysis(problems,networks)

# Number of simulations out of 100 where every hypothesis test was as expected
num_as_expected = sum(power_data$all_expected)

# Write num_as_expected to file
output_directory = 'stats/power_analysis'
make_directory(output_directory)
output_path = paste(output_directory,'num_as_expected.tex',sep='/')
write(num_as_expected,output_path)
