### --- Script for fitting and investigating the bias IRT model to compare the individual bias
### of asocial- and social-bias participants

library(rstan)
library(dplyr)
library(rstanarm)
source('utils.R')

### --- --- --- --- --- --- ### 
### --- Helper functions --- ###
### --- --- --- --- --- --- ### 

recode_indices = function(input_column){
  unique_vals = unique(input_column)
  return(sapply(input_column,function(x) which(unique_vals==x)))
}

get_green_factor = function(proportion_green){
  if (proportion_green==0.48) return(1)
  if (proportion_green==0.49) return(2)
  if (proportion_green==0.51) return(3)
  if (proportion_green==0.52) return(4)
  throw('ERROR: Incorrect proprotion green')
}

fit_stan_model = function(full_data,is_social,num_iterations,num_chains){
  stan_data = list(
    N=nrow(full_data),
    chose_green = full_data$chose_green,
    num_participants = length(unique(full_data$participant_id)),
    num_problems = length(unique(full_data$net_decision_index)),
    participant_index = recode_indices(full_data$participant_id),
    green_factor = sapply(full_data$proportion_green,get_green_factor),
    negative_green = ifelse(full_data$randomization_color=='green',1,-1)
  )
  social_info = as.integer(full_data$social_info_green * 8) - 4
  if (is_social) stan_data$social_info = social_info
  base_path = 'experiment_1/bias_irt'
  model_path = if(is_social) paste0(base_path,'/social_bias_irt/model.stan') else paste0(base_path,'/asocial_bias_irt/model.stan')
  model_for_stan=stan_model(model_path)
  options(mc.cores=1)
  fitted_model = sampling(model_for_stan,stan_data,iter=num_iterations,chains=num_chains)
  return(fitted_model)
}

fit_model_and_save = function(e1_data,is_social,num_iterations,num_chains,refit_model){
  filepath= get_model_path(is_social,num_iterations,num_chains)
  social_str = if(is_social) 'SOCIAL' else 'ASOCIAL'
  if (!refit_model){
    if (file.exists(filepath)){
      print(paste('-- LOADING',social_str,'MODEL --'))
      return(readRDS(filepath))
    }
  }
  print(paste('-- FITTING',social_str,'MODEL --'))
  condition_str = if(is_social) 'social_bias' else 'asocial_bias'
  fitted_model = e1_data %>%
    subset(condition==condition_str) %>% 
    fit_stan_model(is_social,num_iterations,num_chains)
  saveRDS(fitted_model,file=filepath)
}

get_model_path = function(is_social,num_iterations,num_chains){
  directory = if (is_social) 'experiment_1/bias_irt/social_bias_irt' else 'experiment_1/bias_irt/asocial_bias_irt'
  make_directory(directory)
  filename = paste0('iterations=',num_iterations,'_chains=',num_chains,'.rds')
  return(paste(directory, filename,sep='/'))
}

### --- --- --- --- --- --- ### 
### --- Load data and setup--- ###
### --- --- --- --- --- --- ### 

# For fitting the stan models
num_iterations = 2500
num_chains = 8

#T/F if should refit model that already exists (with same num_iterations and num_chains)
refit_models = F

# Load experiment 1 data
e1_data = load_e1_data()

# Fit and save asocial-bias model
fit_model_and_save(e1_data,F,num_iterations,num_chains,refit_models)

# Fit and save social-bias model
fit_model_and_save(e1_data,T,num_iterations,num_chains,refit_models)

### --- --- --- --- --- --- ### 
### --- Write model statistics for use in table --- ###
### --- --- --- --- --- --- ###

# Asocial
asocial_model_path = get_model_path(F,num_iterations,num_chains)
asocial_stats_path = 'stats/e1_bias_irt/asocial'
make_directory(asocial_stats_path)
write_stan_model(asocial_model_path,asocial_stats_path)

# Asocial
social_model_path = get_model_path(T,num_iterations,num_chains)
social_stats_path = 'stats/e1_bias_irt/social'
make_directory(social_stats_path)
write_stan_model(social_model_path,social_stats_path)