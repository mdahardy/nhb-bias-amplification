library(rstan)
library(dplyr)

load_experiment_data = function(){
  return(
    '../data/experiment_1.csv' %>%
      read.csv() %>%
      subset(failed=='f') %>% # Remove failed nodes
      subset(is_overflow=='False') %>% # Exclude overflow participants from analysis
      # Remove participant 2790 - person was an extra participant in SOC:W-U, generation 7, network replicaiton 2 (caused by backend error)
      subset(participant_id != 2790) %>%
      # Exclude practice trials
      subset(is_practice=='False') %>%
      transmute(
        condition = as.factor(as.character(condition)),
        is_cloned = generation == 0 & condition %in% c('Social no bias','Social bias'),
        participant_id = as.factor(participant_id),
        chose_correct = as.logical(chose_correct),
        participant_id = as.integer(participant_id),
        randomization_green = ifelse(randomization_color=='green',1,0),
        chose_green = chose_utility == randomization_green,
        proportion_green = ifelse(randomization_green,proportion_utility,1-proportion_utility),
        green_integers = sapply(proportion_green,add_green_integers)
      ) %>%
      subset(is_cloned == F) %>%
      select(
        generation,
        condition,
        randomization_green,
        participant_id,
        condition_replication,
        k_chose_utility,
        chose_green,
        chose_correct,
        proportion_green,
        chose_utility,
        green_integers
      ) 
  )
}

recode_indices = function(input_column){
  unique_vals = unique(input_column)
  return(sapply(input_column,function(x) which(unique_vals==x)))
}

fit_social_silver_model = function(full_data){
  model_for_stan=stan_model('stan_models/silver_models/social_model.stan')
  k_chose_green = ifelse(full_data$randomization_green==T,full_data$k_chose_utility,8-full_data$k_chose_utility)
  stan_data = list(
    N=nrow(full_data),
    chose_green = full_data$chose_green,
    num_participants=length(unique(full_data$participant_id)),
    num_problems = length(unique(full_data$new_net_decision_index)),
    participant_index = recode_indices(full_data$participant_id),
    problem_index =  recode_indices(full_data$new_net_decision_index),
    green_votes = k_chose_green)
  options(mc.cores=1)
  fitted_model = sampling(model_for_stan,stan_data,iter=2500,chains=8)
  return(fitted_model)
}

fit_asocial_silver_model = function(full_data){
  model_for_stan=stan_model('stan_models/silver_models/asocial_model.stan')
  stan_data = list(
    N=nrow(full_data),
    chose_green = full_data$chose_green,
    num_participants=length(unique(full_data$participant_id)),
    num_problems = length(unique(full_data$new_net_decision_index)),
    participant_index = recode_indices(full_data$participant_id),
    problem_index =  recode_indices(full_data$new_net_decision_index))
  options(mc.cores=1)
  fitted_model = sampling(model_for_stan,stan_data,iter=2500,chains=8)
  return(fitted_model)
}

add_weights = function(full_data,fitted_stan_model,num_problems){
  # Extract estimates and add to full_data
  samples = extract(fitted_stan_model)
  full_data$original_chose_green_probability = colMeans(samples$og_choice_probabilities)
  full_data$no_bias_chose_green_probability = colMeans(samples$bias_0_choice_probabilities)
  participant_biases = colMeans(samples$bias)
  full_data$inferred_bias = rep(participant_biases,each=num_problems)
  # Get weights and adjust to participants' chocies
  full_data$no_bias_choice_probability = ifelse(full_data$chose_green,full_data$no_bias_chose_green_probability,1-full_data$no_bias_chose_green_probability)
  full_data$og_choice_probability = ifelse(full_data$chose_green,full_data$original_chose_green_probability,1-full_data$original_chose_green_probability)
  full_data$weights = full_data$no_bias_choice_probability / full_data$og_choice_probability
  return(full_data)
}

get_transmitted_social_info = function(full_data){
  new_full_data = full_data %>% 
    group_by(new_net_decision_index,condition_replication,resampling_type) %>%
    mutate(
      resample_probability =  weights/sum(weights),
      new_chose_utility_probability =  sum(chose_utility * resample_probability),
      current_original_k_chose_utility = sum(chose_utility),
      # This case_when prevents annoying bug caused when 1's aren't rounded correctly and cause NA's in rbinom
      current_resampled_k_chose_utility = case_when(
        new_chose_utility_probability >= 1 ~ as.integer(8), 
        new_chose_utility_probability <= 0 ~ as.integer(0),
        TRUE ~ rbinom(size=8,n=1,prob=new_chose_utility_probability)
      )
    )
  return(data.frame(new_full_data))
}

fit_silver_model_and_resample = function(all_data,is_social,num_problems){
  fitted_model = if (is_social) fit_social_silver_model(all_data) else fit_asocial_silver_model(all_data)
  full_data = add_weights(all_data,fitted_model,num_problems) 
  full_data_with_resampled_choices = get_transmitted_social_info(full_data)
  return(full_data_with_resampled_choices)
}


fit_social_oracle = function(n_iter,n_chains){
  filepath  = paste0('fitted_oracles/social/iterations=',n_iter,'_','chains=',n_chains,'.rds')
  print(paste("Social model:",filepath))
  # Comment next four lines to force oracle to refit
  if (file.exists(filepath)){
    print('-- SOCIAL ORACLE LOADED --')
    return(readRDS(filepath))
  }
  print('-- FITTING SOCIAL ORACLE --')
  all_soc_wu_data = load_experiment_data() %>% 
    subset(generation != 0 & condition == 'SOC:W-U') %>% 
    mutate(
      k_chose_green = ifelse(randomization_green,k_chose_utility,8-k_chose_utility)
    )
  model_for_stan = stan_model('stan_models/oracles/social_oracle.stan')
  stan_data = list(N=nrow(all_soc_wu_data),
                   chose_green = as.integer(all_soc_wu_data$chose_green),
                   num_participants=length(unique(all_soc_wu_data$participant_id)),
                   participant_index = recode_indices(all_soc_wu_data$participant_id),
                   green_votes = k_chose_green,
                   green_factor = all_soc_wu_data$green_integers,
                   negative_green = ifelse(all_soc_wu_data$randomization_green==T,1,-1))
  options(mc.cores=1)
  fitted_model = sampling(model_for_stan,stan_data,iter=n_iter,chains=n_chains)
  saveRDS(fitted_model, file = filepath)
  return(fitted_model)
}


fit_asocial_oracle = function(n_iter,n_chains){
  filepath  = paste0('fitted_oracles/asocial/iterations=',n_iter,'_','chains=',n_chains,'.rds')
  print(paste("Asocial model:",filepath))
  # Comment next four lines to force oracle to refit
  if (file.exists(filepath)){
    print('-- ASOCIAL ORACLE LOADED --')
    return(readRDS(filepath))
  }
  asocial_data = load_experiment_data() %>% subset(condition == 'ASO:W-U')
  print('-- FITTING ASOCIAL ORACLE --')
  model_for_stan = stan_model('stan_models/oracles/asocial_oracle.stan')
  stan_data = list(N=nrow(asocial_data),
                   chose_green = as.integer(asocial_data$chose_green),
                   num_participants=length(unique(asocial_data$participant_id)),
                   participant_index = recode_indices(asocial_data$participant_id),
                   green_factor = asocial_data$green_integers,
                   negative_green = ifelse(asocial_data$randomization_green==T,1,-1))
  options(mc.cores=1)
  fitted_model = sampling(model_for_stan,stan_data,iter=n_iter,chains=n_chains)
  saveRDS(fitted_model, file = filepath)
  return(fitted_model)
}

recruit_social_generation = function(previous_generation_df,next_generation_num,max_participant_id,num_networks,num_problems,green_proportions){
  previous_generation_clipped = data.frame(previous_generation_df %>% 
                                             select(new_net_decision_index,condition_replication,current_resampled_k_chose_utility,current_original_k_chose_utility,resampling_type)) %>%
    rename(previous_resampled_k_chose_utility = current_resampled_k_chose_utility,
           previous_original_k_chose_utility = current_original_k_chose_utility)
  green_integers =  sapply(green_proportions,add_green_integers)
  resampling_types = unique(previous_generation_clipped$resampling_type)
  num_resampling_types = length(unique(resampling_types))
  num_participants_per_generation = 8
  
  num_participants = num_participants_per_generation * num_resampling_types * num_networks
  total_observations = num_networks * num_participants_per_generation * num_resampling_types * num_problems
  
  new_generation = data.frame(
    generation = rep(next_generation_num,total_observations),
    condition = rep('SOC:W-U',total_observations),
    participant_id = rep((max_participant_id+1):(max_participant_id+num_participants),each=num_problems),
    proportion_green = rep(green_proportions,num_participants),
    green_integers = rep(green_integers,num_participants),
    new_net_decision_index = rep(1:num_problems,num_participants),
    randomization_green = c(rep(0,total_observations/4),rep(1,total_observations/4),rep(0,total_observations/4),rep(1,total_observations/4)),
    condition_replication = rep(rep(1:num_networks,each=num_problems*num_participants_per_generation),2),
    resampling_type = rep(resampling_types,each=total_observations/num_resampling_types),
    chose_green = NA,
    chose_correct = NA,
    chose_utility = NA
  )
  new_generation = merge(new_generation,previous_generation_clipped,by=c('new_net_decision_index','condition_replication','resampling_type'))
  new_generation = new_generation[!duplicated(new_generation), ]
  return(new_generation)
}

recruit_asocial_generation = function(num_networks,num_problems,generation,green_proportions){
  green_integers = sapply(green_proportions,add_green_integers) 
  num_participants_per_generation = 8
  num_participants = num_participants_per_generation * num_networks
  total_observations = num_networks * num_problems * num_participants_per_generation
  new_generation = data.frame(
    generation = rep(generation,total_observations),
    condition = rep('ASO:W-U',total_observations),
    proportion_green = rep(green_proportions,num_participants),
    new_net_decision_index = rep(1:num_problems,num_participants),
    green_integers = rep(green_integers,num_participants),
    randomization_green = rep(c(0,1),each=total_observations/2),
    participant_id = rep(1:(num_participants),each=num_problems),
    condition_replication = rep(1:num_networks,each=num_participants_per_generation*num_problems),
    resampling_type = rep('none',total_observations)
  )
  return(new_generation)
}

simulate_social_choices = function(generation_df,oracle_model){
  # Uses fitted oracle model, sampled biases, and votes to simulate choices
  # Adds choices to the dataframe and returns it
  # A bit complicated because using seperate coeffcients for blue and green model
  samples = extract(oracle_model)
  vote_coefficient = mean(samples$vote_coefficient)
  intercept = mean(samples$intercept)
  observed_k_chose_utility = ifelse(generation_df$resampling_type=='identity',generation_df$previous_original_k_chose_utility,generation_df$previous_resampled_k_chose_utility)
  observed_k_chose_green = ifelse(generation_df$randomization_green==T,observed_k_chose_utility,8-observed_k_chose_utility)
  green_dummy_means = colMeans(samples$green_dummy)
  probability_choice_exp = exp(generation_df$green_bias + intercept +  (vote_coefficient*observed_k_chose_green) + green_dummy_means[generation_df$green_integers])
  probability_choice = probability_choice_exp / (1+probability_choice_exp)
  generation_df$chose_green = rbinom(prob=probability_choice,size=1,n=length(probability_choice))
  generation_df$chose_utility = generation_df$chose_green == generation_df$randomization_green
  generation_df$chose_correct = ifelse(generation_df$chose_green,generation_df$proportion_green>0.5,generation_df$proportion_green<0.5)
  return(generation_df)
}

simulate_asocial_choices = function(generation_df,oracle_model){
  # Uses fitted oracle model, sampled biases, and votes to simulate choices
  # Adds choices to the dataframe and returns it
  # A bit complicated because using seperate coeffcients for blue and green model
  samples = extract(oracle_model)
  intercept = mean(samples$intercept)
  green_dummy_means = colMeans(samples$green_dummy)
  probability_choice_exp = exp(generation_df$green_bias + intercept + green_dummy_means[generation_df$green_integers])
  probability_choice = probability_choice_exp / (1+probability_choice_exp)
  generation_df$chose_green = rbinom(prob=probability_choice,size=1,n=length(probability_choice))
  generation_df$chose_utility = generation_df$chose_green == generation_df$randomization_green
  generation_df$chose_correct = ifelse(generation_df$chose_green,generation_df$proportion_green>0.5,generation_df$proportion_green<0.5)
  return(generation_df)
}


sample_biases = function(input_df,oracle_model,num_networks,num_problems){
  # Fills in "green_bias" column
  samples = extract(oracle_model)
  mean_absolute_bias = mean(samples$absolute_bias)
  mean_bias_sd = mean(samples$bias_standard_deviation)
  num_resampling_types = length(unique(input_df$resampling_type))
  num_participants_per_network_gen = 8
  num_participants = num_participants_per_network_gen * num_resampling_types * num_networks
  bias_df = data.frame(
    'participant_id' = rep(unique(input_df$participant_id),each=num_problems),
    'absolute_biases' = rep(rnorm(n = num_participants,mean = mean_absolute_bias,sd = mean_bias_sd),each=num_problems)
  )
  new_df = merge(input_df,bias_df,by='participant_id')
  new_df = new_df[!duplicated(new_df),]
  new_df$green_biases = ifelse(new_df$randomization_green,new_df$absolute_biases,-1*new_df$absolute_biases)
  new_df = new_df[,!colnames(new_df) %in% 'absolute_biases']
  return(new_df)
}

convert_dataframe_for_binding = function(new_df){
  return(
    new_df %>%
      mutate(
        chose_green = as.logical(chose_green),
        chose_correct = as.logical(chose_correct),
        chose_utility = as.logical(chose_utility),
        condition = as.character(condition)
      ) %>%
      data.frame()
  )
}

custom_rbind = function(...){
  first_call = T
  for (dfi in list(...)){
    converted_dfi = convert_dataframe_for_binding(dfi)
    if (first_call){
      bound_df = converted_dfi
      first_call = F
    } else{
      bound_df = bind_rows(bound_df,converted_dfi)
    }
  }
  return(bound_df)
}

simulate_social_generation = function(full_data,social_oracle,irt_type,current_generation,num_networks,num_problems,green_proportions){
  social_data = full_data %>% 
    # Get previous generation data
    subset(generation==current_generation-1) %>%
    { if( current_generation==2 ) . else subset(., is_asocial==F)} %>%
    # Recruit next generation
    recruit_social_generation(current_generation,max(full_data$participant_id),num_networks,num_problems,green_proportions) %>%
    # Sample biases from oracle model
    sample_biases(social_oracle,num_networks,num_problems) %>%
    # Simulate choices
    simulate_social_choices(social_oracle) %>% 
    # Get propagated k's
    fit_silver_model_and_resample(irt_type=='social',num_problems)
  social_data$is_asocial = F
  return(social_data)
}

simulate_later_generations = function(full_data,asocial_oracle,social_oracle,irt_type,num_networks,num_problems,green_proportions){
  for (generation_i in 2:8){
    current_asocial_data = simulate_asocial_generation(asocial_oracle,num_networks,num_problems,generation_i,green_proportions)
    current_social_data = simulate_social_generation(full_data,social_oracle,irt_type,generation_i,num_networks,num_problems,green_proportions)
    full_data = custom_rbind(full_data,current_asocial_data,current_social_data)
  }
  return(full_data)
}

add_green_integers = function(proportion_green){
  if (proportion_green==0.48) return(1)
  if (proportion_green==0.49) return(2)
  if (proportion_green==0.51) return(3)
  return(4)
}

simulate_asocial_generation = function(asocial_oracle,num_networks,num_problems,generation,green_proportions){
  asocial_data = recruit_asocial_generation(num_networks,num_problems,generation,green_proportions) %>%
    # Add green_bias to asocial generation
    sample_biases(asocial_oracle,num_networks,num_problems) %>%
    # Simulate asocial choices
    simulate_asocial_choices(asocial_oracle)
  asocial_data$is_asocial = T
  return(asocial_data)
}

run_sims_and_save = function(experiment_iteration,output_directory){
  # -- -- -- -- -- -- -- -- 
  # EXPERIMENT PARAMETERS
  # -- -- -- -- -- -- -- --
  
  num_problems = 16 # must be divisible by 4
  num_networks = 14 # must be divisible by 2
  
  # Make sure inputs are ok
  if (as.numeric(num_problems) %% 4 != 0) stop("ERROR: Number of problems must be divisible by four!",call.=FALSE)
  if (as.numeric(num_networks) %% 2 != 0) stop("ERROR: Number of networks must be divisible by two!",call.=FALSE)

  # Set resampling types - don't mess here
  resampling_types = c('identity','simple')
  irt_type = 'asocial'
  
  # Set green proportions
  green_proportions = rep(c(0.48,0.49,0.51,0.52),4)

  # -- -- -- -- -- -- -- -- 
  # RUN SIMULATION
  # -- -- -- -- -- -- -- --
  
  # Fit oracles for simulating choices
  asocial_oracle = fit_asocial_oracle(2500,8)
  social_oracle = fit_social_oracle(2500,8)
  
  # Simulate generation 1 and resample choices
  asocial_data = asocial_oracle %>%
    simulate_asocial_generation(num_networks,num_problems,1,green_proportions) %>%
    fit_silver_model_and_resample(F,num_problems)
  
  # Clone generation 1 asocial data
  cloned_asocial_data = asocial_data[rep(1:nrow(asocial_data), length(resampling_types)), ]
  row.names(cloned_asocial_data) = NULL
  
  # Overwrite resampling types
  cloned_asocial_data$resampling_type = rep(resampling_types,each=num_networks * num_problems * 8)
  
  # Run social generations 2-8
  simulated_data = simulate_later_generations(cloned_asocial_data,asocial_oracle,social_oracle,irt_type,num_networks,num_problems,green_proportions)
  
  # -- -- -- -- -- -- -- -- 
  # SAVE DATA
  # -- -- -- -- -- -- -- --
  
  # Prepare for writing
  simulated_data = simulated_data %>%
    mutate(
      social_type = irt_type,
      og_simulation_num = experiment_iteration,
      green_proportions = paste(as.character(green_proportions),collapse=','),
      num_problems = num_problems,
      num_networks = num_networks
    )

  # Make relevant directories 
  if (!dir.exists(output_directory)) dir.create(output_directory,recursive = T)
  
  # Save data
  filename = paste0(output_directory,'/iterationCount=',experiment_iteration,".csv")
  write.csv(x=simulated_data, file=filename, row.names=F)
}

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 2) stop("ERROR: Wrong number of args supplied - 2 required!", call.=FALSE)

# Inputs: experiment_iteration,output_directory
run_sims_and_save(as.numeric(args[1]),args[2])




