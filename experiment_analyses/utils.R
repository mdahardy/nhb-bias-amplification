# -- -- -- -- -- -- -- -- -- -- - -- #
# -- -- -- -- -- -- -- -- -- -- - -- #
# Helper functions for experiments 1 and 2 analyses
# -- -- -- -- -- -- -- -- -- -- - -- #
# -- -- -- -- -- -- -- -- -- -- - -- #

# -- -- -- -- -- -- -- -- -- -- - -- #
# Load data
# -- -- -- -- -- -- -- -- -- -- - -- #

load_e1_data = function(include_cloned = F){
  return(
    '../../data/experiment_1.csv' %>%
      read.csv() %>%
      transmute(
        net_decision_index,
        randomization_color,
        is_practice = is_practice=='True',
        is_cloned = generation == 0 & grepl('SOC',condition,fixed=T),
        generation,
        condition_replication,
        participant_id,
        proportion_green = 1-proportion_blue,
        green_shown = green_first == 'True',
        is_social = grepl('SOC',condition,fixed=T),
        chose_bias = as.logical(chose_utility),
        chose_correct = as.logical(chose_correct),
        chose_green = ifelse(randomization_color=='green',chose_bias,!chose_bias),
        k_chose_correct = ifelse(proportion_blue>0.5,8-k_chose_green, k_chose_green),
        social_info_accuracy = k_chose_correct/8,
        social_info_bias = k_chose_utility/8,
        social_info_green = k_chose_green/8,
        matched = proportion_utility > 0.5,
        condition = case_when(
          condition == 'ASO:N-U' ~ 'asocial_control',
          condition == 'ASO:W-U' ~ 'asocial_bias',
          condition == 'SOC:N-U' ~ 'social_control',
          TRUE ~ 'social_bias'
        ),
        creation_time,
        end_time_float
      ) %>% 
      subset(is_practice == F) %>%
      { if (include_cloned) . else subset(., is_cloned == F) }
    )
}

load_e2_data = function(include_cloned = F){
  return(
    '../../data/experiment_2.csv' %>%
      read.csv() %>%
      transmute(
        generation,
        stimulus_id,
        is_cloned = as.character(front_end_condition) != as.character(condition),
        is_practice = is_practice=='True',
        randomization_color,
        participant_id,
        proportion_green,
        k_chose_green_tie,
        node_table_id,
        front_end_node_id,
        green_shown = green_first == 'True',
        condition_replication = paste0(condition_replication,randomization_color),
        network_identifier = paste0(condition,condition_replication),
        chose_correct = chose_correct == "True",
        chose_bias = chose_utility == 'True',
        chose_green = ifelse(randomization_color=='green',chose_bias,!chose_bias),
        is_social = grepl('SOC',condition,fixed=T),
        k_chose_correct = ifelse(proportion_green>0.5,k_chose_green, 8-k_chose_green),
        k_chose_green,
        social_info_accuracy = k_chose_correct/8,
        social_info_bias = k_chose_utility/8,
        social_info_green = k_chose_green/8,
        condition = case_when(
          grepl('ASO',condition,fixed=T) ~ 'asocial_bias',
          grepl('W-R', condition,fixed=T) ~ 'social_resampling',
          TRUE ~ 'social_bias'
        )
      ) %>% 
      { if (include_cloned) . else subset(., is_cloned == F) } %>%
      subset(is_practice == F)
  )
}

load_e3_data = function(){
    experiment_data = '../../data/experiment_3.csv' %>%
      read.csv() %>%
      subset(is_practice == 'False') %>%
      transmute(
        hit_id,
        stimulus_id,
        randomization_color,
        participant_id = paste(hit_id,participant_id),
        proportion_green,
        k_chose_green_tie,
        node_table_id,
        front_end_node_id,
        green_shown = green_first == 'True',
        condition_replication = paste0(condition_replication,randomization_color),
        network_identifier = paste0(front_end_condition,condition_replication),
        chose_correct = chose_correct == "True",
        chose_bias = chose_utility == 'True',
        chose_green = ifelse(randomization_color=='green',chose_bias,!chose_bias),
        is_social = is_asocial == 'False',
        k_chose_correct = ifelse(proportion_green>0.5,k_chose_green, 8-k_chose_green),
        k_chose_green,
        social_info_accuracy = k_chose_correct/8,
        social_info_bias = k_chose_utility/8,
        social_info_green = k_chose_green/8,
        condition = case_when(
          front_end_condition == 'asocial-none' ~ 'asocial',
          front_end_condition == 'social-oversampling' ~ 'oversampling',
          front_end_condition == 'social-resampling' ~ 'resampling',
          TRUE ~ 'other'
        )
      )
    
    # Get ids of participants that did all the trials
    participant_ids = experiment_data %>%
      group_by(participant_id) %>%
      summarize(n_trials = n()) %>%
      subset(n_trials == 16) %>%
      data.frame()
    
    # Exclude participants who didn't do all the trials
    experiment_data = subset(experiment_data,participant_id %in% participant_ids$participant_id)
    
    return(experiment_data)
    
}

# -- -- -- -- -- -- -- -- -- -- - -- #
# Fit and report unrestricted models and betas
# -- -- -- -- -- -- -- -- -- -- - -- #

fix_condition_names = function(condition_name){
  condition_name = gsub("condition","",condition_name)
  if (condition_name == 'asocial_bias') return('aso_bias')
  if (condition_name == 'social_bias') return('soc_bias')
  if (condition_name == 'social_control') return('soc_control')
  if (condition_name == 'asocial_control') return('aso_control')
  if (condition_name == 'social_resampling') return('soc_resampling')
  if (condition_name == 'social_oversampling') return('soc_oversampling')
  if (condition_name == 'randomization_colorgreen') return('green_rc')
  stop("Unexpected condition name: ",condition_name)
}

fit_and_report_betas_and_cis = function(experiment_data,model_formula,dependent_variables,regression_type,output_path,additional_str=''){
  conditions = unique(experiment_data$condition)
  experiment_data$condition = as.factor(experiment_data$condition)
  for (dv in dependent_variables){
    print(paste('Running',dv,'models'))
    dv_output_path = paste(output_path,'betas',dv,additional_str,sep='/')
    for (condition in conditions){
      print(paste('Running comparisons on',condition))
      pretty_condition = fix_condition_names(condition)
      experiment_data$condition = relevel(experiment_data$condition,ref=condition)
      formula_str = paste(dv, '~ + condition', model_formula)
      regression_fn = if (regression_type=='lmer') lmer_model else glmer_model
      model = regression_fn(experiment_data,formula_str)
      confidence_intervals = round(confint(model,devtol = Inf),2)
      print(confidence_intervals)
      betas = round(fixef(model)[-1],2)
      for (i in 1:length(betas)){
        beta_i = betas[i]
        alt_condition = names(betas[i])
        pretty_alt_condition = fix_condition_names(alt_condition)
        output_dir = paste0(dv_output_path,'/',pretty_condition,'-',pretty_alt_condition)
        make_directory(output_dir)
        
        # Write beta
        beta_filepath = paste(output_dir,'beta.tex',sep='/')
        beta_string = paste0('\\beta=', beta_i) %>% 
          wrap_string('$')
        write(beta_string,beta_filepath)
        
        cis = confidence_intervals[alt_condition,]
        names(cis) = NULL
        
        # Write CI
        pasted_cis = paste(cis,collapse=', ')
        ci_string = paste0('(',pasted_cis,')') %>% wrap_string('$')
        ci_filepath = paste(output_dir,'ci.tex',sep='/')
        write(ci_string,ci_filepath)
      }
    }
  }
}

# -- -- -- -- -- -- -- -- -- -- - -- #
# Print condition means function
# -- -- -- -- -- -- -- -- -- -- - -- #
get_condition_means = function(input_data){
  return(
    input_data %>%
      group_by(condition) %>%
      summarize(
        bias = round(mean(chose_bias),3),
        accuracy = round(mean(chose_correct),3)
      ) %>%
      data.frame()
  )
}

# -- -- -- -- -- -- -- -- -- -- - -- #
# Get chi-square statistics
# -- -- -- -- -- -- -- -- -- -- - -- #

# Returns a dataframe with all the p-values and chi-square statsitics for both the reported
# and prereg comparisons
compare_conditions = function(input_data,dependent_vars,regression_type,common_formula,prereg_addition){
  comparisons = unique(input_data$condition) %>% combn(2) %>% t()
  results_df = data.frame()
  print("-- -- Fitting unrestricted models -- --")
  unrestricted_models = input_data %>% 
    make_unrestricted_models(regression_type,common_formula,prereg_addition,dependent_vars)
  for (dv_i in dependent_vars){
    for (i in 1:nrow(comparisons)){
      vars = comparisons[i,]
      print(paste('-- -- Comparing', vars[1],vars[2],dv_i,'-- --'))
      restricted_models = input_data %>% 
        make_restricted_models(regression_type,common_formula,prereg_addition,dv_i,vars)
      prereg_anova = anova(unrestricted_models$prereg[[dv_i]],restricted_models$prereg)
      reported_anova = anova(unrestricted_models$reported[[dv_i]],restricted_models$reported)
      results_df = results_df %>%
        rbind(
          data.frame(
          condition_1 = vars[1],
          condition_2 = vars[2],
          dependent_variable = dv_i,
          reported_chisq = round(reported_anova[2,'Chisq'],2),
          reported_p = reported_anova[2,'Pr(>Chisq)'],
          prereg_chisq =  round(prereg_anova[2,'Chisq'],2),
          prereg_p = prereg_anova[2,'Pr(>Chisq)'])
        )
    }
  }
  return(results_df)
}


# Returns a list where the names correspond to the model type (prereg and reported)
make_restricted_models = function(input_data,regression_type,common_formula,prereg_addition,dv,vars){
  model_list = list()
  for (model_type in c('reported','prereg')){
    curr_addition = if (model_type == 'prereg') prereg_addition else ''
    input_data = mutate(input_data, restricted_condition = ifelse(condition %in% vars,'comp', condition))
    condition_str = if (length(unique(input_data$restricted_condition)) == 1) '1' else 'restricted_condition'
    model_str = paste0(dv,'~',condition_str,common_formula,curr_addition)
    regression_fn = if(regression_type=='lmer') lmer_model else glmer_model
    model_list[[model_type]] = regression_fn(input_data,model_str)
  }
  return(model_list)
}

# Returns a list where the names correspond to the model type (prereg and reported)
# and the dependent variable
make_unrestricted_models = function(input_data,regression_type,common_formula,prereg_addition,dependent_vars){
  common_formula = paste('~condition',common_formula)
  unrestricted_models = list('reported' = list(),'prereg'=list())
  for (model_type in c('reported','prereg')){
    for (dv in dependent_vars){
      curr_addition = if (model_type=='prereg') prereg_addition else ''
      model_str = paste0(dv,common_formula,curr_addition)
      regression_fn = if(regression_type=='lmer') lmer_model else glmer_model
      unrestricted_models[[model_type]][[dv]]= regression_fn(input_data,model_str)
    }
  }
  return(unrestricted_models)
}

# Helper function for making unrestricted and restricted logistics glmer models
glmer_model = function(input_data,formula_string){
  return(
    formula_string %>%
      as.formula() %>%
      glmer(data=input_data,family='binomial',control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))
  )
}

# Helper function for making unrestricted and restricted lmer models
lmer_model = function(input_data,formula_string){
  return(
    formula_string %>%
      as.formula() %>%
      lmer(data=input_data,REML=F)
  )
}

# -- -- -- -- -- -- -- -- -- -- - -- #
# Write chi-square statistics for use in paper
# -- -- -- -- -- -- -- -- -- -- - -- #

write_tests = function(comparison_df,exp_dir,write_table_stats,blue_green_string = ''){
  print('-- -- -- Writing statistical tests -- -- --')
  for (i in 1:nrow(comparison_df)){
    row_i = comparison_df[i,]
    cond_comparison_str = paste(row_i$condition_1,row_i$condition_2,sep='-')
    stats_dir = paste(exp_dir,
                      'tests',
                      row_i$dependent_variable,
                      blue_green_string,
                      cond_comparison_str,
                      sep='/') %>% shorten_condition_string()
    make_dir_write_stats(row_i,stats_dir,'text')
    if (write_table_stats){
      table_stats_dir = gsub('tests','table_stats',stats_dir)
      make_dir_write_stats(row_i,table_stats_dir,'table')
    }
  }
}

make_dir_write_stats = function(stats_row,output_directory,statistics_type){
  make_directory(output_directory)
  chi2_function = if (statistics_type=='text') chi2_text_str else chi2_table_str
  write_chi2_statistic(stats_row,chi2_function,'reported',output_directory)
  write_chi2_statistic(stats_row,chi2_function,'prereg',output_directory)
}

write_chi2_statistic = function(stats_row,chi2_str_function,test_type,dir_string){
  stats_string = chi2_str_function(stats_row[paste0(test_type,'_chisq')],stats_row[paste0(test_type,'_p')])
  stats_file = paste0(dir_string,'/',test_type,'.tex')
  write(stats_string, stats_file)
}

make_directory = function(dir_string){
  if (!dir.exists(dir_string)) dir.create(dir_string,recursive = T)
}

chi2_text_str = function(chi2,p_val){
  p_string = if (p_val<0.001) 'p<0.001' else paste0('p=',round(p_val,3))
  chi2_string = paste0('\\chi^2(1)=',round(chi2,2))
  return(paste0(wrap_string(chi2_string,'$'), ', ', wrap_string(p_string,'$')))
}

chi2_table_str = function(chi2,p_val){
  p_stars = get_p_stars(p_val)
  stats_string = round(chi2,1) %>%
    paste0(p_stars) %>%
    wrap_string('$')
  return(stats_string)
}

get_p_stars = function(p_val){
  if (p_val<0.001) return ('^{***}')
  if (p_val<0.01) return ('^{**}')  
  if (p_val<0.05) return ('^{*}')
  return('')
}

wrap_string = function(string,wrapping_char){
  return(paste0(wrapping_char,string,wrapping_char))
}

shorten_condition_string = function(string){
  string = gsub('asocial','aso',string)
  return(gsub('social','soc',string))
}

# -- -- -- -- -- -- -- -- -- -- - -- #
# Write condition means for use in paper
# -- -- -- -- -- -- -- -- -- -- - -- #

# Returns subset given a list (i.e., dict) of condition
subset_at = function(input_data,condition_list){
  for (name_i in names(condition_list)){
    input_data = input_data %>% subset(name_i == condition_list[name_i])
  }
  return(input_data)
}

write_means = function(all_data,exp_dir,dependent_variables){
  for (condition_i in unique(all_data$condition)){
    condition_subset = all_data %>% subset(condition == condition_i)
    shortened_condition = shorten_condition_string(condition_i)
    for (dv_i in dependent_variables){
      write_condition_mean(condition_subset,shortened_condition,exp_dir,dv_i)
    }
  }  
}


write_condition_mean = function(condition_data,additional_dir_info,exp_dir,dependent_var){
  output_dir = paste(exp_dir,'means',dependent_var,additional_dir_info,sep='/')
  make_directory(output_dir)
  condition_mean = mean(condition_data[,dependent_var]) %>% round(3)
  write_statistic(condition_mean,'proportion',output_dir)
  write_statistic(condition_mean*100,'percentage',output_dir)
}

write_statistic = function(condition_statistic,statistic_type,output_directory){
  output_filepath = paste0(output_directory,'/',statistic_type,'.tex')
  write(condition_statistic,output_filepath)
}

write_blue_green_means = function(input_data,color,cond,base_dir){
  condition_dir = paste(base_dir,cond,sep='/')
  color_data = input_data %>% subset(condition == cond)
  color_mean = mean(color_data$social_info_green) %>% round(3)
  color_dir = paste(condition_dir,paste0(color,'_bias'),sep='/')
  make_directory(color_dir)
  write_statistic(color_mean,'proportion',color_dir)
  write_statistic(color_mean*100,'percentage',color_dir)
}


# -- -- -- -- -- -- -- -- -- -- - -- #
# -- -- -- -- -- -- -- -- -- -- - -- #
# Helper functions for perceptual priming exploratory analyses
# -- -- -- -- -- -- -- -- -- -- - -- #
# -- -- -- -- -- -- -- -- -- -- - -- #


group_and_write_summary_stats = function(all_tie_data,grouping_var,grouping_var_level,mean_var,base_directory){
  relevant_data = all_tie_data[all_tie_data[,grouping_var] == grouping_var_level,]
  output_dir = paste(base_directory,paste(grouping_var,grouping_var_level,sep='='),sep='/')
  write_summary_stats(relevant_data,mean_var,output_dir)
}

write_summary_stats = function(all_data,mean_var,output_dir){
  mean_val = mean(all_data[,mean_var]) %>% round(3) * 100
  mean_var_dir = paste(output_dir,mean_var,sep='/')
  make_directory(mean_var_dir)
  # Write the mean
  write(mean_val,paste(mean_var_dir,'percentage.tex',sep='/'))
  # Write the n
  n = formatC(nrow(all_data), format="f", big.mark=",", digits=0)
  write(n,paste(output_dir,'n.tex',sep='/'))
}

z_stat_string = function(estimate,confidence_intervals,z_statistic,p_value){
  rounded_estimate = round(estimate,2)
  estimate_string = paste0('\\beta=',rounded_estimate) %>% wrap_string('$')
  inner_ci_string = paste(confidence_intervals,collapse=', ')
  ci_string = paste0('95\\% CI: $(',inner_ci_string,')$')
  rounded_z_statistic = round(z_statistic,2)
  z_string = paste('z',rounded_z_statistic,sep='=') %>% wrap_string('$')
  ci_and_z_string = paste(ci_string,z_string,sep='; ')
  p_string = if (p_value<0.001) 'p<0.001' else paste0('p=',round(p_value,3))
  p_string = p_string %>% wrap_string('$')
  return(paste(estimate_string,ci_and_z_string,p_string,sep=', '))
}

fit_tie_model_and_write = function(all_data,formula_string,stats_dir){
  fitted_model = formula_string %>%
    as.formula() %>%
    glm(data=all_data,family=binomial)
  confidence_intervals = round(confint(fitted_model),2)
  model_coefs = fitted_model %>%
    summary() %>%
    coef()
  rc_index = which(rownames(model_coefs)=="randomization_colorgreen")
  gs_index = which(rownames(model_coefs)=="green_shownTRUE")
  make_directory(stats_dir)
  write_tie_statistic(
    model_coefs[rc_index,1],
    model_coefs[rc_index,3],
    model_coefs[rc_index,4],
    confidence_intervals[rc_index,],
    stats_dir,
    'randomization_color.tex')
  write_tie_statistic(
    model_coefs[gs_index,1],
    model_coefs[gs_index,3],
    model_coefs[gs_index,4],
    confidence_intervals[gs_index,],
    stats_dir,
    'green_shown.tex')
}

write_tie_statistic = function(beta_estimate,z_statistic,p_val,confidence_intervals,stats_dir,output_name){
  names(confidence_intervals) = NULL
  z_string = z_stat_string(beta_estimate,confidence_intervals,z_statistic,p_val)
  write(z_string,paste(stats_dir,output_name,sep='/'))
}

### --- --- --- --- --- --- ### 
### --- Helper functions for writing a stan model --- ###
### --- --- --- --- --- --- ### 

write_stan_stat = function(stat,num_digits,stat_name,stat_directory){
  rounded_stat = round(stat,num_digits)
  write_statistic(rounded_stat,stat_name,stat_directory)
}

write_samples = function(stan_samples,model_directory){
  round_digits = 3
  for (var_i in colnames(stan_samples)){
    param_directory = paste(model_directory,var_i,sep='/')
    make_directory(param_directory)
    samples = stan_samples[,var_i]
    write_stan_stat(mean(samples),round_digits,'mean',param_directory)
    write_stan_stat(quantile(samples,0.05),round_digits,'lower_ci',param_directory)
    write_stan_stat(quantile(samples,0.95),round_digits,'upper_ci',param_directory)
  }
  n = formatC(nrow(stan_samples), format="f", big.mark=",", digits=0)
  write_statistic(n,'n',model_directory)
}

write_stan_model = function(stan_model_filepath,save_directory){
  stan_model_filepath %>% 
    readRDS() %>%
    extract() %>%
    data.frame() %>%
    select(-matches('participant_bias|lp')) %>%
    rename(
      'green_dots_48' = green_dummy.1,
      'green_dots_49' = green_dummy.2,
      'green_dots_51' = green_dummy.3,
      'green_dots_52' = green_dummy.4
    ) %>%
    write_samples(save_directory)
}