# -- -- -- -- -- -- -- -- -- -- - -- #
# -- -- -- -- -- -- -- -- -- -- - -- #
# Helper functions specifically for plotting scripts
# -- -- -- -- -- -- -- -- -- -- - -- #
# -- -- -- -- -- -- -- -- -- -- - -- #

# Get the standard error of a binary variable
bernoulli_se = function(p,n){
  return(sqrt((p*(1-p))/n))
}

# Groups data across grouping_variables and summarizes relevatn variables
summarize_choice_data = function(raw_data,grouping_variables){
  return(
    raw_data %>%
      group_by_at(grouping_variables) %>%
      summarize(
        n=n(),
        bias = mean(chose_bias),
        bias_se = bernoulli_se(bias,n),
        accuracy = mean(chose_correct),
        green = mean(chose_green),
        green_se = bernoulli_se(green,n),
        accuracy_se = bernoulli_se(accuracy,n),
      ) %>%
      data.frame()
  )
}

summarize_observed_data = function(raw_data,grouping_variables){
  return(
    raw_data %>%
      group_by_at(grouping_variables) %>%
      summarize(
        n=n(),
        social_info_green = mean(social_info_green),
        social_info_green_se = bernoulli_se(social_info_green,n),
        social_info_bias = mean(social_info_bias),
        social_info_bias_se = bernoulli_se(social_info_bias,n),
        social_info_correct = mean(social_info_accuracy),
        social_info_correct_se = bernoulli_se(social_info_correct,n)
      ) %>%
      data.frame()
  )
}

# Converts the condition names to pretty names for plotting
pretty_condition_names = function(raw_data){
  return(
    raw_data %>%
      mutate(
        condition = case_when(
          condition == 'asocial_control' ~ 'Asocial control',
          condition == 'asocial_bias' ~ 'Asocial bias',
          condition == 'social_bias' ~ 'Social bias',
          condition == 'social_control' ~ 'Social control',
          condition == 'social_resampling' ~ 'Social resampling'
        )
      )
  )
}

exclude_asocial_data = function(input_data){
  return(
    input_data %>%
      subset(
        grepl('Soc',condition,fixed=T)
      )
  )
}

theme_styling = function(){
  return(
    list(
      theme_classic(),
      theme(
        axis.title = element_text(size=20),
        axis.text = element_text(color='black',size=18),
        axis.title.y = element_text(margin = margin(r=10)),
        axis.title.x = element_text(margin = margin(t=10)),
        legend.title = element_text(size=20),
        legend.text = element_text(size=18),
        strip.text = element_text(size=20),
        plot.margin = unit(c(7,5.5,5.5,5.5), "pt")
      )
    )
  )
}

barplot_styling = function(){
  return(
    list(
      theme(legend.position = 'none',
            axis.text.x = element_text(size=20),
            axis.title.x=element_blank())
    )
  )
}

get_plot_colors = function(experiment_num){
  global_colors = c(
    "Asocial control" = "#ffffbf",
    "Asocial bias" =  "#fdae61",
    "Social control" ='#abd9e9',
    "Social bias" = "#d7191c", 
    "Social resampling" = '#2c7bb6',
    "Social correlated" = '#3BBA6E'
  )
  condition_colors = c(
    global_colors['Asocial bias'],
    global_colors['Social bias']
  )
  if (experiment_num == 1){
    condition_colors = c(
      condition_colors,
      global_colors['Asocial control'],
      global_colors['Social control'])
  } else if (experiment_num == 3){
    condition_colors = c(global_colors['Asocial bias'],global_colors['Social correlated'],global_colors['Social resampling'])
  } else{
    condition_colors = c(condition_colors,global_colors['Social resampling'])
  }
  return(list(
    condition_colors = condition_colors,
    blue_green_colors = c('blue'='#2c7fb8','green'='#a1dab4')
    ))
}

refactor_condition = function(df,ordering){
  return(
    df %>%
      mutate(condition = factor(condition,levels = ordering))
  )
}