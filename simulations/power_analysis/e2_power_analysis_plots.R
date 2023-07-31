### --- Script for generating and saving the simulation results plot --- ###

library(ggplot2)
library(dplyr)
library(ggpubr)
library(facetscales)

# -- -- -- -- -- --
# -- -- -- -- -- --
# Load helpers
# -- -- -- -- -- --
# -- -- -- -- -- --

source('plotting_utils.R')
source('utils.R')

# -- -- -- -- -- --
# -- -- -- -- -- --
# Custom helper functions
# -- -- -- -- -- --
# -- -- -- -- -- --

get_simulation_filepath = function(num_problems,simulation_num){
  return(
    paste0(
      '../data/simulated_data/numProblems=',num_problems,
      '_numNetworks=14',
      '_iterationCount=',simulation_num,
      '.csv')
  )
}

clean_simulation_data = function(input_data){
  return(
    input_data %>%
      transmute(
        generation,
        chose_utility = as.logical(chose_utility),
        chose_correct = as.logical(chose_correct),
        condition = paste(condition,resampling_type,sep=':'),
        # bottom puts generation 1 in the condition they're yoked to
        condition = case_when(
          condition == "ASO:W-U:simple" ~ 'Social resampling',
          condition == "ASO:W-U:identity" ~ 'Social motivated',
          condition == "ASO:W-U:none" ~ 'Asocial motivated',
          condition == "SOC:W-U:simple" ~ 'Social resampling',
          condition == "SOC:W-U:identity" ~ 'Social motivated',
          TRUE ~ condition
        ),
        is_cloned = generation == 1 & condition != 'Asocial motivated',
      ) %>%
      data.frame()
  )
}

fix_yoking_issues = function(input_data){
  return(
    input_data %>%
      subset(generation == 1) %>%
      subset(resampling_type == 'identity') %>%
      mutate(
        resampling_type = 'none'  
      ) %>%
      rbind(input_data)
  )
}

summarize_simulation = function(simulation_num,adjustment_amount){
  raw_sim_data = get_simulation_filepath(16,simulation_num) %>% 
    read.csv() %>% 
    fix_yoking_issues() %>%
    clean_simulation_data()
  # Get data for each generation - include cloned data for better plotting
  generation_data = raw_sim_data %>%
    group_by(generation,condition) %>%
    summarize(
      accuracy = mean(chose_correct),
      bias = mean(chose_utility),
      n = n()
    ) %>% 
    mutate(type = 'generation',
           generation_double = as.double(generation)
    ) %>%
    data.frame()
  # Get data pooled across generations - remove cloned
  pooled_data = raw_sim_data %>%
    subset(is_cloned == F) %>%
    group_by(condition) %>%
    summarize(
      accuracy = mean(chose_correct),
      bias =  mean(chose_utility),
      n = n()
    ) %>% 
    data.frame() %>%
    mutate(generation = -1,
           type = 'pooled',
           generation_double =case_when(
             condition == 'Asocial motivated' ~ -1 - adjustment_amount,
             condition == 'Social motivated' ~ -1,
             condition == 'Social resampling' ~ -1 + adjustment_amount
           )
    )
  # Bind together and add info columns
  return(
    generation_data %>%
      rbind(pooled_data) %>%
      mutate(
        simulation_num = simulation_num,
        is_cloned = generation == 1 & condition != "Asocial motivated",
        network_identifier = paste(condition,simulation_num,sep='-')
      )
  )
}

summarize_all_simulations = function(adjustment_amount){
  simulation_data = data.frame()
  for (i in 0:99){
    print(paste("Analyzing experiment simulation",i+1, 'of 100'))
    curr_sim_data = summarize_simulation(i,adjustment_amount)
    simulation_data = rbind(simulation_data,curr_sim_data)
  }
  return(simulation_data)
}

se = function(var){
  return(sd(var)/sqrt(length(var)))
}

# -- -- -- -- -- --
# -- -- -- -- -- --
# Make plots
# -- -- -- -- -- --
# -- -- -- -- -- --

# The units for defining the dimensions of the plots: cm is centimeters
plots_units = 'cm'
# The height in plots_units for saving the figures
plots_height = 11.5
# The width in plots_units for saving the figures
plots_width = 30

# The directory where the plots should be saved
plots_directory = 'plots/power_analysis/'
# Make this directory if it doesn't exist
make_directory(plots_directory)

# -- -- -- -- -- --
# Load and clean data data
# -- -- -- -- -- --

# For every simulation, summarize bias and accuracy 
simulation_data = summarize_all_simulations(2.5)

# Get averages across all generations
generation_means = simulation_data %>%
  subset(is_cloned == F) %>%
  group_by(generation,condition) %>%
  summarize(
    total_n = n() * n[1],
    n = n(), #* unique(n),
    bias_se = se(bias),
    bias = mean(bias),
    bias_bernoulli_se = bernoulli_se(bias,total_n),
    accuracy_se = se(accuracy),
    accuracy = mean(accuracy),
    accuracy_bernoulli_se = bernoulli_se(accuracy,total_n),
    generation_double = unique(generation_double)
  ) %>% 
  mutate(
    type = ifelse(generation != -1,'generation', 'pooled')
  ) %>%
  data.frame()


# -- -- -- -- -- --
# Plot styling
# -- -- -- -- -- --

# Condition colors
condition_colors = get_plot_colors(2)$condition_colors

# Condition shapes
condition_shapes = c(
  'Asocial motivated' = 16,
  'Social motivated' = 15,
  'Social resampling' = 17)

# Change relative width of generation and pooled plots
x_scales = list(
  'pooled' = scale_x_continuous(limits = c(-4.5,2.5),breaks=NULL),
  'generation' = scale_x_continuous(limits=c(1,8),breaks=1:8))

# -- -- -- -- -- --
# Bias plot
# -- -- -- -- -- --

generation_means$type = factor(generation_means$type,levels=c('pooled','generation'))
simulation_data$type = factor(simulation_data$type,levels=c('pooled','generation'))

# Plot bias by generation
bias_plot = generation_means %>%
  ggplot(aes(x=generation_double,
             y=bias,
             group=condition,
             color=condition,
             shape = condition,
             fill=condition)) +
  geom_line(
    data = subset(simulation_data,type == 'generation'),
    aes(x=generation_double,y=bias,group=network_identifier,color=condition),
    alpha = 0.11) +
  geom_jitter(
    data = subset(simulation_data,type == 'pooled'),
    aes(x=generation_double,y=bias,group=network_identifier,color=condition),
    alpha = 0.3,height = 0,width=1.25) +
  geom_violin(
    data = subset(simulation_data,type=='pooled'),
    aes(x=generation_double,y=bias,group=condition,color=condition),
    alpha = 0.15,position='identity')+
  geom_point(size=4) +
  geom_errorbar(aes(ymin=bias-bias_se, ymax=bias+bias_se),
                size=0.5,width=.37) +
  labs(x='Generation',y='Proportion of biased choices') +
  scale_y_continuous(breaks=c(0.52,0.55,0.58,0.61,0.64,0.67),
                     limits=c(0.52,0.6703),expand=c(0,0))+
  scale_x_continuous(breaks = 1:8) +
  scale_color_manual("legend", values = condition_colors)+
  scale_fill_manual("legend", values = condition_colors)+
  scale_shape_manual("legend", values = condition_shapes)+
  facet_grid_sc(cols=vars(type),space='free_x',scales=list(x=x_scales)) +
  theme_styling()  +
  theme(strip.background = element_blank(),
        strip.text.x = element_blank(),
        panel.grid.major = element_line(colour = "#ededed"),
        legend.position='none')

bias_plot

# Save
paste0(plots_directory,'bias_raw.pdf') %>%
  ggsave(bias_plot,width=plots_width,height=plots_height,units=plots_units)

# -- -- -- -- -- --
# Accuracy plot
# -- -- -- -- -- --

accuracy_plot = generation_means %>%
  ggplot(aes(x=generation_double,
             y=accuracy,
             group=condition,
             color=condition,
             shape = condition,
             fill=condition)) +
  geom_line(
    data = subset(simulation_data,type == 'generation'),
    aes(x=generation_double,y=accuracy,group=network_identifier,color=condition),
    alpha = 0.11) +
  geom_jitter(
    data = subset(simulation_data,type == 'pooled'),
    aes(x=generation_double,y=accuracy,group=network_identifier,color=condition),
    alpha = 0.3,height = 0,width=1.25) +
  geom_violin(
    data = subset(simulation_data,type=='pooled'),
    aes(x=generation_double,y=accuracy,group=condition,color=condition),
    alpha = 0.15,position='identity')+
  # geom_boxplot(
  #   data = subset(simulation_data,type=='pooled'),
  #   aes(x=generation_double,y=accuracy,group=condition,color=condition),
  #   alpha = 0.15,position='identity')+
  geom_point(size=4) +
  geom_errorbar(aes(ymin=accuracy-accuracy_se, ymax=accuracy+accuracy_se),
                size=0.5,width=.37) +
  labs(x='Generation',y='Proportion of correct choices') +
  scale_y_continuous(breaks = c(0.58,0.61,0.64,0.67,0.7))+
  scale_x_continuous(breaks = 1:8) +
  scale_color_manual("legend", values = condition_colors)+
  scale_fill_manual("legend", values = condition_colors)+
  scale_shape_manual("legend", values = condition_shapes)+
  facet_grid_sc(cols=vars(type),space='free_x',scales=list(x=x_scales)) +
  theme_styling()  +
  theme(strip.background = element_blank(),
        strip.text.x = element_blank(),
        panel.grid.major = element_line(colour = "#ededed"),
        legend.position='none')

accuracy_plot

# Save
paste0(plots_directory,'accuracy_raw.pdf') %>%
  ggsave(accuracy_plot,width=plots_width,height=plots_height,units=plots_units)



















