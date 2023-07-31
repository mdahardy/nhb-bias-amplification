### --- Script for generating and saving the figures for Experiment 3 --- ###

library(ggplot2)
library(dplyr)
library(ggsignif)
library(ggthemes)
# Helper functions
source('../utils.R')
source('../plotting_utils.R')

### --- --- --- --- ### 
### --- Set up --- ###
### --- --- --- --- ###

# The units for defining the dimensions of the plots: cm is centimeters
plots_units = 'cm'
# The height in plots_units for saving the figures
plots_height = 13
# The width in plots_units for saving the figures
plots_width = 14

# Add line breaks for better styling
condition_names_with_breaks = c('Asocial\nbias','Social\ncorrelated','Social\nresampling')
# The directory where the plots should be saved
plots_directory = '../../paper/plots/e3/'
# Make this directory if it doesn't exist
make_directory(plots_directory)

# -- -- -- -- -- --
# Plot styling
# -- -- -- -- -- --

# Colors
plot_colors = get_plot_colors(3)
condition_colors = plot_colors$condition_colors

# Condition shapes
condition_shapes = c(
  'Asocial bias' = 21,
  'Social correlated' = 23,
  'Social resampling'=24)

### --- --- --- --- ### 
### --- Load and prepare data --- ###
### --- --- --- --- ### 

# All e2 data
e3_data = load_e3_data() %>%
  mutate(
    condition = case_when(
      condition == 'asocial' ~ 'Asocial bias',
      condition == 'oversampling' ~ 'Social correlated',
      condition == 'resampling' ~ 'Social resampling',
      T ~ 'other'
    )
  )

### --- --- --- --- --- --- --- --- ### 
### --- --- --- PLOTS --- --- --- ---###
### --- --- --- --- --- --- --- --- ### 

### --- --- --- --- --- --- --- --- ### 
### --- Bias by condition bar plot --- ###
### --- --- --- --- --- --- --- --- ### 

# Bias bar plot
bias_plot = e3_data %>%
  summarize_choice_data('condition') %>%
  ggplot(aes(x=condition,y=bias,fill=condition)) + 
  geom_bar(stat='identity') +
  scale_fill_manual("legend", values = plot_colors$condition_colors)+
  geom_errorbar(aes(ymin=bias-bias_se,ymax=bias+bias_se),width=0.2) +
  geom_signif(y_position = c(0.588,0.588), xmin = c(1,2.05), 
              xmax = c(1.95,3), annotation = c('***','**'),
              textsize=7,vjust=0.3) +
  ylab('Proportion of bias-aligned\njudgments') + 
  coord_cartesian(ylim=c(0.515,0.59))+
  scale_x_discrete(labels=condition_names_with_breaks) +
  theme_styling() + 
  barplot_styling()

bias_plot

# Save
paste0(plots_directory,'bias.pdf') %>%
  ggsave(bias_plot,width=plots_width,height=plots_height,units=plots_units)

### --- --- --- --- --- --- --- --- --- ### 
### --- Accuracy by condition bar plot --- ###
### --- --- --- --- --- --- --- --- --- ### 

# Accuracy bar plot
accuracy_plot = e3_data %>%
  summarize_choice_data('condition') %>%
  ggplot(aes(x=condition,y=accuracy,fill=condition)) + 
  geom_bar(stat='identity') +
  scale_fill_manual("legend", values = plot_colors$condition_colors)+
  geom_errorbar(aes(ymin=accuracy-accuracy_se,ymax=accuracy+accuracy_se),width=0.2) +
  # geom_signif(y_position = c(0.64,0.65), xmin = c(1,1), 
  #             xmax = c(2,3), annotation = c('***','***'),
  #             textsize=7,vjust=0.3) +
  ylab('Proportion of correct judgments') + 
  coord_cartesian(ylim=c(0.5575,0.6172))+
  scale_x_discrete(labels=condition_names_with_breaks)+
  theme_styling() +
  barplot_styling()

accuracy_plot

# Save
paste0(plots_directory,'accuracy.pdf') %>%
  ggsave(accuracy_plot,width=plots_width,height=plots_height,units=plots_units)
