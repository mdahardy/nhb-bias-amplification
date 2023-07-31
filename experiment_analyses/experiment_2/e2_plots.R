### --- Script for generating and saving the figures for Experiment 2 --- ###

library(ggplot2)
library(dplyr)
library(ggsignif)
library(ggthemes)
# Helper functions
source('utils.R')
source('plotting_utils.R')

### --- --- --- --- ### 
### --- Set up --- ###
### --- --- --- --- ###

# The units for defining the dimensions of the plots: cm is centimeters
plots_units = 'cm'
# The height in plots_units for saving the figures
plots_height = 13
# The width in plots_units for saving the figures
plots_width = 14

# Fill colors for each condition in the bar plots
plot_colors = get_plot_colors(2)

# Add line breaks for better styling
condition_names_with_breaks = c('Asocial\nmotivated','Social\nmotivated','Social\nresampling')
# The directory where the plots should be saved
plots_directory = 'plots/e2/'
# Make this directory if it doesn't exist
make_directory(plots_directory)

# -- -- -- -- -- --
# Plot styling
# -- -- -- -- -- --

# Colors
plot_colors = get_plot_colors(2)
condition_colors = plot_colors$condition_colors
blue_green_colors = plot_colors$blue_green_colors

# Condition shapes
condition_shapes = c(
  'Asocial motivated' = 21,
  'Social motivated' = 22,
  'Social resampling'=24)

### --- --- --- --- ### 
### --- Load and prepare data --- ###
### --- --- --- --- ### 

# All e2 data
e2_data = load_e2_data(include_cloned = T) %>% 
  pretty_condition_names()

### --- Choice data for comparing original vs. resampled choices --- ###
original_choice_data = e2_data %>%
  subset(condition == 'Social resampling') %>%
  summarize_choice_data(c('generation','network_identifier','randomization_color')) %>%
  mutate(type='original')

resampled_choice_data = e2_data %>%
  subset(condition == 'Social resampling') %>%
  subset(generation>1) %>%
  group_by(generation,network_identifier,randomization_color) %>%
  summarize(
    n=n(),
    bias = mean(social_info_bias),
    bias_se = sqrt(var(social_info_bias) / n),
    green = mean(social_info_green),
    green_se = sqrt(var(social_info_green) / n),
    accuracy = mean(social_info_accuracy),
    accuracy_se = sqrt(var(social_info_accuracy) / length(accuracy)),
  ) %>%
  mutate(generation = generation -1,type = 'resampled')

choice_data = rbind(original_choice_data,resampled_choice_data) %>%
  arrange(match(randomization_color, c('green','blue'))) %>%
  mutate(randomization_color = factor(randomization_color,levels=c('green','blue')))

### --- Organize choice_data differently for scatter plots --- ###
original_data = choice_data %>%
  subset(generation<8) %>%
  subset(type=='original') %>%
  transmute(
    generation_network = paste(generation,network_identifier,sep='-'),
    network_identifier,
    randomization_color,
    original_bias = bias,
    original_bias_se = bias_se,
    original_green = green,
    original_green_se = green_se
  )

resampled_data = choice_data %>%
  subset(generation<8) %>%
  subset(type=='resampled') %>%
  transmute(
    generation_network = paste(generation,network_identifier,sep='-'),
    randomization_color,
    network_identifier,
    resampled_bias = bias,
    resampled_bias_se = bias_se,
    resampled_green = green,
    resampled_green_se = green_se
  )

scatter_data = merge(original_data,resampled_data,by='generation_network')

### --- --- --- --- --- --- --- --- ### 
### --- --- --- PLOTS --- --- --- ---###
### --- --- --- --- --- --- --- --- ### 

### --- --- --- --- --- --- --- --- ### 
### --- Bias by condition bar plot --- ###
### --- --- --- --- --- --- --- --- ### 

# Bias bar plot
bias_plot = e2_data %>%
  subset(is_cloned == F) %>% 
  summarize_choice_data('condition') %>%
  ggplot(aes(x=condition,y=bias,fill=condition)) + 
  geom_bar(stat='identity') +
  scale_fill_manual("legend", values = plot_colors$condition_colors)+
  geom_errorbar(aes(ymin=bias-bias_se,ymax=bias+bias_se),width=0.2) +
  geom_signif(y_position = c(0.575,0.575), xmin = c(1,2.05), 
              xmax = c(1.95,3), annotation = c('***','**'),
              textsize=7,vjust=0.3) +
  ylab('Proportion of biased choices') + 
  coord_cartesian(ylim=c(0.52,0.5772))+
  scale_x_discrete(labels=condition_names_with_breaks) +
  theme_styling() + 
  barplot_styling()

# Save
paste0(plots_directory,'bias.pdf') %>%
  ggsave(bias_plot,width=plots_width,height=plots_height,units=plots_units)

### --- --- --- --- --- --- --- --- --- ### 
### --- Accuracy by condition bar plot --- ###
### --- --- --- --- --- --- --- --- --- ### 

# Accuracy bar plot
accuracy_plot = e2_data %>%
  subset(is_cloned == F) %>% 
  summarize_choice_data('condition') %>%
  ggplot(aes(x=condition,y=accuracy,fill=condition)) + 
  geom_bar(stat='identity') +
  scale_fill_manual("legend", values = plot_colors$condition_colors)+
  geom_errorbar(aes(ymin=accuracy-accuracy_se,ymax=accuracy+accuracy_se),width=0.2) +
  geom_signif(y_position = c(0.64,0.65), xmin = c(1,1), 
              xmax = c(2,3), annotation = c('***','***'),
              textsize=7,vjust=0.3) +
  ylab('Proportion of correct choices') + 
  coord_cartesian(ylim=c(0.55,0.6548))+
  scale_x_discrete(labels=condition_names_with_breaks)+
  theme_styling() +
  barplot_styling()

# Save
paste0(plots_directory,'accuracy.pdf') %>%
  ggsave(accuracy_plot,width=plots_width,height=plots_height,units=plots_units)

### --- --- --- --- --- --- --- --- --- ### 
### --- Participant level data: Average green votes observed by bias color and condition --- ###
### --- --- --- --- --- --- --- --- --- ### 

participant_level_data = e2_data %>%
  subset(is_cloned==F) %>%
  summarize_observed_data(c('participant_id','randomization_color','condition')) %>%
  exclude_asocial_data() %>%
  arrange(condition,-social_info_green) %>%
  mutate(
    observed_green_votes = social_info_green * 8,
    green_ordering = rep(1:(n()/2),2)
  )

participant_observations = participant_level_data %>%
  ggplot(aes(x=green_ordering,y=observed_green_votes/8,fill=randomization_color)) +
  geom_bar(stat='identity')+
  facet_grid(cols = vars(condition)) +
  coord_cartesian(ylim=c(0.31,0.82),expand=c(0,0))+
  scale_fill_manual(name='Bias color', labels=c('Blue','Green'), 
                    values = plot_colors$blue_green_colors)+
  labs(y='Proportion of observed judgements green') + 
  theme_styling() +
  theme(
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank(),
    strip.background = element_blank(),
    legend.position = 'none',
    panel.spacing =unit(1.25, "lines"),
    panel.grid.major = element_line(colour = "#ededed"),
    strip.text = element_blank())

# Save
paste0(plots_directory,'participant_observations.pdf') %>%
  ggsave(participant_observations,width=25,height=15,units=plots_units)

### --- --- --- --- --- --- --- --- ### 
### --- Scatter plot: participant biases and how often their were transmitted  --- ###
### --- --- --- --- --- --- --- --- ###

participant_bias_plot = e2_data %>%
  subset(is_cloned==F) %>%
  subset(impressions>0) %>%
  group_by(participant_id) %>%
  summarize(
    bias = unique(estimated_bias),
    impressions = unique(impressions),
    condition = unique(condition),
    randomization_color = unique(randomization_color)
  ) %>% 
  ggplot(aes(x=bias,y=impressions/128,fill=randomization_color)) +
  geom_hline(yintercept=1,linetype='dashed') +
  geom_vline(xintercept=0,linetype='dashed') +
  geom_point(shape=21,color='#5e5e5e',size=4,alpha=0.6) +
  # scale_y_continuous(breaks=c(60,90,120,150))+
  scale_fill_manual(name='Bias color', labels=c('Blue','Green'), 
                    values = plot_colors$blue_green_colors)+
  labs(y='Number of choices transmitted',x='Estimated green bias') +
  theme_styling() +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_line(colour = "#ededed"),
    axis.ticks = element_blank(),
    legend.position='none'
  )

bias_densities = e2_data %>%
  subset(is_cloned==F) %>%
  subset(impressions>0) %>%
  group_by(participant_id) %>%
  summarize(
    bias = unique(estimated_bias),
    randomization_color = unique(randomization_color)
  ) %>% 
  ggplot(aes(x = bias, fill = randomization_color)) + 
  geom_density(alpha = 0.6,color='#5e5e5e',) + 
  scale_fill_manual(name='Bias color', labels=c('Blue','Green'), 
                    values = plot_colors$blue_green_colors)+
  theme_void() + 
  theme(legend.position = "none")

impression_densities = e2_data %>%
  subset(is_cloned==F) %>%
  subset(impressions>0) %>%
  group_by(participant_id) %>%
  summarize(
    impressions = unique(impressions),
    randomization_color = unique(randomization_color)
  ) %>% 
  ggplot(aes(x = impressions, fill = randomization_color)) + 
  geom_density(alpha = 0.6,color='#5e5e5e',) + 
  scale_fill_manual(name='Bias color', labels=c('Blue','Green'), 
                    values = plot_colors$blue_green_colors)+
  theme_void() + 
  theme(legend.position = "none") +
  coord_flip()


final_participant_plot = bias_densities + plot_spacer() + participant_bias_plot + impression_densities + 
  plot_layout(ncol = 2, nrow = 2, widths = c(4, 1), heights = c(1, 4))


# Save
paste0(plots_directory,'participant_biases.pdf') %>%
  ggsave(final_participant_plot,width=15,height=15,units=plots_units)




(5000 + 4600) / (5000 + 5000 + (6000-4600))
