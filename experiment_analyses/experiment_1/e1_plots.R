### --- Script for generating and saving the figures for Experiment 1 --- ###

library(ggplot2)
library(dplyr)
library(ggsignif)# Helper functions
source('utils.R')
source('plotting_utils.R')

### --- --- --- --- ### 
### --- Set up --- ###
### --- --- --- --- ### 

# The units for defining the dimensions of the plots: cm is centimeters
plots_units = 'cm'
# The height in plots_units for saving the figures
plots_height = 14
# The width in plots_units for saving the figures
plots_width = 14

# Add line breaks for better styling
condition_names_with_breaks = c('Asocial\ncontrol','Social\ncontrol',
                                'Asocial\nmotivated','Social\nmotivated')

# The directory where the plots should be saved
plots_directory = 'plots/e1/'
# Make this directory if it doesn't exist
make_directory(plots_directory)

### --- --- --- --- ### 
### --- Load data --- ###
### --- --- --- --- ### 

e1_data = load_e1_data(include_cloned = T) %>% 
  pretty_condition_names()


# -- -- -- -- -- --
# Plot styling
# -- -- -- -- -- --

# Condition colors
condition_colors = get_plot_colors(1)$condition_colors

# Condition shapes
condition_shapes = c(
  'Asocial control' = 23,
  'Asocial motivated' = 21,
  'Social control' = 25,
  'Social motivated' = 22)

### --- --- --- --- --- --- --- --- ### 
### --- Make bias plot and save --- ###
### --- --- --- --- --- --- --- --- ### 

bias_plot = e1_data %>%
  subset(is_cloned == F) %>%
  summarize_choice_data('condition') %>%
  refactor_condition(c('Asocial control','Social control','Asocial motivated','Social motivated')) %>%
  ggplot(aes(condition,y=bias,fill=condition)) + 
  geom_bar(stat='identity') +
  scale_fill_manual("legend", values = condition_colors)+
  scale_x_discrete(labels=condition_names_with_breaks) +
  geom_errorbar(aes(ymin=bias-bias_se,ymax=bias+bias_se),width=0.2) +
  geom_signif(y_position = c(0.654,0.644,0.634,0.5875,0.5775), xmin = c(1,2,3,1,2), 
               xmax = c(4,4,4,3,3), annotation = rep('***',5),
               textsize=7,vjust=0.5) +
  ylab('Proportion of biased choices') + 
  coord_cartesian(ylim=c(0.5,0.656)) +
  theme_styling() +
  barplot_styling()

# Save
paste0(plots_directory,'bias.pdf') %>%
  ggsave(bias_plot,width=plots_width,height=plots_height,units=plots_units)

### --- --- --- --- --- --- --- --- --- ### 
### --- Make accuracy plot and save --- ###
### --- --- --- --- --- --- --- --- --- ### 

# Accuracy bar plot
accuracy_plot = e1_data %>%
  mutate(true_color = ifelse(proportion_green>0.5,'green','blue'),
         matched = randomization_color == true_color) %>%
  subset(is_cloned == F) %>%
  summarize_choice_data(c('condition','matched')) %>%
  refactor_condition(c('Asocial control','Social control','Asocial motivated','Social motivated')) %>%
  ggplot(aes(x=condition,y=accuracy,fill=condition)) + 
  geom_bar(stat='identity') +
  scale_fill_manual("legend", values = condition_colors)+
  scale_x_discrete(labels=condition_names_with_breaks) +
  #geom_errorbar(aes(ymin=accuracy-accuracy_se,ymax=accuracy+accuracy_se),width=0.2) +
  # geom_signif(y_position = c(0.6675,0.6675,0.6735,0.6795), xmin = c(1,2.05,3,1),
  #            xmax = c(1.95,3,4,4), annotation = rep('***',4),
  #            textsize=7,vjust=0.5) +
  ylab('Proportion of correct choices') + 
  #coord_cartesian(ylim=c(0.59,0.6825))+
  facet_grid(rows=vars(matched))+
  theme_styling() +
  barplot_styling()

# Save
paste0(plots_directory,'accuracy.pdf') %>%
  ggsave(accuracy_plot,width=plots_width,height=plots_height,units=plots_units)

### --- --- --- --- --- --- --- --- ### 
### --- Bias plot by generation --- ###
### --- --- --- --- --- --- --- --- ###

bias_by_generation = e1_data %>%
  summarize_choice_data(c('condition','generation')) %>%
  mutate(alpha_scale=ifelse(condition=='Asocial control','Dark','Light'),
         generation = generation+1) %>%
  ggplot(aes(x=generation,y=bias,fill=condition,color=condition,shape=condition)) +
  geom_ribbon(aes(ymin=bias-bias_se, ymax=bias+bias_se,alpha=alpha_scale),colour=NA) +
  geom_line(size=1)+
  geom_point(size=4.95,color='black') +
  labs(x='Generation',y="Proportion of biased choices")+
  scale_alpha_manual(values=c('Dark'=0.55,'Light'=0.2))+
  scale_color_manual("legend", values = condition_colors)+
  scale_fill_manual("legend", values = condition_colors)+
  scale_shape_manual("legend", values = condition_shapes)+
  scale_x_continuous(breaks = 1:8,expand=c(0,0)) +
  #scale_y_continuous(breaks = c(0.52,0.55,0.58)) +
  theme_styling() +
  theme(strip.background = element_blank(),
        strip.text.x = element_blank(),
        panel.grid.major = element_line(colour = "#ededed"),
        legend.position='none')

# Save
paste0(plots_directory,'bias_by_generation.pdf') %>%
  ggsave(bias_by_generation,width=16,height=13,units=plots_units)


### --- --- --- --- --- --- --- --- ### 
### --- Accuracy plot by generation --- ###
### --- --- --- --- --- --- --- --- ###

accuracy_by_generation = e1_data %>%
  summarize_choice_data(c('condition','generation')) %>%
  mutate(alpha_scale=ifelse(condition=='Asocial control','Dark','Light'),
         generation=generation+1) %>%
  ggplot(aes(x=generation,y=accuracy,fill=condition,color=condition,shape=condition)) +
  geom_ribbon(aes(ymin=accuracy-accuracy_se, ymax=accuracy+accuracy_se,alpha=alpha_scale),colour=NA) +
  geom_line(size=1)+
  geom_point(size=4.95,color='black') +
  labs(x='Generation',y="Proportion of correct choices")+
  scale_alpha_manual(values=c('Dark'=0.55,'Light'=0.2))+
  scale_color_manual("legend", values = condition_colors)+
  scale_fill_manual("legend", values = condition_colors)+
  scale_shape_manual("legend", values = condition_shapes)+
  scale_x_continuous(breaks = 1:8,expand=c(0,0)) +
  scale_y_continuous(breaks = c(0.58,0.62,0.66,0.7)) +
  theme_styling() +
  theme(strip.background = element_blank(),
        strip.text.x = element_blank(),
        panel.grid.major = element_line(colour = "#ededed"),
        legend.position='none')


# Save
paste0(plots_directory,'accuracy_by_generation.pdf') %>%
  ggsave(accuracy_by_generation,width=16,height=13,units=plots_units)


