library(ggplot2)
library(dplyr)
library(ggpubr)
library(lme4)
library(facetscales)
setwd("/Users/mdhardy/Documents/princeton_research/collective_intelligence/public_repo/simulations/network_simulations")

# -- -- -- -- -- --
# -- -- -- -- -- --
# Params
# -- -- -- -- -- --
# -- -- -- -- -- --

alphas = c(1.5,2,3)
network_sizes = c(64,128,256)

# -- -- -- -- -- --
# -- -- -- -- -- --
# Load helpers
# -- -- -- -- -- --
# -- -- -- -- -- --

source('../../analyses/plotting_utils.R')
source('../../analyses/utils.R')

# Helper functions
read_data = function(network_size, alpha,num_replications=100){
    # Logging
    logging_string = paste0("Reading in data for network_size=", network_size, " and alpha=", alpha)
    print(logging_string)
    # Read in data
    alpha_str = sprintf("%.1f", alpha)
    data = read.csv(paste0("../../data/network_simulations/results/network_size=", network_size, "-alpha=", alpha_str, "-num_replications=", num_replications, ".csv"))
    # Add network size and alpha
    data$network_size = network_size
    data$alpha = alpha
    return(data)
}

se = function(var){
  return(sd(var)/sqrt(length(var)))
}

# Read in data and rbind all the data together
data = lapply(alphas, function(alpha) {
  lapply(network_sizes, function(network_size) {
    read_data(network_size, alpha)
  }) %>% do.call(rbind, .)
}) %>% do.call(rbind, .)


# Get means bias by network size, alpha, and resampling
grouped_data = data %>%
  group_by(network_size, alpha, resampling, iteration) %>%
  summarize(
    mean_bias = mean(chose_bias=="True"),
    mean_accuracy = mean(chose_correct),
    bias_se = se(chose_bias=="True"),
    accuracy_se = se(chose_correct)
  ) %>%
  mutate(
    condition = case_when(
      iteration == 0 ~ 'Asocial bias',
      resampling == 'True' ~ 'Social resampling',
      T ~ 'Social bias'
    ),
    remove = iteration == 0 & resampling == "False"
  ) %>%
  subset(remove==F)
  

data_grouped_by_network = data %>%
  group_by(network_size, alpha, resampling, iteration,simulation_num) %>%
  summarize(
    mean_bias = mean(chose_bias=="True"),
    mean_accuracy = mean(chose_correct),
    bias_se = se(chose_bias=="True"),
    accuracy_se = se(chose_correct)
  ) %>%
  mutate(
    condition = ifelse(resampling=='True','Social resampling','Social bias'),
    network_identifier = paste(simulation_num,network_size,alpha,resampling)
  )


# Now make the plots look good. Add faint lines for each replication and make the lines thicker for the mean
# Make colors line up with condition colors from the paper
    
    
# -- -- -- -- -- --
# Plot styling
# -- -- -- -- -- --

# Condition colors
condition_colors = get_plot_colors(2)$condition_colors

# Condition shapes
condition_shapes = c('Asocial bias' = 16,'Social bias' = 15,'Social resampling'= 17)

facet_labeller = function(variable, value) {
  if (variable == "network_size") {
    return(paste("Network size:", value))
  }
  if (variable == 'alpha'){
    return(paste("Alpha:", value))
  }
  return(value)
}

# Change iteration so it starts at 1
data_grouped_by_network = data_grouped_by_network %>% mutate(iteration = iteration+1)
grouped_data = grouped_data %>% mutate(iteration = iteration+1)

# Plot bias by generation
# Size: 6.5x10.5
bias_plot = grouped_data %>%
  ggplot(aes(x=iteration,y=mean_bias,group=condition,color=condition,shape = condition,fill=condition)) +
  geom_line(
    data = data_grouped_by_network,
    aes(x=iteration,y=mean_bias,group=network_identifier,color=condition),
    alpha = 0.04) +
  geom_point(size=3) +
  geom_errorbar(aes(ymin=mean_bias-bias_se, ymax=mean_bias+bias_se), size=0.5,width=0.5) +
  labs(x='Iteration',y='Proportion of bias-aligned choices') +
  # scale_y_continuous(breaks=c(0.52,0.55,0.58,0.61,0.64,0.67),
  #                    limits=c(0.52,0.6703),expand=c(0,0))+
  scale_x_continuous(breaks = 1:8) +
  scale_color_manual("legend", values = condition_colors)+
  scale_fill_manual("legend", values = condition_colors)+
  scale_shape_manual("legend", values = condition_shapes)+
  theme_classic()  +
  theme(
    strip.background = element_blank(),
    panel.grid.major = element_line(colour = "#ededed"),
    legend.position='none',
    panel.spacing = grid::unit(10, "points")
  ) + 
  facet_grid(alpha~network_size,labeller=facet_labeller)

ggsave("../../paper/plots/network_plots/bias.pdf", bias_plot, width=9,height=4.5)

# Plot accuracy by generation
accuracy_plot = grouped_data %>%
  ggplot(aes(x=iteration,y=mean_accuracy,group=condition,color=condition,shape = condition,fill=condition)) +
  geom_line(
    data = data_grouped_by_network,
    aes(x=iteration,y=mean_accuracy,group=network_identifier,color=condition),
    alpha = 0.04) +
  geom_point(size=3) +
  geom_errorbar(aes(ymin=mean_accuracy-accuracy_se, ymax=mean_accuracy+accuracy_se), size=0.5,width=0.5) +
  labs(x='Iteration',y='Proportion of correct choices') +
  scale_x_continuous(breaks = 1:8) +
  scale_color_manual("legend", values = condition_colors)+
  scale_fill_manual("legend", values = condition_colors)+
  scale_shape_manual("legend", values = condition_shapes)+
  theme_classic()  +
  theme(
    strip.background = element_blank(),
    panel.grid.major = element_line(colour = "#ededed"),
    legend.position='none',
    panel.spacing = grid::unit(10, "points")
  ) + 
  facet_grid(alpha~network_size,labeller=facet_labeller)


ggsave("../../paper/plots/network_plots/accuracy.pdf", accuracy_plot, width=9,height=4.5)


# Run analyses

results = data.frame()
means = data.frame()
for (alpha_i in (unique(data$alpha))){
  for (network_size_i in (unique(data$network_size))){
    print(paste0('Alpha: ',alpha_i,' Network size: ',network_size_i))
    data_subset = data %>%
      subset(iteration != 0) %>%
      subset(alpha == alpha_i) %>%
      subset(network_size == network_size_i) %>%
      mutate(
        chose_utility = chose_bias == 'True',
        chose_correct = as.logical(chose_correct)
      )
      
    # Bias
    unrestricted_bias = glm(chose_utility ~ resampling + color,data=data_subset,family='binomial')
    restricted_bias = glm(chose_utility ~ color,data=data_subset,family='binomial')
    bias_anova = anova(restricted_bias,unrestricted_bias,test='Chisq')
    bias_anova_p_value = bias_anova[2,5]
    bias_anova_chi_square = bias_anova[2,4]
    bias_df = data.frame(
      'alpha'=alpha_i,
      'network_size'=network_size_i,
      'p_value'=bias_anova_p_value,
      'chi_square'=bias_anova_chi_square,
      'dv'='bias'
    )
    results = rbind(results,bias_df)

    # Accuracy
    unrestricted_accuracy = glm(chose_correct ~ resampling + color,data=data_subset,family='binomial')
    restricted_accuracy = glm(chose_correct ~ color,data=data_subset,family='binomial')
    accuracy_anova = anova(restricted_accuracy,unrestricted_accuracy,test='Chisq')
    accuracy_anova_p_value = accuracy_anova[2,5]
    accuracy_anova_chi_square = accuracy_anova[2,4]
    accuracy_df = data.frame(
      'alpha'=alpha_i,
      'network_size'=network_size_i,
      'p_value'=accuracy_anova_p_value,
      'chi_square'=accuracy_anova_chi_square,
      'dv'='accuracy'
    )
    results = rbind(results,accuracy_df)
  }
}

# Round chi_square to 1
results = results %>% mutate(
  chi_square = round(chi_square,1)
)
print(results) 

data %>%
  subset(iteration != 0) %>%
  group_by(alpha,network_size,resampling) %>%
  summarize(
    bias = mean(chose_bias == 'True'),
    accuracy = mean(chose_correct)
  )



