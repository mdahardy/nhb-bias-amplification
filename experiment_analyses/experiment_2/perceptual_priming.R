### --- Script for running the perceptual priming analyses for experiment 2 --- ###
library(lme4)
library(dplyr)
source('utils.R') # helper functions

# -- -- -- -- -- -- -- -- -- -- - -- #
# Load data
# -- -- -- -- -- -- -- -- -- -- - -- #

tie_data =  load_e2_data() %>%
  subset(social_info_green == 0.5) %>%
  subset(condition %in% c('social_bias','social_resampling'))

base_dir = 'stats/e2_perceptual_priming'

# -- -- -- -- -- -- -- -- -- -- - -- #
# Write summary statistics (means and ns)
# -- -- -- -- -- -- -- -- -- -- - -- #
summaries_dir = paste(base_dir,'summaries',sep='/')

# Write overall summaries
write_summary_stats(tie_data,'chose_green',summaries_dir)

# Write summaries at specific variable levels
group_and_write_summary_stats(tie_data,'randomization_color','green','chose_green',summaries_dir)
group_and_write_summary_stats(tie_data,'randomization_color','blue','chose_green',summaries_dir)
group_and_write_summary_stats(tie_data,'green_shown',T,'chose_green',summaries_dir)
group_and_write_summary_stats(tie_data,'green_shown',F,'chose_green',summaries_dir)

# -- -- -- -- -- -- -- -- -- -- - -- #
# Write model statistics
# -- -- -- -- -- -- -- -- -- -- - -- #

# Write stats
stats_dir = paste(base_dir,'stats',sep='/')
formula_string = 'chose_green ~ randomization_color + green_shown + proportion_green'
fit_tie_model_and_write(tie_data,formula_string,stats_dir)