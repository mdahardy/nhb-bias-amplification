data {
  int <lower=0, upper=100000> N;
  int <lower=0, upper=100000> num_participants;
  int <lower=0, upper=1> chose_green[N];
  int <lower=1, upper=num_participants> participant_index[N];
  int <lower=1, upper=4> green_factor[N];
  vector[N] negative_green;
}

parameters {
  real absolute_bias;
  real bias_standard_deviation;
  vector[num_participants] participant_bias;
  vector[4] green_dummy;
  real intercept;
}

model {
  absolute_bias ~ normal(0,3);
  bias_standard_deviation ~ lognormal(0,2);
  participant_bias ~ normal(absolute_bias,bias_standard_deviation);
  green_dummy ~ normal(0,20);
  intercept ~ normal(0,20);
  chose_green ~ bernoulli_logit(intercept  + (negative_green .* participant_bias[participant_index])  + green_dummy[green_factor]); 
}






