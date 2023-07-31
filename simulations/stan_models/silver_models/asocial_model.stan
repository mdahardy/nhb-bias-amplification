data {
  int <lower=0, upper=100000> N;
  int <lower=0, upper=100000> num_participants;
  int <lower=0, upper=100000> num_problems;
  int <lower=0, upper=1> chose_green[N];
  int <lower=1, upper=num_participants> participant_index[N];
  int <lower=1, upper=num_problems> problem_index[N];
}

parameters {
  vector[num_participants] bias;
  vector[num_problems] difficulty;
}

transformed parameters {
    vector[N] og_choice_probabilities;
    vector[N] bias_0_choice_probabilities;
    for (n in 1:N){
      og_choice_probabilities[n] = exp(bias[participant_index[n]] - difficulty[problem_index[n]]) / (1 + exp(bias[participant_index[n]] - difficulty[problem_index[n]]));
      bias_0_choice_probabilities[n] = exp( - difficulty[problem_index[n]]) / (1 + exp(- difficulty[problem_index[n]]));
    }
}

model {
  bias ~ normal(0,3);
  difficulty ~ normal(0,3);
  for (n in 1:N){
       chose_green[n] ~ bernoulli_logit(bias[participant_index[n]] - difficulty[problem_index[n]]);
  }
}
