data {
  int <lower=0, upper=100000> N;
  int <lower=0, upper=100000> num_participants;
  int <lower=0, upper=100000> num_problems;
  int <lower=0, upper=1> chose_green[N];
  int <lower=1, upper=num_participants> participant_index[N];
  int <lower=1, upper=num_problems> problem_index[N];
  vector[N] green_votes;

}

parameters {
  vector[num_participants] bias;
  vector[num_problems] difficulty;
  real vote_coefficient;
}

transformed parameters {
    vector[N] og_choice_probabilities;
    vector[N] bias_0_choice_probabilities;
    for (n in 1:N){
      og_choice_probabilities[n] = exp(bias[participant_index[n]] - difficulty[problem_index[n]] + (vote_coefficient*green_votes[n])) / 
    (1 + exp(bias[participant_index[n]] -difficulty[problem_index[n]] + (vote_coefficient*green_votes[n])));
      bias_0_choice_probabilities[n] = exp(-difficulty[problem_index[n]] + vote_coefficient*green_votes[n]) / (1 + exp(-difficulty[problem_index[n]] + vote_coefficient*green_votes[n]));
    }
}

model {
  bias ~ normal(0,2);
  difficulty ~ normal(0,2);
  vote_coefficient ~ normal(0,10);
  for (n in 1:N){
    chose_green[n] ~ bernoulli_logit(bias[participant_index[n]] - difficulty[problem_index[n]] + (vote_coefficient*green_votes[n]));
  }
}






