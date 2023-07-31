#!/bin/bash

# Define the values for network_size and alpha
network_sizes=(64 128 256)
alphas=(1.5 2 3)
num_replications=100
num_iterations=8

# Loop over network_size values
for network_size in "${network_sizes[@]}"; do
  # Loop over alpha values
  for alpha in "${alphas[@]}"; do
    echo "Running script with network_size: $network_size and alpha: $alpha"

    # Run your script with the current network_size and alpha values
    # Replace 'your_script.sh' with the name of the script you want to run
    python3 network_simulations.py "$network_size" "$alpha" "$num_replications" "$num_iterations"
  done
done