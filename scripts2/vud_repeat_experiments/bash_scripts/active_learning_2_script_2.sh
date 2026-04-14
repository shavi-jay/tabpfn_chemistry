#!/bin/bash

# Define dataset and endpoint combinations admet i=0,1,2,3,4 and potency i=0,1
declare -a datasets=("potency")
declare -a endpoints_admet=(0 1 2 3 4)
declare -a endpoints_potency=(0 1)

# Define uncertainty types
declare -a uncertainty_methods=("total" "random" "epistemic")

num_seeds=10

# Loop over datasets, endpoints, uncertainty types, and seeds
for dataset in "${datasets[@]}"; do
    if [ "$dataset" == "admet" ]; then
        endpoints=("${endpoints_admet[@]}")
    else
        endpoints=("${endpoints_potency[@]}")
    fi
    for endpoint in "${endpoints[@]}"; do
        for uncertainty_method in "${uncertainty_methods[@]}"; do
            for seed in $(seq 0 $((num_seeds - 1))); do
                echo "Running active learning for dataset: $dataset, endpoint: $endpoint, uncertainty method: $uncertainty_method, seed: $seed"
                python scripts2/vud_repeat_experiments/vud_active_learning_2.py \
                    --dataset_name "${dataset}_${endpoint}" \
                    --uncertainty_method "$uncertainty_method" \
                    --active_learning_seed "$seed" \
                    --save_path "results/active_learning/active_learning_2/${uncertainty_method}"
            done
        done
    done
done