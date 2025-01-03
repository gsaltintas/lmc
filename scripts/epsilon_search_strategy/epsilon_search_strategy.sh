#!/bin/bash

# ./scripts/epsilon_search_strategy/epsilon_search_strategy_single.sh 0 0 99 true

for REPLICATE in 1 2 3; do
    for DETERMINISTIC in false true; do
        for SCALE in 0 0.0001 0.001 0.01 0.1; do
            sbatch ./scripts/epsilon_search_strategy/epsilon_search_strategy_single.sh 0 $SCALE $REPLICATE $DETERMINISTIC
        done
    done

    for SCALE in 0 0.01 0.1 1 10; do
        sbatch ./scripts/epsilon_search_strategy/epsilon_search_strategy_single.sh 1950 $SCALE $REPLICATE false
    done
done
