#!/bin/bash

# ./scripts/epsilon_search_strategy/epsilon_search_strategy_single.sh 0 0 99 true

for REPLICATE in 1 2; do
    for SCALE in 0.000001 0.0001 0.01 0.1 1; do
        for STEP in 0 1950; do
            sbatch ./scripts/small_butterfly/small_butterfly_single.sh $STEP $SCALE $REPLICATE resnet20-64 128 0.1 0.02 ".*norm.*" "shallow-wide"
            sbatch ./scripts/small_butterfly/small_butterfly_single.sh $STEP $SCALE $REPLICATE resnet32-8 128 0.1 0.02 ".*norm.*" "deep-narrow"
            sbatch ./scripts/small_butterfly/small_butterfly_single.sh $STEP $SCALE $REPLICATE resnet20-32 128 0.001 0.02 ".*norm.*" "lr-0.001"
            sbatch ./scripts/small_butterfly/small_butterfly_single.sh $STEP $SCALE $REPLICATE resnet20-32 16384 0.1 0.02 ".*norm.*" "batch-16384"
            sbatch ./scripts/small_butterfly/small_butterfly_single.sh $STEP $SCALE $REPLICATE resnet20-32 128 0.1 0 ".*norm.*" "no-warmup"

            sbatch ./scripts/small_butterfly/small_butterfly_single.sh $STEP $SCALE $REPLICATE resnet20-32 128 0.1 0.02 '^((?!norm).)*$' "norm-only"
            for BLOCK in 0 1 2; do
                for SUBBLOCK in 0 1 2; do
                    sbatch ./scripts/small_butterfly/small_butterfly_single.sh $STEP $SCALE $REPLICATE resnet20-32 128 0.1 0.02 '^((?!block'$BLOCK.$SUBBLOCK'.conv).)*$' "block$BLOCK.$SUBBLOCK-only"
                done
            done
        done
    done
done