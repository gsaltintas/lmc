#!/bin/bash
set -e

for REPLICATE in $(seq 1 1 3); do
    for STEP in 0; do
        for SCALE in 0.00000001 0.0000001 0.000001; do
            sbatch ./scripts/lyapunov_exponent/lyapunov_exponent_single.sh $STEP $SCALE $REPLICATE
            sleep 0.1
        done
    done
done

for REPLICATE in $(seq 1 1 3); do
    for STEP in 1950; do
        for SCALE in 0.0001 0.001 0.01; do
            sbatch ./scripts/lyapunov_exponent/lyapunov_exponent_single.sh $STEP $SCALE $REPLICATE
            sleep 0.1
        done
    done
done
