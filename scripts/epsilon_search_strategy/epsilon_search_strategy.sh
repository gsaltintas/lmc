#!/bin/bash
set -e

BRANCH=exp/epsilon_search_strategy

# make sure code is consistent
git checkout $BRANCH
git log -1 --oneline

# for REPLICATE in 1 2 3; do
#     for DETERMINISTIC in false true; do
#         for SCALE in 0.00001 0.00002 0.00005 0.0002 0.0005 0.002 0.005; do
#             sbatch ./scripts/epsilon_search_strategy/epsilon_search_strategy_single.sh 0 $SCALE $REPLICATE $DETERMINISTIC
#             sleep 1
#         done
#     done

#     # for SCALE in 0 0.01 0.1 1 10; do
#     #     sbatch ./scripts/epsilon_search_strategy/epsilon_search_strategy_single.sh 1950 $SCALE $REPLICATE false
#     #     sleep 1
#     # done
# done

#  6 7 8 9 10
#  # 0.00000001 0.0000001 0.000001
for REPLICATE in 1 2 3 4 5; do
    for SCALE in 0.000000000001 0.00000000001 0.0000000001 0.000000001; do
        sbatch ./scripts/epsilon_search_strategy/epsilon_search_strategy_single.sh 0 $SCALE $REPLICATE "true"
        sleep 1
    done
done
