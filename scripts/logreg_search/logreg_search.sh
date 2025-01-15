#!/bin/bash
set -e

#logreg_search_single.sh PERTURB_STEP THRESHOLD INIT_SCALE N_RUNS SEED
sbatch ./scripts/logreg_search/logreg_search_single.sh 0 0.1 0.0001 10 100
sbatch ./scripts/logreg_search/logreg_search_single.sh 1950 0.1 1 10 101
