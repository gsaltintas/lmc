#!/bin/bash

# List of GLUE datasets, Omitting "wnli"
DATASETS=("sst2" "cola" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte" )

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do
    # Submit the job with the dataset as an argument
    sbatch --job-name="glue_${DATASET}" scripts/finetune/finetune_glue.sh "${DATASET}"
    # sbatch --job-name="glue_${DATASET}" scripts/finetune/finetune_glue.sh "${DATASET}"
    
    # Optional: add a small delay between submissions
    sleep 2
done