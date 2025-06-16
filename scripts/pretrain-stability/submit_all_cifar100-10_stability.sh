#!/bin/bash

## from cifar100 to cifar10
CKPT_STEPS=("191ep319st" "57ep213st" )
# CKPT_STEPS=("0ep180st" "3ep327st")
# CKPT_STEPS=("0ep180st" "3ep327st" "191ep319st" "57ep213st"  )
## layernorm version, seed=2
ROOT_="/home/mila/g/gul-sena.altintas/pretrain/trainer_fad2a6ac-25-01-28-31364"
SLURM_JOB_="5978180"
## seed=1
# ROOT_="/home/mila/g/gul-sena.altintas/pretrain/trainer_46e21419-25-01-28-51223"
# SLURM_JOB_="5978179"

CONFIG_YAML="${ROOT_}/config.yaml"
CKPT_ROOT="${ROOT_}/model1/checkpoints"

PERTURB_STEPS=(1 100 2000)
PERTURB_SCALES=(0.1 0.5 0.01)
# PERTURB_SCALES=( 0.00001 0.001  0.05  )
PERTURB_MODES=("gaussian" "batch")

## Gaussian
# echo submitting Gaussian perturbations
# for CKPT_STEP in "${CKPT_STEPS[@]}"; do
#     for PERTURB_STEP in "${PERTURB_STEPS[@]}"; do
#         for PERTURB_SCALE in "${PERTURB_SCALES[@]}"; do
#             sbatch --time 360 --dependency="${SLURM_JOB_}" --job-name="pretrain-stab-${CKPT_STEP}" \
#                 scripts/pretrain-stability/train-cifar100-from-cifar10.sh "${CONFIG_YAML}" "${CKPT_ROOT}" "${CKPT_STEP}" "${PERTURB_STEP}" "${PERTURB_SCALE}" "gaussian"
#             sleep 0.2
#         done
#     done
# done

PERTURB_SCALES=(0.00001 0.001 0.05 0.01 0.005)
## Batch
echo submitting Batch perturbations
for CKPT_STEP in "${CKPT_STEPS[@]}"; do
    for PERTURB_STEP in "${PERTURB_STEPS[@]}"; do
        for PERTURB_SCALE in "${PERTURB_SCALES[@]}"; do
            # Submit the job with the CKPT_STEP as an argument
            sbatch --time 180 --dependency="${SLURM_JOB_}" --job-name="pretrain-stab-${CKPT_STEP}" \
                scripts/pretrain-stability/train-cifar100-from-cifar10.sh "${CONFIG_YAML}" "${CKPT_ROOT}" "${CKPT_STEP}" "${PERTURB_STEP}" "${PERTURB_SCALE}" "batch"
            sleep 1
        done
    done
done
