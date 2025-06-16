#!/bin/bash

CKPT_STEPS=("191ep319st" "57ep213st" )
CKPT_STEPS=("0ep180st" "3ep327st")
CKPT_STEPS=("0ep180st" "3ep327st" "191ep319st" "57ep213st")
## layernorm version
ROOT_="/network/scratch/g/gul-sena.altintas/pretrain/train-25-01-23-50553"
# ## batchnorm version
# ROOT_="/network/scratch/g/gul-sena.altintas/pretrain/train-25-01-23-45363"
# layernorm with seed=2 ->WRONG WARMUP
# ROOT_="/network/scratch/g/gul-sena.altintas/pretrain/train-25-01-27-74810"
# layernorm with seed=3
ROOT_="/network/scratch/g/gul-sena.altintas/pretrain/trainer_a471ace1-25-01-28-98526"

CONFIG_YAML="${ROOT_}/config.yaml"
CKPT_ROOT="${ROOT_}/model1/checkpoints"

PERTURB_STEPS=(1 100 2000 10000)
PERTURB_SCALES=(0.1 0.5 0.01)
PERTURB_SCALES=(0.00001 0.001 0.05 0.01)
# PERTURB_SCALES=( 0.00001 0.001  0.05  )
PERTURB_MODES=("gaussian" "batch")

## Gaussian
echo submitting Gaussian perturbations
for CKPT_STEP in "${CKPT_STEPS[@]}"; do
    for PERTURB_STEP in "${PERTURB_STEPS[@]}"; do
        for PERTURB_SCALE in "${PERTURB_SCALES[@]}"; do
            sbatch --time 480 --dependency="5965022" --job-name="cifar10-${CKPT_STEP}" \
                scripts/pretrain-stability/train-cifar10-from-cifar100.sh "${CONFIG_YAML}" "${CKPT_ROOT}" "${CKPT_STEP}" "${PERTURB_STEP}" "${PERTURB_SCALE}" "gaussian"
            sleep 0.2
        done
    done
done

exit 0
PERTURB_SCALES=(0.00001 0.001 0.05 0.01)
## Batch
echo submitting Batch perturbations
for CKPT_STEP in "${CKPT_STEPS[@]}"; do
    for PERTURB_STEP in "${PERTURB_STEPS[@]}"; do
        for PERTURB_SCALE in "${PERTURB_SCALES[@]}"; do
            # Submit the job with the CKPT_STEP as an argument
            sbatch --time 480 --dependency="5965022" --job-name="cifar10-${CKPT_STEP}" \
                scripts/pretrain-stability/train-cifar10-from-cifar100.sh "${CONFIG_YAML}" "${CKPT_ROOT}" "${CKPT_STEP}" "${PERTURB_STEP}" "${PERTURB_SCALE}" "batch"
            sleep 1
        done
    done
done
