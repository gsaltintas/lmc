#!/bin/bash
#SBATCH --time=750
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=gsa


### don't perturb some parameters when perturbing the model

use_wandb=true
njobs_at_time=25
norm="layernorm"; seed1=22; seed2=43; 
scale=0.1; lr=0.1
perturb_mode="batch"
perturb_mode="gaussian"

seeds1=(22 45 987)
seeds2=(43 66 1008)
repetition=3
cnt=0
# runs job no-start until and including job-end
start=0
end=40
# Don't perturb bias and norm parameters")
# dont_perturb_settings=( "'.*\.bias$' '.*\.norm2\..*'")
# Only perturb activations (perturb norm parameters))
dont_perturb_settings=( "'.*\.bias$' '.*\.conv\..*' '.*\.fc\..*'  "  "'.*\.conv\..*' '.*\.fc\..*'  " )
project=BatchNoisePerturbNorm

for dont in "${!dont_perturb_settings[@]}"; do
dont=${dont_perturb_settings[dont]}
for perturb_mode in "batch" ; do
# for perturb_mode in "batch" "gaussian"; do
for i in "${!seeds1[@]}"; do
if [[ $((i+1)) -gt $repetition ]]; then break; fi
  seed1=${seeds1[i]}
  seed2=${seeds1[i]}
for lr in 0.1 ; do
# for lr in 0.1 0.01; do
for perturb_step in 0 1 2 3 4 5; do
for scale in 0.1 0.5; do
((cnt++))
if [[ $cnt -lt $start ]]; then echo Skipping job-$cnt; continue; fi
if [[ $cnt -gt $end ]]; then echo Skipping job-$cnt; continue; fi
while [[ $(squeue -u $USER | wc -l) -ge $njobs_at_time ]]; do sleep 1; done

sbatch --time=120 --dependency=5773567 --mem-per-cpu=2G --cpus-per-gpu=4 --tmp=4G --job-name="$cnt" --gres=gpu:rtx8000:1 --wrap="python main.py perturb --training_steps=50ep \
--model_name resnet20-32 --norm=$norm \
--dataset cifar10 --hflip true --random_rotation=10 --random_crop=false \
--lr_scheduler triangle --lr $lr  --warmup_ratio=0.02 \
--optimizer=sgd --momentum=0.9 \
--save_early_iters=true --log_dir=/network/scratch/g/gul-sena.altintas/perm/butterfly \
--cleanup_after=false --use_wandb $use_wandb --group=noise-sampled-at-t --run_name=DontBiasNorm-$perturb_mode-@${perturb_step}x$scale-lr=$lr-$i --project=$project --n_models=2 --seed1=$seed1 --loader_seed1=$seed1 \
--seed2=$seed2 --loader_seed2=$seed2  \
--perturb_step=$perturb_step --perturb_inds 1 --perturb_mode=$perturb_mode --perturb_scale=$scale --sample_noise_at=perturb \
--deterministic=true --use_tqdm=false --dont_perturb_module_patterns ${dont}"

echo Submitted job-$cnt
done 
done 
done
done
done
done


# # ## baseline
# norm="layernorm"; lr=0.1; seed1=22
# python main.py train --training_steps=50ep \
# --model_name resnet20-32 --norm=$norm \
# --dataset cifar10 --hflip true --random_rotation=10 --random_crop=false \
# --lr_scheduler triangle --lr $lr  --warmup_ratio=0.02 \
# --optimizer=sgd --momentum=0.9 \
# --save_early_iters=true --log_dir=/network/scratch/g/gul-sena.altintas/perm/butterfly \
# --cleanup_after=false --use_wandb true --group=longer --run_name=baseline-lr=$lr-$repetition --project=Perm-Stability --n_models=1 --seed1=$seed1 --loader_seed1=$seed1

PERTURB_TYPE="batch"
# SCALE=0.1
# PERTURB_STEP=800
# PERTURB_STEP=380
# MODEL="resnet20-32"
# NORM="layernorm"
# DETERMINISTIC=false
# python main.py perturb  \
#     --project="$SSETUP_PROJECT_NAME-$SSETUP_EXP_NAME"  \
#         --run_name=$RUN_NAME  \
#         --path=$SLURM_TMPDIR/data/$DATASET  \
#         --log_dir=$SSETUP_OUTPUT_DIR  \
#         --save_early_iters=true  \
#         --cleanup_after=false  \
#         --use_wandb=true  \
#         --zip_and_save_source=false  \
#     --model_name=$MODEL  \
#         --norm=$NORM  \
#     --dataset=cifar10  \
#         --hflip=true  \
#         --random_rotation=10  \
#         --random_crop=false  \
#     --optimizer=sgd  \
#         --training_steps=5ep  \
#         --lr=0.1   \
#         --lr_scheduler=triangle  \
#         --warmup_ratio=0.02  \
#         --momentum=0.9  \
#     --n_models=2  \
#         --perturb_mode=$PERTURB_TYPE  \
#         --perturb_scale=$SCALE  \
#         --perturb_step=$PERTURB_STEP  \
#         --perturb_inds=1  \
#     --deterministic=$DETERMINISTIC  \
#         --seed1=$SEED  \
#         --seed2=$SEED  \
#         --loader_seed1=$SEED  \
#         --loader_seed2=$SEED  \
#         --perturb_seed1=$SEED  