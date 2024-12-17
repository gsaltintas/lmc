#!/bin/bash
#SBATCH --time=750
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=gsa
#SBATCH --tmp=4G

njobs_at_time=15
norm="layernorm"; seed1=22; seed2=43; 
scale=0.1; lr=0.1
perturb_mode="batch"
perturb_mode="gaussian"

seeds1=(22 45 987)
seeds2=(43 66 1008)
repetition=2
cnt=0
# runs job no-start until and including job-end
start=0
end=2000

for perturb_mode in "batch" "gaussian"; do
for i in "${!seeds1[@]}"; do
if [[ $((i+1)) -gt $repetition ]]; then break; fi
  seed1=${seeds1[i]}
  seed2=${seeds2[i]}
for lr in 0.1 ; do
# for lr in 0.1 0.01; do
for perturb_step in 0 1 2 3 4 5 390 1950; do
for scale in 0.1 0.01 0.5; do
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
--cleanup_after=false --use_wandb true --group=noise-sampled-at-t --run_name=perm-$perturb_mode@${perturb_step}x$scale-lr=$lr-$repetition --project=ButterflyBatchNoise --n_models=2 --seed1=$seed1 --loader_seed1=$seed1 \
--seed2=$seed2 --loader_seed2=$seed2  \
--perturb_step=$perturb_step --perturb_inds 1 --perturb_mode=$perturb_mode --perturb_scale=$scale --sample_noise_at=perturb \
--deterministic=true --use_tqdm=false"

echo Submitted job-$cnt
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