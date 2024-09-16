#!/bin/bash
#SBATCH --time=750
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=gsa
#SBATCH --tmp=4G


norm="layernorm"; seed1=22; seed2=43; 
scale=0.1; lr=0.1

repetition=2
cnt=0
start=2
end=1000

for lr in 0.1 0.01; do
for perturb_step in 0 1 2 3 4 5 390 1950; do
for scale in 0.01 0.1; do
((cnt++))
if [[ $cnt -lt $start ]]; then echo Skipping job-$cnt; continue; fi
if [[ $cnt -gt $end ]]; then echo Skipping job-$cnt; continue; fi
sbatch --time=120 --mem-per-cpu=2G --cpus-per-gpu=4 --tmp=4G --job-name="$cnt" --gres=gpu:rtx8000:1 --wrap="python main.py perturb --training_steps=50ep \
--model_name resnet20-32 --norm=$norm \
--dataset cifar10 --hflip true --random_rotation=10 --random_crop=false \
--lr_scheduler triangle --lr $lr  --warmup_ratio=0.02 \
--optimizer=sgd --momentum=0.9 \
--save_early_iters=true --log_dir=/network/scratch/g/gul-sena.altintas/perm/butterfly \
--cleanup_after=false --use_wandb true --run_name=gauss@${perturb_step}x$scale-lr=$lr-$repetition --project=clean --n_models=1 --seed1=$seed1 --loader_seed1=$seed1 \
--perturb_step=$perturb_step --perturb_inds 1 --perturb_mode=gaussian --perturb_scale=$scale "

echo Submitted job-$cnt
done 
done 
done
