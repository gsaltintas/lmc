#!/bin/bash
#SBATCH --time=750
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=test_deterministic
#SBATCH --tmp=4G


NORM="layernorm"
SEED=22
REPLICATE=1
PERTURB_STEP=1
SCALE=0
PERTURB_TYPE="gaussian"

python main.py perturb  \
    --training_steps=50ep  \
    --model_name resnet20-32  \
    --norm=$NORM  \
    --path=$SLURM_TMPDIR/data  \
    --dataset=cifar10  \
    --hflip=true  \
    --random_rotation=10  \
    --random_crop=false  \
    --lr_scheduler=triangle  \
    --lr=0.1   \
    --warmup_ratio=0.02  \
    --optimizer=sgd  \
    --momentum=0.9  \
    --save_early_iters=true  \
    --log_dir=$HOME/scratch/butterfly/epsilon-search-strategy  \
    --cleanup_after=false  \
    --use_wandb=true  \
    --run_name=$PERTURB_TYPE-$PERTURB_STEP-$SCALE-lr$lr-$REPLICATE  \
    --project=butterfly-v2  \
    --n_models=2  \
    --seed1=$SEED  \
    --seed2=$SEED  \
    --loader_seed1=$SEED  \
    --loader_seed2=$SEED  \
    --perturb_step=$PERTURB_STEP  \
    --perturb_inds=1  \
    --perturb_mode=$PERTURB_TYPE  \
    --perturb_scale=$SCALE  \
