#!/bin/bash
#SBATCH --time=750
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=epsilon_search_strategy
#SBATCH --tmp=4G


# example usage: ./scripts/epsilon_search_strategy/epsilon_search_strategy_single.sh 0 0 99 true

LOGDIR=$HOME/scratch/butterfly/epsilon_search_strategy

PERTURB_STEP=$1
SCALE=$2
REPLICATE=$3
DETERMINISTIC=$4

MODEL=resnet20-32
DATASET="cifar10"
NORM="layernorm"
PERTURB_TYPE="gaussian"
SEED=$REPLICATE
RUN_NAME="$PERTURB_TYPE-p$PERTURB_STEP-s$SCALE-r$REPLICATE-d$DETERMINISTIC"

echo $RUN_NAME

source $HOME/ssetup-uv.sh $DATASET

mkdir -p $LOGDIR

python main.py perturb  \
    --training_steps=50ep  \
    --model_name=$MODEL  \
    --norm=$NORM  \
    --path=$SLURM_TMPDIR/data/$DATASET  \
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
    --log_dir=$LOGDIR  \
    --cleanup_after=false  \
    --use_wandb=true  \
    --run_name=$RUN_NAME  \
    --project=butterfly-epsilon_search_strategy  \
    --n_models=2  \
    --seed1=$SEED  \
    --seed2=$SEED  \
    --loader_seed1=$SEED  \
    --loader_seed2=$SEED  \
    --perturb_seed1=$SEED  \
    --perturb_step=$PERTURB_STEP  \
    --perturb_inds=1  \
    --perturb_mode=$PERTURB_TYPE  \
    --perturb_scale=$SCALE  \
    --deterministic=$DETERMINISTIC  \
