#!/bin/bash
#SBATCH --time=480
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --tmp=8G
#SBATCH --job-name=cifar10_pretrain_stab
#SBATCH --begin=now+0minutes

DATASET=cifar10 # Pass dataset as argument
LR=0.1
WD=1e-4
# WD=0
SEED1=42
SEED2=42
SEED3=42


SEED1=101
SEED2=101
SEED3=101
BS=128
GRAD_ACCUM=1
MODEL="resnet50"
USE_WANDB="true"
STEPS="3000st"
STEPS="20000st"
CONFIG_YAML=$1
CKPT_ROOT=$2
CKPT_STEP=$3
PERTURB_STEP=$4
PERTURB_SCALE=$5
PERTURB_MODE=${6-gaussian}
WARMUP_RATIO=0.032
WARMUP_RATIO=0.025

# /home/mila/g/gul-sena.altintas/clean/main.py perturb --config_file /network/scratch/g/gul-sena.altintas/pretrain/trainer_d28f2652eb0fc00e8f4996950925ff69-20-01-25-07-41-301734/config.yaml --config_file /network/scratch/g/gul-sena.altintas/pretrain/trainer_f79899f1a464935a99bbc2dfa4d91289-19-01-25-21-58-220181/config.yaml --model_name resnet50-64 --ckpt_path=/network/scratch/g/gul-sena.altintas/pretrain/trainer_f79899f1a464935a99bbc2dfa4d91289-19-01-25-21-58-220181/model1-seed_1-ls_1/checkpoints/ep-200.ckpt --norm=batchnorm --dataset cifar10 --path=data/cifar10 --hflip true --random_rotation=10 --random_translate=4 --cutout=2 --optimizer=sgd --training_steps=5ep --lr_scheduler onecycle --lr 0.1 --momentum=0.9 --warmup_ratio=0.025 --batch_size=128 --log_dir=/network/scratch/g/gul-sena.altintas/finetune --enforce_new_model_dir=True --use_tqdm=true --save_freq=5ep --cleanup_after=false --use_wandb true --group=cifar10-resnet50 --run_name=resnet50-cifar10-1. --project=LMCFinetuning --n_models=2 --seed1=1 --loader_seed1=1 --seed2=1 --loader_seed2=1 --deterministic=true --n_points=5 --normalize_perturb true --perturb_inds 1 --perturb_step=1 --dont_perturb_module_patterns ".*norm.*|.*bias.*" --perturb_scale=1. --same_steps_pperturb=false

# PROJECT="LMCPretrainingStability-Cifar"
PROJECT="LMCPretrainingStability"

python main.py perturb \
    --config_file=$CONFIG_YAML \
    --model_name=resnet50-64 \
    --ckpt_path="${CKPT_ROOT}/${CKPT_STEP}.ckpt" \
    --norm=batchnorm \
    --norm=layernorm \
    --dataset $DATASET \
    --path=data/$DATASET \
    --hflip true \
    --random_rotation=10 \
    --random_translate=4 \
    --cutout=2 \
    --optimizer sgd \
    --training_steps=$STEPS \
    --lr $LR \
    --momentum=0.9 \
    --weight_decay $WD \
    --lr_scheduler onecycle \
    --warmup_ratio $WARMUP_RATIO \
    --batch_size=$BS \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --log_dir=$SCRATCH/finetune_lmc \
    --cleanup_after=false \
    --use_tqdm=true \
    --enforce_new_model_dir=true \
    --use_wandb $USE_WANDB \
    --group=$DATASET-$MODEL-$CKPT \
    --run_name=cifar10-$CKPT_STEP@$PERTURB_STEPx$PERTURB_SCALE \
    --project=$PROJECT \
    --tags=pretrain-stability \
    --n_models=2 \
    --seed1=$SEED1 \
    --loader_seed1=$SEED1 \
    --seed2=$SEED2 \
    --loader_seed2=$SEED2 \
    --deterministic=true \
    --perturb_mode=$PERTURB_MODE \
    --perturb_step=$PERTURB_STEP \
    --perturb_scale=$PERTURB_SCALE \
    --perturb_inds 1 \
    --normalize_perturb=true \
    --same_steps_pperturb=false \
    --dont_perturb_module_patterns '.*norm.*|.*bias.*' 


exit 0
    --perturb_inds 1 2 \
    --n_models=3 \
    --seed1=$SEED1 \
    --loader_seed1=$SEED1 \
    --seed2=$SEED2 \
    --loader_seed2=$SEED2 \
    --seed3=$SEED3 \
    --loader_seed3=$SEED3
