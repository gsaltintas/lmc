#!/bin/bash
#SBATCH --time=249
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --tmp=8G
#SBATCH --job-name=resnet20-cinic
#SBATCH --begin=now+0minutes

seeds1=(22 45 987)
seeds2=(43 66 1008)
repetition=3
NORM=layernorm
# NORM=batchnorm
DATASET=cinic10
# DATASET=cinic10_wo_cifar10
LR=0.1
WD=1e-4
SEED1=1
BS=256
MODEL="resnet20-64"
USE_WANDB="true"

python main.py train \
    --model_name $MODEL \
        --norm=$NORM \
    --dataset $DATASET \
        --path=data \
        --hflip true \
        --random_rotation=10 \
        --random_translate=4 \
        --cutout=4 \
        --download=true \
    --optimizer=sgd \
        --training_steps=200ep \
        --lr_scheduler onecycle \
        --lr $LR \
        --momentum=0.9 \
        --warmup_ratio=0.025 \
        --batch_size=$BS \
    --log_dir=$SCRATCH/pretrain \
    --cleanup_after=false \
    --use_wandb $USE_WANDB \
        --group=$DATASET-$MODEL \
        --run_name=$MODEL-$DATASET-$NORM \
        --project=ImagePreTraining \
    --use_tqdm=true \
    --n_models=1 \
        --seed1=$SEED1 \
        --loader_seed1=$SEED1 \
        --deterministic=false 
