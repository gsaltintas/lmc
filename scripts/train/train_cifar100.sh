#!/bin/bash
#SBATCH --time=249
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name=resnet50

seeds1=(22 45 987)
seeds2=(43 66 1008)
repetition=3
NORM=batchnorm
NORM=layernorm
DATASET=cifar100
LR=0.1
WD=1e-4
SEED1=1
SEED1=2
SEED1=3
STEPS="75000st"
WANDB=true
# WARMUP_RATIO=0.032
# CUTOUT=2
CUTOUT=4
WARMUP_RATIO=0.025

python main.py train \
    --model_name resnet50-64 \
        --norm=$NORM \
    --dataset $DATASET \
        --path=data/cifar100 \
        --hflip true \
        --random_rotation=10 \
        --random_translate=4 \
        --cutout=$CUTOUT \
    --optimizer=sgd \
        --training_steps=$STEPS \
        --lr_scheduler onecycle \
        --lr $LR \
        --momentum=0.9 \
        --warmup_ratio=$WARMUP_RATIO \
        --batch_size=128 \
    --log_dir=/network/scratch/g/gul-sena.altintas/pretrain \
    --cleanup_after=false \
    --use_wandb $WANDB \
        --group=cifar100-resnet50-new \
        --run_name=resnet50-cifar100-$NORM \
        --project=ImagePreTraining \
    --save_freq=1500st \
    --save_specific_steps="1st,180st,1ep,2000st" \
    --use_tqdm=true \
    --n_models=1 \
        --seed1=$SEED1 \
        --loader_seed1=$SEED1 \
        --deterministic=false  
