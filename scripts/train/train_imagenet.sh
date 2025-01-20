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
# NORM=layernorm
DATASET=imagenet1k
LR=0.1
WD=1e-4
SEED1=1
STEPS=200ep
WANDB=false

python main.py train \
    --model_name resnet50-64 \
        --norm=$NORM \
    --dataset $DATASET \
        --path=data/cifar100 \
        --hflip true \
        --random_rotation=10 \
        --random_translate=4 \
        --cutout=4 \
    --optimizer=sgd \
        --training_steps=$STEPS \
        --lr_scheduler onecycle \
        --lr $LR \
        --momentum=0.9 \
        --warmup_ratio=0.025 \
        --batch_size=128 \
    --log_dir=/network/scratch/g/gul-sena.altintas/pretrain \
    --cleanup_after=false \
    --use_wandb $WANDB \
        --group=ncifar100-resnet50 \
        --run_name=resnet50-cifar100-$NORM \
        --project=ImagePreTraining \
    --use_tqdm=true \
    --n_models=1 \
        --seed1=$SEED1 \
        --loader_seed1=$SEED1 \
        --deterministic=false  \
    --save_freq=20ep
