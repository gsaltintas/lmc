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
DATASET=cifar10
LR=0.1
WD=1e-4
SEED1=1
SEED2=2
SEED3=3
STEPS=5ep
WANDB=false


python main.py train \
    --config_file /network/scratch/g/gul-sena.altintas/pretrain/trainer_d28f2652eb0fc00e8f4996950925ff69-20-01-25-07-41-301734/config.yaml \
    --config_file "/network/scratch/g/gul-sena.altintas/pretrain/trainer_f79899f1a464935a99bbc2dfa4d91289-19-01-25-21-58-220181/config.yaml" \
    --model_name resnet50-64 \
        --norm=$NORM \
        --ckpt_path="/network/scratch/g/gul-sena.altintas/pretrain/trainer_f79899f1a464935a99bbc2dfa4d91289-19-01-25-21-58-220181/model1-seed_1-ls_1/checkpoints/ep-200.ckpt" \
    --dataset $DATASET \
        --path=data/cifar10 \
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
    --log_dir=/network/scratch/g/gul-sena.altintas/finetune \
        --enforce_new_model_dir=True \
        --use_tqdm=true \
        --save_freq=5ep \
        --cleanup_after=false \
    --use_wandb $WANDB \
        --group=cifar10-resnet50 \
        --run_name=resnet50-cifar10 \
        --project=LMCFinetuning \
    --n_models=2 \
        --seed1=$SEED1 \
        --loader_seed1=$SEED1 \
        --seed2=$SEED2 \
        --loader_seed2=$SEED2 \
        --deterministic=true  
