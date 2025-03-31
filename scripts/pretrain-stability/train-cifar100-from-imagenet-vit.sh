#!/bin/bash
#SBATCH --time=249
#SBATCH --gres=gpu:l40s:1
###SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=32G
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name=vit

seeds1=(22 45 987)
seeds2=(43 66 1008)
repetition=3
NORM=batchnorm
# NORM=layernorm
DATASET=${1-cifar100}
# DATASET=${1-eurosat}
# DATASET=cifar10
# microsoft/swin-tiny-patch4-window7-224
MODEL="ismgar01/vit-base-patch16-imagenette"
# google/vit-large-patch16-224
MODEL=${2-"google/vit-base-patch16-224"}

LR=0.1
OPT=sgd
OPT=adamw
WD=1e-4
SEED1=1
SEED2=2
SEED3=3
STEPS=5ep
WANDB=true
tqdm=false
RESIZE=224
BS=32

LR=2e-4

PERTURB_STEP=${3-0}
PERTURB_SCALE=${4-0.1}
PERTURB_MODE=${5-batch}
WARMUP_RATIO=${6-0.1}
SEED1=${7-$SEED1}

python main.py perturb \
    --model_name $MODEL \
    --initialization_strategy=pretrained \
    --pin_memory=false \
    --dataset $DATASET \
        --path=data \
        --hflip true \
        --random_rotation=10 \
        --random_translate=16 \
        --cutout=16 \
        --download=true \
        --resize=$RESIZE \
        --num_workers=4 \
    --optimizer=$OPT \
        --training_steps=$STEPS \
        --lr_scheduler onecycle \
        --lr $LR \
        --momentum=0.9 \
        --warmup_ratio=$WARMUP_RATIO \
        --batch_size=$BS \
        --test_batch_size=128 \
    --log_dir=/network/scratch/g/gul-sena.altintas/finetune \
        --enforce_new_model_dir=True \
        --use_tqdm=$tqdm \
        --save_freq=5ep \
        --cleanup_after=false \
    --use_wandb $WANDB \
        --group=$MODEL\
        --run_name=$MODEL-$DATASET-$PERTURB_MODE \
        --project=LMCFinetuning-ViT \
    --n_models=2 \
        --seed1=$SEED1 \
        --loader_seed1=$SEED1 \
        --seed2=$SEED1 \
        --loader_seed2=$SEED1 \
        --deterministic=true  \
    --perturb_mode=$PERTURB_MODE \
    --perturb_step=$PERTURB_STEP \
    --perturb_scale=$PERTURB_SCALE \
    --perturb_inds 1 \
    --normalize_perturb=true \
    --same_steps_pperturb=false \
    --dont_perturb_module_patterns '.*norm.*|.*bias.*' 

# srun --time 480 --mem-per-cpu 16G --tmp 32G --cpus-per-gpu 4 --gres=gpu:l40s:1 --pty bash
# srun --time 480 --mem-per-cpu 16G --tmp 32G --cpus-per-gpu 4 --gres=gpu:rtx8000:1 --pty bash