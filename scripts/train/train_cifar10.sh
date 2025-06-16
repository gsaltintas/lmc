#!/bin/bash
#SBATCH --time=249
#SBATCH --gres=gpu:rtx8000:1
##SBATCH --gres=gpu:1
##SBATCH --nodelist=quartet5
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --job-name=resnet50


# On the DCS cluster

# (phd) gsaltintas@guppy4:~/lmc$ squeue_ FAILED
#              JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
#             206855        ml resnet50 gsaltint  R       0:07      1 quartet5
#             206856        ml resnet50 gsaltint  R       0:07      1 quartet5

# 57, 58
seeds1=(22 45 987)
seeds2=(43 66 1008)
repetition=3
NORM=batchnorm
NORM=layernorm
DATASET=cifar10
LR=0.1
WD=1e-4
SEED1=1
SEED1=2
STEPS="75000st"
WANDB=true
WARMUP_RATIO=0.025
# WARMUP_RATIO=0.032
CUTOUT=4
CUTOUT=2
# LR=0.01
# LR=0.001
# LR=0.1
# WD=5e-4
# WARMUP_RATIO=0.1
# WARMUP_RATIO=0.

python main.py train \
    --model_name resnet50-64 \
        --norm=$NORM \
    --dataset $DATASET \
        --path=$SCRATCH/data/$DATASET \
        --hflip true \
        --random_rotation=10 \
        --random_translate=4 \
        --cutout=$CUTOUT \
        --download=True \
    --optimizer=sgd \
        --training_steps=$STEPS \
        --lr_scheduler onecycle \
        --lr $LR \
        --momentum=0.9 \
        --warmup_ratio=$WARMUP_RATIO \
        --batch_size=128 \
    --log_dir=$HOME/pretrain \
    --cleanup_after=false \
    --use_wandb $WANDB \
        --group=$DATASET-resnet50-new \
        --run_name=resnet50-$DATASET-$NORM \
        --project=ImagePreTraining \
    --save_freq=1500st \
    --save_specific_steps="1st,180st,1ep,2000st" \
    --use_tqdm=true \
    --n_models=1 \
        --seed1=$SEED1 \
        --loader_seed1=$SEED1 \
        --deterministic=false  
