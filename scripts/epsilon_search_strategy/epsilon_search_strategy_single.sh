#!/bin/bash

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

python main.py perturb  \
    --project="$SSETUP_PROJECT_NAME-$SSETUP_EXP_NAME"  \
        --run_name=$RUN_NAME  \
        --path=$SLURM_TMPDIR/data/$DATASET  \
        --log_dir=$SSETUP_OUTPUT_DIR  \
        --save_early_iters=true  \
        --cleanup_after=false  \
        --use_wandb=true  \
        --zip_and_save_source=false  \
    --model_name=$MODEL  \
        --norm=$NORM  \
    --dataset=cifar10  \
        --hflip=true  \
        --random_rotation=10  \
        --random_crop=false  \
    --optimizer=sgd  \
        --training_steps=50ep  \
        --lr=0.1   \
        --lr_scheduler=triangle  \
        --warmup_ratio=0.02  \
        --momentum=0.9  \
    --n_models=2  \
        --perturb_mode=$PERTURB_TYPE  \
        --perturb_scale=$SCALE  \
        --perturb_inds=1  \
    --deterministic=$DETERMINISTIC  \
        --seed1=$SEED  \
        --seed2=$SEED  \
        --loader_seed1=$SEED  \
        --loader_seed2=$SEED  \
        --perturb_seed1=$SEED  \
        --perturb_step=$PERTURB_STEP  \
