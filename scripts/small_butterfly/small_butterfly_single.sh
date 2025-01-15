#!/bin/bash

PERTURB_STEP=$1
SCALE=$2
REPLICATE=$3
MODEL=$4
BATCH_SIZE=$5
LR=$6
WARMUP_RATIO=$7
DONT_PERTURB_PATTERNS=$8
GROUP=$9

DATASET="cifar10"
NORM="layernorm"
PERTURB_TYPE="batch"

SEED=$REPLICATE
RUN_NAME="$GROUP-$MODEL-$PERTURB_TYPE-p$PERTURB_STEP-s$SCALE-r$REPLICATE"
echo $RUN_NAME

source $HOME/ssetup-uv.sh $DATASET

python main.py perturb  \
    --project="$SSETUP_PROJECT_NAME-$SSETUP_EXP_NAME"  \
        --run_name=$RUN_NAME  \
        --group=$GROUP  \
        --path=$SLURM_TMPDIR/data/$DATASET  \
        --log_dir=$SSETUP_OUTPUT_DIR  \
        --save_early_iters=false  \
        --cleanup_after=true  \
        --zip_and_save_source=false  \
        --use_wandb=true  \
    --model_name=$MODEL  \
        --norm=$NORM  \
    --dataset=$DATASET  \
        --hflip=true  \
        --random_rotation=10  \
        --random_crop=false  \
    --optimizer=sgd  \
        --training_steps=50ep  \
        --batch_size=$BATCH_SIZE  \
        --lr=$LR   \
        --lr_scheduler=triangle  \
        --warmup_ratio=$WARMUP_RATIO  \
        --momentum=0.9  \
    --n_models=2  \
        --perturb_mode=$PERTURB_TYPE  \
        --perturb_scale=$SCALE  \
        --perturb_step=$PERTURB_STEP  \
        --perturb_inds=1  \
        --same_steps_pperturb=false  \
        --normalize_perturb=false  \
    --deterministic=true  \
        --seed1=$SEED  \
        --seed2=$SEED  \
        --loader_seed1=$SEED  \
        --loader_seed2=$SEED  \
        --perturb_seed1=$SEED  \
    --lmc_check_perms=false  \
        --lmc_on_epoch_end=false  \
        --lmc_on_train_end=true  \
    --dont_perturb_module_patterns=$DONT_PERTURB_PATTERNS  \
