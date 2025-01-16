#!/bin/bash
PERTURB_STEP=$1
THRESHOLD=$2
INIT_SCALE=$3
N_RUNS=$4
SEED=$5

MODEL=resnet20-32
DATASET="cifar10"
NORM="layernorm"
PERTURB_MODE="gaussian"

RUN_NAME="logreg-$PERTURB_STEP-$N_RUNS"
echo $RUN_NAME

source $HOME/ssetup-uv.sh $DATASET

python main.py logreg  \
    --logreg_n=$N_RUNS  \
        --logreg_x="perturb_scale"  \
        --logreg_y="lmc-0-1/lmc/loss/weighted/increase_end0_train"  \
        --logreg_threshold=$THRESHOLD  \
        --logreg_max_step_ratio=10  \
        --logreg_reseed_every_run=true  \
    --project="$SSETUP_PROJECT_NAME-$SSETUP_EXP_NAME"  \
        --run_name=$RUN_NAME  \
        --path=$SLURM_TMPDIR/data/$DATASET  \
        --log_dir=$SSETUP_OUTPUT_DIR  \
        --save_early_iters=false  \
        --cleanup_after=false  \
        --use_wandb=true  \
        --zip_and_save_source=false  \
    --model_name=$MODEL  \
        --norm=$NORM  \
    --dataset=$DATASET  \
        --hflip=true  \
        --random_rotation=10  \
    --optimizer=sgd  \
        --training_steps=50ep  \
        --lr=0.1   \
        --lr_scheduler=triangle  \
        --warmup_ratio=0.02  \
        --momentum=0.9  \
    --n_models=2  \
        --perturb_mode=$PERTURB_MODE  \
        --perturb_step=$PERTURB_STEP  \
        --perturb_scale=$INIT_SCALE  \
        --perturb_inds=1  \
        --same_steps_pperturb=false  \
    --deterministic=true  \
        --seed1=$SEED  \
        --seed2=$SEED  \
        --loader_seed1=$SEED  \
        --loader_seed2=$SEED  \
        --perturb_seed1=$SEED  \
    --lmc_check_perms=false  \
        --lmc_on_epoch_end=false  \
        --lmc_on_train_end=true  \
