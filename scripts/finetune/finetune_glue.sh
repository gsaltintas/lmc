#!/bin/bash
#SBATCH --time=249
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --tmp=8G
#SBATCH --job-name=glue_train
#SBATCH --begin=now+0minutes

DATASET=$1  # Pass dataset as argument
LR=2e-5
WD=0.1
SEED1=42    # Changed seed for better initialization
SEED2=$SEED1
BS=128
GRAD_ACCUM=4
MODEL="bert-base-uncased"
USE_WANDB="true"
STEPS="2500st"

# Adjust batch size and steps based on dataset
if [[ "$DATASET" =~ ^(cola|mrpc|rte|stsb)$ ]]; then
    BS=32
    GRAD_ACCUM=8
    # STEPS="5ep"
elif [[ "$DATASET" =~ ^(qqp|mnli)$ ]]; then
    BS=256
    # STEPS="2ep"
fi

python main.py train \
    --model_name $MODEL \
    --tokenizer $MODEL \
    --dataset $DATASET \
    --optimizer adamw \
    --training_steps=$STEPS \
    --lr $LR \
    --weight_decay $WD \
    --lr_scheduler onecycle \
    --warmup_ratio 0.1 \
    --batch_size=$BS \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --log_dir=$SCRATCH/finetune_lmc \
    --cleanup_after=false \
    --use_wandb $USE_WANDB \
    --group=$DATASET-$MODEL \
    --run_name=$MODEL-$DATASET-$STEPS \
    --project=LMCFinetuning-NLP \
    --use_tqdm=true \
    --n_models=1 \
    --seed1=$SEED1 \
    --loader_seed1=$SEED1 \
    --deterministic=true