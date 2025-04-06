#!/bin/bash
#SBATCH --time=249
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=8
#SBATCH --tmp=8G
#SBATCH --job-name=glue_train
#SBATCH --begin=now+0minutes
#SBATCH --output=slurm-%j.out
#### vector
##SBATCH --account=deadline
##SBATCH --qos=deadline

#  salloc  --cpus-per-task=8 --mem=72G --time=03:00:00 --gres=gpu:a100:1 --qos=deadline --account deadline
# gsm8k, math, mathqa
DATASET=${1-"gsm8k"} # Pass dataset as argument
REVISION="stage2-ingredient3-step90000-tokens38B"
REVISION=${2-"stage1-step99000-tokens416B"}
PERTURB_STEP=${3-1}
PERTURB_SCALE=${4-0.001}
PERTURB_MODE=${5-"batch"}
WARMUP_RATIO=${6-0.1}
TRAIN_SEED=${7-42}
LR=2e-5
BS=2
GRAD_ACCUM=8

# DATASET=${1-"mathqa"} # Pass dataset as argument
# ics.py", line 310, in compute_metrics
#     res = func(predictions, references)
#   File "/home/mila/g/gul-sena.altintas/clean/lmc/utils/metrics.py", line 125, in compute_classification_metrics
#     assert predictions.shape == references.shape
# DATASET=${1-"asdiv"} # Pass dataset as argument
# BS=2
# GRAD_ACCUM=8


STEPS="1000st"
STEPS="3ep"

WD=0.01

SEED1=$TRAIN_SEED
SEED2=$TRAIN_SEED

USE_WANDB="true"
USE_WANDB="false"
NPOINTS=11
# NPOINTS=2
# STEPS="10st"
PROJECT="LMCPretrainingStability-OLMo"

DATASET=${1-"mathqa"} # Pass dataset as argument
DATASET=${1-"gsm8k"} # Pass dataset as argument

# Dataset-specific adjustments
case $DATASET in
    "gsm8k")
        BS=16
        BS=4
        GRAD_ACCUM=4
        GRAD_ACCUM=1
        MAX_SEQ_LENGTH=512
        TBS=32
        TBS=8
        ;;
    "math")
        MAX_SEQ_LENGTH=2048  # Longer sequences for mathematical proofs
        BS=1
        GRAD_ACCUM=8        # Increased accumulation due to longer sequences
        TBS=8
        ;;
    "asdiv")
        BS=16
        GRAD_ACCUM=1
        MAX_SEQ_LENGTH=512
        TBS=32
        ;;
    "mathqa")
        BS=16
        GRAD_ACCUM=1
        MAX_SEQ_LENGTH=512
        TBS=32
        ;;
esac

# MODEL="allenai/OLMo-2-1124-7B"
# TOKENIZER="allenai/OLMo-2-1124-7B"
chat_template="olmo"
# REVISION="stage1-step812000-tokens3406B"


## olmoe
# MODEL="allenai/OLMoE-1B-7B-0924"
# TOKENIZER="allenai/OLMoE-1B-7B-0924"
# REVISION="step995000-tokens4173B"

MODEL="allenai/OLMo-1B"
TOKENIZER="allenai/OLMo-1B"
REVISION="step738020-tokens3095B"
MODEL="allenai/OLMo-1B-hf"
TOKENIZER="allenai/OLMo-1B-hf"
REVISION="step738000-tokens3094B"
# BS=16
# GRAD_ACCUM=1

seed_str="   --n_models=1 --seed1=$SEED1 --loader_seed1=$SEED1 "
seed_str="   --n_models=2 --seed1=$SEED1 --loader_seed1=$SEED1 --seed2=$SEED1 --loader_seed2=$SEED1 "

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python main.py perturb \
    --eval_freq=none \
    --eval_specific_steps="1000000000st" \
    --model_name $MODEL \
    --revision $REVISION \
    --initialization_strategy=pretrained \
    --tokenizer $TOKENIZER \
    --chat_template $chat_template \
    --max_seq_length $MAX_SEQ_LENGTH \
    --use_bfloat16=true \
    --gradient_checkpointing=true \
    --dataset $DATASET \
    --optimizer adamw \
    --training_steps=$STEPS \
    --lr $LR \
    --weight_decay $WD \
    --lr_scheduler onecycle \
    --warmup_ratio $WARMUP_RATIO \
    --test_batch_size=$TBS \
    --batch_size=$BS \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --log_dir=$SCRATCH/finetune_lmc \
    --cleanup_after=false \
    --use_wandb $USE_WANDB \
    --group=$DATASET-$MODEL \
    --run_name=olmo-$CKPT-$DATASET-p${PERTURB_STEP}x${PERTURB_SCALE}-$STEPS \
    --project=$PROJECT \
    --tags=perturb \
    --use_tqdm=true \
    --save_freq="1ep" \
    --n_points=$NPOINTS \
    --deterministic=true \
    --perturb_mode=$PERTURB_MODE \
    --perturb_step=$PERTURB_STEP \
    --perturb_scale=$PERTURB_SCALE \
    --normalize_perturb=true \
    --perturb_inds 1 \
    --same_steps_pperturb=false \
    --dont_perturb_module_patterns '.*norm.*|.*bias.*|.*embeddings.*' \
    $seed_str


exit 0
    --seed2=$SEED1 \
    --loader_seed2=$SEED1 \


exit 0
    --n_models=3 \
    --perturb_inds 1 2 \
    --seed1=$SEED1 \
    --loader_seed1=$SEED1 \
    --seed2=$SEED2 \
    --loader_seed2=$SEED2 \
    --perturb_seed2=$PSEED2 \
    --seed3=$SEED3 \
    --loader_seed3=$SEED3 \
    --perturb_seed3=$PSEED3 