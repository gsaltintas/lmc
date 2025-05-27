#!/bin/bash
#SBATCH --time=249
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --tmp=8G
#SBATCH --job-name=glue_train
#SBATCH --begin=now+0minutes
DATASET=${1-"sst2"} # Pass dataset as argument
CKPT=${2-"40k"}
PERTURB_STEP=$3
PERTURB_SCALE=$4
PERTURB_MODE=${5-"gaussian"}
WARMUP_RATIO=${6-0.1}
BASE_MODEL_SEED=${7-0}
TRAIN_SEED=${8-42}
LR=2e-5
WD=0.01
SEED1=42    
SEED2=24
SEED2=42
SEED3=42
PSEED2=24
PSEED3=23

SEED1=$TRAIN_SEED
SEED2=$TRAIN_SEED

BS=64
GRAD_ACCUM=4
GRAD_ACCUM=1
USE_WANDB="true"
# USE_WANDB="false"
STEPS="7500st"
# STEPS="10st"
NPOINTS=11
# NPOINTS=2
# STEPS="10st"
PROJECT="LMCPretrainingStability-MultiBert-2"
PROJECT="LMCPretrainingStability-MultiBert-2-CKA"

# Adjust batch size and steps based on dataset size and complexity
if [[ "$DATASET" =~ ^(rte|mrpc)$ ]]; then
    # Small datasets (~2.5k-3.7k examples)
    BS=32
    STEPS="500st"
elif [[ "$DATASET" =~ ^(cola|stsb)$ ]]; then
    # Medium-small datasets (~5.7k-8.5k examples)
    BS=32
    STEPS="1500st"
elif [[ "$DATASET" =~ ^(sst2)$ ]]; then
    # Medium dataset (~67k examples)
    BS=128
    STEPS="2500st"
elif [[ "$DATASET" =~ ^(qnli)$ ]]; then
    # Large dataset (~105k examples)
    BS=32
    # BS=64
    GRAD_ACCUM=1
    STEPS="3ep"
elif [[ "$DATASET" =~ ^(qqp)$ ]]; then
    # Very large dataset (~364k examples)
    BS=256
    BS=32
    GRAD_ACCUM=1
    STEPS="3ep"
elif [[ "$DATASET" =~ ^(mnli)$ ]]; then
    # Largest dataset (393k examples) with three-class classification
    BS=32
    GRAD_ACCUM=1
    STEPS="3ep"
fi

# BS=32
# STEPS="3ep"

# CKPT="100k"
# CKPT="40k"
MODEL="google/multiberts-seed_${BASE_MODEL_SEED}-step_${CKPT}"
TOKENIZER="bert-base-uncased"

python main.py perturb \
    --model_name $MODEL \
    --initialization_strategy=pretrained \
    --tokenizer $TOKENIZER \
    --dataset $DATASET \
    --optimizer adamw \
    --training_steps=$STEPS \
    --lr $LR \
    --weight_decay $WD \
    --lr_scheduler onecycle \
    --warmup_ratio $WARMUP_RATIO \
    --batch_size=$BS \
    --test_batch_size=128 \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --log_dir=$SCRATCH/finetune_lmc \
    --cleanup_after=false \
    --use_wandb $USE_WANDB \
    --group=$DATASET-$MODEL \
    --run_name=bert-$CKPT-$DATASET-p${PERTURB_STEP}x${PERTURB_SCALE}-$STEPS \
    --project=$PROJECT \
    --tags=perturb \
    --use_tqdm=true \
    --n_models=2 \
    --save_freq="${STEPS}" \
    --seed1=$SEED1 \
    --loader_seed1=$SEED1 \
    --seed2=$SEED1 \
    --loader_seed2=$SEED1 \
    --n_points=$NPOINTS \
    --deterministic=true \
    --perturb_mode=$PERTURB_MODE \
    --perturb_step=$PERTURB_STEP \
    --perturb_scale=$PERTURB_SCALE \
    --normalize_perturb=true \
    --perturb_inds 1  \
    --same_steps_pperturb=false \
    --dont_perturb_module_patterns '.*norm.*|.*bias.*|.*embeddings.*' \
    --cka_n_train=10000 \
    --cka_n_test=10000 \
    --cka_strategy="last_element" \
    --cka_include="encoder.layer.0.output.out,encoder.layer.2.output.out,encoder.layer.5.output.out,encoder.layer.8.output.out,encoder.layer.11.output.out"
    
    # encoder.layer.0.output,encoder.layer.2.output,encoder.layer.5.output,encoder.layer.8.output,encoder.layer.11.output,encoder.layer.0.output,encoder.layer.2.output,encoder.layer.5.output,encoder.layer.8.output,encoder.layer.11.output"


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