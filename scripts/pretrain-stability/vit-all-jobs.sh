#!/bin/bash

# Help and documentation
show_help() {
    cat <<EOF
Script for submitting ViT stability experiments with different configurations.

Usage: 
    $(basename $0) [--preset NAME] [--checkpoint STEP...] [--dataset NAME...] [--mode MODE]

Options:
    --preset NAME     - Load a predefined configuration set
        Available presets:
        - gaussian_full   : Full-scale gaussian noise experiments (8 datasets)
        - batch          : Batch-wise perturbation experiments

    --checkpoint ... - Override model name from hf steps to run
        Example: --checkpoint "google/vit-base-patch16-224"

    --dataset NAME...    - Override datasets to run
        Available datasets: cifar100, cifar10, etc.
        Example: --dataset 

    --mode MODE      - Submission mode (default: grouped)
        Available modes: grouped, single, direct

    --base_seed BASE_MODEL_SEED     - multibert seed to run from
        Available seeds: 0, 1 Rest (TODO)

    --perturb_steps STEP...     - Override perturb steps to perturb models
        Example: --perturb_step 1 100

    --perturb_scales SCALE...     - Override perturb scales to perturb models
        Example: --perturb_scale 0.1 0.01

    --warmup_ratios WARMUP...     - Override warmup ratios
        Example: --warmup_ratios 0.1

Examples:
    $(basename $0) --preset gaussian_small --checkpoint 20k 200k --dataset sst2 cola
    $(basename $0) --preset gaussian_full --checkpoint 2000k
    $(basename $0) --preset batch --dataset mnli --mode single

Environment:
    Set NJOBS=<number> to limit concurrent jobs (default: 40000)
EOF
    exit 0
}

# Show help if requested
[[ "$1" == "--help" || "$1" == "-h" ]] && show_help

# Common configuration parameters
ALL_DATASETS=("cifar100")
ALL_MODEL_NAMES=("google/vit-base-patch16-224")
DEFAULT_WARMUP_RATIOS=(0 0.1)
if [[ $(pwd) == *"mila"* ]]; then
    GPU="l40s"
    gpu_str=" --gres=gpu:l40s:1 --mem-per-cpu=16G --tmp=32G --cpus-per-gpu=4 "
else
    GPU="a6000"
    gpu_str=" --nodelist=quartet5 --gres=gpu:1 --mem-per-cpu=8G --cpus-per-task=4 "
fi
echo gpu: $GPU

# Default values
MODE="single"
PRESET="gaussian_full" # Default preset
PRESET="batch" # Default preset
PERTURB_MODE="batch"
BASE_MODEL_SEED="0"
PERTURB_STEPS=(0 1 375 750)
WARMUP_RATIOS=(0 0.1)
TRAIN_SEED=42
NJOBS=${NJOBS:-40000}
declare -a CUSTOM_CHECKPOINTS=()
declare -a CUSTOM_DATASETS=()
declare -a CUSTOM_PERTURB_STEPS=()
declare -a CUSTOM_WARMUP_RATIOS=()
declare -a CUSTOM_PERTURB_SCALES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --preset)
        PRESET="$2"
        shift 2
        ;;
    --checkpoint)
        shift
        while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
            CUSTOM_CHECKPOINTS+=("$1")
            shift
        done
        ;;
    --dataset)
        shift
        while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
            CUSTOM_DATASETS+=("$1")
            shift
        done
        ;;
    --perturb_steps)
        shift
        while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
            CUSTOM_PERTURB_STEPS+=("$1")
            shift
        done
        ;;
    --perturb_scales)
        shift
        while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
            CUSTOM_PERTURB_SCALES+=("$1")
            shift
        done
        ;;
    --warmup_ratios)
        shift
        while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
            CUSTOM_WARMUP_RATIOS+=("$1")
            shift
        done
        ;;
    --mode)
        MODE="$2"
        shift 2
        ;;
    --train_seed)
        TRAIN_SEED="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        show_help
        exit 1
        ;;
    esac
done


# Check if CUSTOM_PERTURB_STEPS is empty and set perturbation steps accordingly
if [ ${#CUSTOM_PERTURB_STEPS[@]} -eq 0 ]; then
    echo "Using default perturbation steps: ${PERTURB_STEPS[@]}"
else
    PERTURB_STEPS=("${CUSTOM_PERTURB_STEPS[@]}")
    echo "Using custom perturbation steps: ${PERTURB_STEPS[@]}"
fi


# Check if CUSTOM_WARMUP_RATIOS is empty and set warmup ratios accordingly
if [ ${#CUSTOM_WARMUP_RATIOS[@]} -eq 0 ]; then
    echo "Using default warmup ratios: ${WARMUP_RATIOS[@]}"
else
    WARMUP_RATIOS=("${CUSTOM_WARMUP_RATIOS[@]}")
    echo "Using custom warmup ratios: ${WARMUP_RATIOS[@]}"
fi

# Configuration presets
gaussian_full_config() {
    DATASETS=("${ALL_DATASETS[@]}")
    CKPT_STEPS=("${ALL_MODEL_NAMES[@]}")
    PERTURB_MODE="gaussian"
}

batch_config() {
    DATASETS=("cifar100")
    CKPT_STEPS=("google/vit-base-patch16-224")
    PERTURB_MODE="batch"
}


if [[ $PERTURB_MODE -eq "batch" ]]; then
    PERTURB_SCALES=(0.00001 0.001 0.01 0.1)
else
    PERTURB_SCALES=(0.1 0.5 0.01)
fi

# Check if CUSTOM_PERTURB_SCALES is empty and set perturbation scales accordingly
if [ ${#CUSTOM_PERTURB_SCALES[@]} -eq 0 ]; then
    echo "Using default perturbation scales: ${PERTURB_SCALES[@]}"
else
    PERTURB_SCALES=("${CUSTOM_PERTURB_SCALES[@]}")
    echo "Using custom perturbation scales: ${PERTURB_SCALES[@]}"
fi

# Load base configuration first
case $PRESET in
"gaussian_full") gaussian_full_config ;;
"batch") batch_config ;;
*)
    echo "Invalid preset name: $PRESET"
    echo "Available presets: gaussian_full, batch"
    exit 1
    ;;
esac

# Override with custom values if provided
if [[ ${#CUSTOM_CHECKPOINTS[@]} -gt 0 ]]; then
#     # Validate checkpoints
#     for step in "${CUSTOM_CHECKPOINTS[@]}"; do
#         if [[ ! " ${ALL_MODEL_NAMES[@]} " =~ " ${step} " ]]; then
#             echo "Invalid checkpoint step: $step"
#             echo "Available steps: ${ALL_MODEL_NAMES[*]}"
#             exit 1
#         fi
#     done
    CKPT_STEPS=("${CUSTOM_CHECKPOINTS[@]}")
fi

if [[ ${#CUSTOM_DATASETS[@]} -gt 0 ]]; then
    # Validate datasets
    for dataset in "${CUSTOM_DATASETS[@]}"; do
        if [[ ! " ${ALL_DATASETS[@]} " =~ " ${dataset} " ]]; then
            echo "Invalid dataset: $dataset"
            echo "Available datasets: ${ALL_DATASETS[*]}"
            exit 1
        fi
    done
    DATASETS=("${CUSTOM_DATASETS[@]}")
fi


# Validate mode
case $MODE in
"grouped" | "single" | "direct") ;;
*)
    echo "Invalid mode: $MODE"
    echo "Available modes: grouped, single, direct"
    exit 1
    ;;
esac

# Echo selected configuration
echo "Mode: $MODE"
echo "Datasets: ${DATASETS[*]}"
echo "Checkpoint steps: ${CKPT_STEPS[*]}"
echo "Perturbation steps: ${PERTURB_STEPS[*]}"
echo "Perturbation scales: ${PERTURB_SCALES[*]}"
echo "Warmup ratios: ${WARMUP_RATIOS[*]}"
echo "Perturbation mode: $PERTURB_MODE"
echo "Base model seed: $BASE_MODEL_SEED"
echo "Training seed: $TRAIN_SEED"

# exit

# Main submission logic
case $MODE in
"direct")
    # Run directly without sbatch
    for CKPT_STEP in "${CKPT_STEPS[@]}"; do
        for PERTURB_STEP in "${PERTURB_STEPS[@]}"; do
            for PERTURB_SCALE in "${PERTURB_SCALES[@]}"; do
                for WARMUP_RATIO in "${WARMUP_RATIOS[@]}"; do
                    for DATASET in "${DATASETS[@]}"; do
                        bash scripts/pretrain-stability/train-cifar100-from-imagenet-vit.sh \
                            "${DATASET}" \
                            "${CKPT_STEP}" \
                            "${PERTURB_STEP}" \
                            "${PERTURB_SCALE}" \
                            "${PERTURB_MODE}" \
                            "${WARMUP_RATIO}" \
                            "${TRAIN_SEED}" 
                        sleep 0.2
                    done
                done
            done
        done
    done
    ;;

"single")
    # Submit each combination as a separate job
    for CKPT_STEP in "${CKPT_STEPS[@]}"; do
        for PERTURB_STEP in "${PERTURB_STEPS[@]}"; do
            for PERTURB_SCALE in "${PERTURB_SCALES[@]}"; do
                for WARMUP_RATIO in "${WARMUP_RATIOS[@]}"; do
                    for DATASET in "${DATASETS[@]}"; do
                        while [[ $(squeue -u $USER | wc -l) -ge $NJOBS ]]; do
                            sleep 1
                        done
                        sbatch --time=360 $gpu_str \
                            --job-name="pretrain-bert-stab-${DATASET}" \
                            --wrap="bash scripts/pretrain-stability/train-cifar100-from-imagenet-vit.sh \
                                    '${DATASET}' \
                                    '${CKPT_STEP}' \
                                    '${PERTURB_STEP}' \
                                    '${PERTURB_SCALE}' \
                                    '${PERTURB_MODE}' \
                                    '${WARMUP_RATIO}' \
                                    '${TRAIN_SEED}'"
                        sleep 0.2
                    done
                done
            done
        done
    done
    ;;

"grouped")
    # Group jobs by dataset
    for DATASET in "${DATASETS[@]}"; do
        while [[ $(squeue -u $USER | wc -l) -ge $NJOBS ]]; do
            sleep 1
        done

        # Convert arrays to space-separated strings
        CKPT_STEPS_STR="${CKPT_STEPS[*]}"
        PERTURB_STEPS_STR="${PERTURB_STEPS[*]}"
        PERTURB_SCALES_STR="${PERTURB_SCALES[*]}"
        WARMUP_RATIOS_STR="${WARMUP_RATIOS[*]}"

        sbatch --time=1040 $gpu_str \
            --job-name="pretrain-vit-stab-${DATASET}" \
            --wrap="/bin/bash -c '
                    IFS=\" \" read -r -a CKPT_STEPS <<< \"$CKPT_STEPS_STR\"
                    IFS=\" \" read -r -a PERTURB_STEPS <<< \"$PERTURB_STEPS_STR\"
                    IFS=\" \" read -r -a PERTURB_SCALES <<< \"$PERTURB_SCALES_STR\"
                    IFS=\" \" read -r -a WARMUP_RATIOS <<< \"$WARMUP_RATIOS_STR\"

                    for CKPT_STEP in \"\${CKPT_STEPS[@]}\"; do
                        for PERTURB_STEP in \"\${PERTURB_STEPS[@]}\"; do
                            for PERTURB_SCALE in \"\${PERTURB_SCALES[@]}\"; do
                                for WARMUP_RATIO in \"\${WARMUP_RATIOS[@]}\"; do
                                    bash scripts/pretrain-stability/train-cifar100-from-imagenet-vit.sh  \
                                        \"${DATASET}\" \
                                        \"\${CKPT_STEP}\" \
                                        \"\${PERTURB_STEP}\" \
                                        \"\${PERTURB_SCALE}\" \
                                        \"${PERTURB_MODE}\" \
                                        \"\${WARMUP_RATIO}\" \
                                        \"${TRAIN_SEED}\" 
                                    sleep 0.2
                                done
                            done
                        done
                    done'"
    done
    ;;
esac
