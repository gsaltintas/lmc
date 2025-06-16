#!/bin/bash

## Plan:
# 1. run finetuning from 20k step on all dataset
# 2. on datasets showing instability, run from 200k
# 3. last run from last ckpt (2000k)
# Repeat w/o warm-up (Juneja uses no warm-up)
# MODE can be: "direct" (run directly), "single" (one job per run), or "grouped" (one job per dataset)
MODE="grouped"
MODE="single"
# MODE="direct"
MODE=${MODE:-"grouped"}  # Default to grouped if not set

## Now running
# 20k, warmup on all 8

CKPT_STEPS=("20k" "40k" "60k" "80k" "100k" "200k" "500k" "1000k" "2000k")
CKPT_STEPS=("20k" "40k" "200k" "1000k" "2000k")
CKPT_STEPS=("20k" "200k" "2000k")
# ## layernorm version
# ROOT_="/network/scratch/g/gul-sena.altintas/pretrain/train-25-01-23-50553"
# # ## batchnorm version
# # ROOT_="/network/scratch/g/gul-sena.altintas/pretrain/train-25-01-23-45363"
# CONFIG_YAML="${ROOT_}/config.yaml"
# CKPT_ROOT="${ROOT_}/model1/checkpoints"
DATASETS=("sst2" "cola" "mrpc" "stsb")
DATASETS=("qqp" "mnli" "qnli" "rte")

DATASETS=("mnli")
CKPT_STEPS=("200k")
WARMUP_RATIOS=(0 0.1)
# DATASETS=("sst2" "cola" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte")
DATASETS=("qqp")
PERTURB_STEPS=(1 100)

# check this job on mila 5959296
# DATASETS=("sst2" "cola" "mrpc" "stsb")
# # DATASETS=("qqp" "mnli" "qnli" "rte")
# CKPT_STEPS=("20k")
# # DATASETS=("sst2" "cola" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte")
# WARMUP_RATIOS=(0 0.1)
# PERTURB_STEPS=(1 100)



CKPT_STEPS=("2000k")
WARMUP_RATIOS=(0 0.1)
DATASETS=("sst2" "cola" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte")
PERTURB_STEPS=(1 100)


PERTURB_MODE="gaussian"
PERTURB_SCALES=(0.1 0.5 0.01)



PERTURB_MODE="batch"
PERTURB_SCALES=(0.00001 0.001 0.01)
# DATASETS=("cola" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte")
njobs_at_time=40000
# PERTURB_STEPS=(1 100 2000)
# PERTURB_SCALES=( 0.00001 0.001  0.05  )
PERTURB_MODES=("gaussian" "batch")

## Gaussian

echo "submitting Gaussian perturbations in ${MODE} mode"
case $MODE in
    "direct")
        # Run directly without sbatch
        for DATASET in "${DATASETS[@]}"; do
            for CKPT_STEP in "${CKPT_STEPS[@]}"; do
                for PERTURB_STEP in "${PERTURB_STEPS[@]}"; do
                    for PERTURB_SCALE in "${PERTURB_SCALES[@]}"; do
                        for WARMUP_RATIO in "${WARMUP_RATIOS[@]}"; do
                            bash scripts/pretrain-stability/train-multibert.sh \
                                "${DATASET}" \
                                "${CKPT_STEP}" \
                                "${PERTURB_STEP}" \
                                "${PERTURB_SCALE}" \
                                "${PERTURB_MODE}" \
                                "${WARMUP_RATIO}"
                            sleep 0.2
                        done
                    done
                done
            done
        done
        ;;
        
    "single")
        # Submit each combination as a separate job
        for DATASET in "${DATASETS[@]}"; do
            for CKPT_STEP in "${CKPT_STEPS[@]}"; do
                for PERTURB_STEP in "${PERTURB_STEPS[@]}"; do
                    for PERTURB_SCALE in "${PERTURB_SCALES[@]}"; do
                        for WARMUP_RATIO in "${WARMUP_RATIOS[@]}"; do
                            while [[ $(squeue -u $USER | wc -l) -ge $njobs_at_time ]]; do
                                sleep 1
                            done
                            # sbatch --time=360 --gres=gpu:rtx8000:1 --mem-per-cpu=8G --tmp=8G --cpus-per-gpu=4 \
                            #     --job-name="pretrain-bert-stab-${DATASET}" \
                            #     --wrap=
                                echo "bash scripts/pretrain-stability/train-multibert.sh \
                                    '${DATASET}' \
                                    '${CKPT_STEP}' \
                                    '${PERTURB_STEP}' \
                                    '${PERTURB_SCALE}' \
                                    '${PERTURB_MODE}' \
                                    '${WARMUP_RATIO}'"
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
            while [[ $(squeue -u $USER | wc -l) -ge $njobs_at_time ]]; do
                sleep 1
            done

            # Convert arrays to space-separated strings
            CKPT_STEPS_STR="${CKPT_STEPS[*]}"
            PERTURB_STEPS_STR="${PERTURB_STEPS[*]}"
            PERTURB_SCALES_STR="${PERTURB_SCALES[*]}"
            WARMUP_RATIOS_STR="${WARMUP_RATIOS[*]}"

            sbatch --time=1040 --gres=gpu:rtx8000:1 --mem-per-cpu=8G --tmp=8G --cpus-per-gpu=4 \
                --job-name="pretrain-bert-stab-${DATASET}" \
                --wrap="/bin/bash -c '
                    IFS=\" \" read -r -a CKPT_STEPS <<< \"$CKPT_STEPS_STR\"
                    IFS=\" \" read -r -a PERTURB_STEPS <<< \"$PERTURB_STEPS_STR\"
                    IFS=\" \" read -r -a PERTURB_SCALES <<< \"$PERTURB_SCALES_STR\"
                    IFS=\" \" read -r -a WARMUP_RATIOS <<< \"$WARMUP_RATIOS_STR\"

                    for CKPT_STEP in \"\${CKPT_STEPS[@]}\"; do
                        for PERTURB_STEP in \"\${PERTURB_STEPS[@]}\"; do
                            for PERTURB_SCALE in \"\${PERTURB_SCALES[@]}\"; do
                                for WARMUP_RATIO in \"\${WARMUP_RATIOS[@]}\"; do
                                    bash scripts/pretrain-stability/train-multibert.sh \
                                        \"${DATASET}\" \
                                        \"\${CKPT_STEP}\" \
                                        \"\${PERTURB_STEP}\" \
                                        \"\${PERTURB_SCALE}\" \
                                        \"${PERTURB_MODE}\" \
                                        \"\${WARMUP_RATIO}\"
                                    sleep 0.2
                                done
                            done
                        done
                    done'"
        done
        ;;
    *)
        echo "Invalid MODE. Please use 'direct', 'single', or 'grouped'"
        exit 1
        ;;
esac
exit 0

PERTURB_SCALES=(0.00001 0.001 0.01)
## Batch
echo submitting Batch perturbations
for CKPT_STEP in "${CKPT_STEPS[@]}"; do
    for PERTURB_STEP in "${PERTURB_STEPS[@]}"; do
        for PERTURB_SCALE in "${PERTURB_SCALES[@]}"; do
            for DATASET in "${DATASETS[@]}"; do
                # while [[ $(squeue -u $USER | wc -l) -ge $njobs_at_time ]]; do sleep 1; done
                # Submit the job with the CKPT_STEP as an argument
                sbatch --job-name="pretrain-bert-stab-${CKPT_STEP}" \
                    scripts/pretrain-stability/train-multibert.sh "${DATASET}" "${CKPT_STEP}" "${PERTURB_STEP}" "${PERTURB_SCALE}" "batch" "${WARMUP_RATIO}"
                sleep 0.2
            done
        done
    done
done
