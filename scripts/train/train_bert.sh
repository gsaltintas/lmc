## Bert scripts

python main.py train \
    --n_models=2 \
    --model_name bert-base-uncased \
    --tokenizer bert-base-uncased \
    --training_steps 10st \
    --dataset snli \
    --batch_size 8 \
    --max_seq_length 256 \
    --lr 3e-5 \
    --weight_decay 0.01 \
    --use_wandb false \
    --deterministic=true \
    --seed1=1 --loader_seed1=1 \
    --seed2=1 --loader_seed2=1 
    # --frozen_layers "^embeddings" "^encoder\.layer\.[0-2]" \



python main.py train \
    --training_steps 1st \
    --n_models=2 \
    --model_name bert-base-uncased \
    --tokenizer bert-base-uncased \
    --dataset snli \
    --batch_size 256 \
    --lr 3e-5 \
    --weight_decay 0.01 \
    --use_wandb false \
    --deterministic=true \
    --seed1=1 --loader_seed1=1 \
    --seed2=1 --loader_seed2=1 




sbatch --time=120 --cpus-per-gpu=4 --gres=gpu:rtx8000:1 --tmp=8G --mem 8G --wrap="python main.py train \
    --training_steps 3ep \
    --log_dir=$SCRATCH/experiments \
    --n_models=1 \
    --model_name bert-base-uncased \
    --tokenizer bert-base-uncased \
    --dataset snli \
    --batch_size 64 --gradient_accumulation_steps=4 \
    --num_workers=4 \
    --lr 3e-5 \
    --weight_decay 0.01 \
    --use_wandb true \
    --project=lmc_nlp \
    --deterministic=true \
    --seed1=1 --loader_seed1=1 \
    --n_models=2 \
    --seed2=1 --loader_seed2=1 "

scale=0.1; step=1
for scale in 0.1 0.01 0.5; do
for step in 1 5 8597; do
sbatch --time=240 --cpus-per-gpu=4 --gres=gpu:rtx8000:1 --tmp=8G --mem 8G --wrap="python main.py perturb \
    --training_steps 3ep \
    --log_dir=$SCRATCH/experiments \
    --n_models=1 \
    --model_name bert-base-uncased \
    --tokenizer bert-base-uncased \
    --dataset snli \
    --batch_size 64 --gradient_accumulation_steps=4 \
    --test_batch_size 128 \
    --num_workers=4 \
    --lr 3e-5 \
    --weight_decay 0.01 \
    --use_wandb true \
    --project=lmc_nlp \
    --deterministic=true \
    --seed1=1 --loader_seed1=1 \
    --n_models=2 \
    --seed2=1 --loader_seed2=1  \
    --perturb_step=$step \
    --perturb_mode=batch --perturb_scale=$scale \
    --n_points=5 --save_freq=1st --use_wandb=false --resume_from /network/scratch/g/gul-sena.altintas/experiments/perturbed-trainer_81a00479c05e4b075fe8fcdcb6cc0ded-17-01-25-23-18-103633/
    
    --ckpt_path=/network/scratch/g/gul-sena.altintas/experiments/perturbed-trainer_81a00479c05e4b075fe8fcdcb6cc0ded-17-01-25-23-18-103633/model1-seed_1-ls_1/checkpoints/ep-1-st-1.ckpt"
done
done