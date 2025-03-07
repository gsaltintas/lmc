# Linear Mode-Connectivity and Perutation Invariance

This is the code base to perform several experiments related to lmc.

Pass models as below

```bash
python train.py train --model=mlp --mlp.width=1024
```

or 
```bash
python train.py train --model=resnet --resnet.width_multiplier=64
```

## Setup
## Installation
Install dependencies with
```bash
poetry install
```

If using ffcv, do the following instead:
```
conda create -y -n ffcvlmc python=3.10 libjpeg-turbo opencv -c conda-forge
conda activate ffcvlmc
pip install -r requirements.txt
```

Make sure to define these variables in your `.bashrc` file or for every run:
```
export WANDB_ENTITY="WANDB_ENTITY"
export WANDB_API_KEY="YOUR_API_KEY"
export SCRATCH="PATH_TO_SCRATCH"
export HUGGINGFACE_HUB_CACHE="$SCRATCH/.cache"
```

To disable torchvision warnings add this to your profile
```bash
export PYTHONWARNINGS="ignore::UserWarning:torchvision.transforms.v2,ignore::UserWarning:torchvision.datapoints"
```