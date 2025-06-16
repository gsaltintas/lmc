# The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions

This repository contains the code for reproducing experiments from our ICML 2025 paper investigating the sensitivity of neural network training to initial conditions.

## Setup

### Installation

We recommend using `uv` for installation, call
```bash
uv sync
```

### Environment Variables

Add the following variables to your `.bashrc` file or export them for each run:

```bash
export WANDB_ENTITY="YOUR_WANDB_ENTITY"
export WANDB_API_KEY="YOUR_API_KEY"
export SCRATCH="PATH_TO_SCRATCH_DIRECTORY"
export HUGGINGFACE_HUB_CACHE="$SCRATCH/.cache"
```

## Running Butterfly Experiments

All scripts used to produce results in the paper can be found under [lmc/scripts](lmc/scripts). 

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{altintas2025butterfly,
  title={The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions},
  author={Altıntaş, Gül Sena and Kwok, Devin and Raffel, Colin and Rolnick, David},
  booktitle    = {Forty-second International Conference on Machine Learning, {ICML} 2025,
                  Vancouver, Canada, July 13-19, 2025},
  publisher={OpenReview.net}
  year={2025},
  url={https://openreview.net/forum?id=L1Bm396P0X},
}
```
  
