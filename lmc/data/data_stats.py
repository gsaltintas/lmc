
from enum import Enum
from typing import Callable, Dict

import numpy as np
from torchvision import datasets as D

# Define all the relevant stats for the datasets to look up
import lmc.data.nlp_data_stats as NDS
import lmc.data.vision_data_stats as VDS

# Number of samples
SAMPLE_DICT = VDS.SAMPLE_DICT | NDS.SAMPLE_DICT
# Number of classes
CLASS_DICT = VDS.CLASS_DICT | NDS.CLASS_DICT
IS_GENERATION_TASK = VDS.IS_GENERATION_TASK | NDS.IS_GENERATION_TASK
TASK_MAPPING = VDS.TASK_MAPPING | NDS.TASK_MAPPING

### Vision only
# Number of channels
CHANNELS_DICT = VDS.CHANNELS_DICT
# Image resolutions
DEFAULT_RES_DICT = VDS.DEFAULT_RES_DICT

# Parent directory name
DATA_DICT = VDS.DATA_DICT
MODE_DICT = VDS.MODE_DICT

# Standardization statistics
MEAN_DICT = VDS.MEAN_DICT
STD_DICT = VDS.STD_DICT

# Whether dataset can be cached in memory, available in torch
OS_CACHED_DICT = VDS.OS_CACHED_DICT
TORCH_DICT = VDS.TORCH_DICT


### NLP only
# Vocabulary sizes for language models
VOCAB_SIZE_DICT = NDS.VOCAB_SIZE_DICT

# For language models, we might want sequence lengths
MAX_SEQ_LENGTH_DICT = NDS.MAX_SEQ_LENGTH_DICT

HUGGING_FACE_DICT = NDS.HUGGING_FACE_DICT

# Dataset configurations where needed
HF_CONFIG_DICT = NDS.HF_CONFIG_DICT

DATASET_SPLITS = NDS.DATASET_SPLITS


