"""
"""

from enum import Enum
from typing import Callable, Dict

import numpy as np
from torchvision import datasets as D

from .utils import TaskType

# Define all the relevant stats for the datasets to look up

# Dataset sizes for NLP (training examples)
SAMPLE_DICT = {
    "wikitext-2": 2088628,
    "wikitext-103": 103227021,
    "squad": 87599,
    "glue": {
        "cola": 8551,
        "sst2": 67349,
        "mnli": 392702,
        "qqp": 363849,
        "qnli": 104743,
        "rte": 2490,
        "wnli": 635,
        "mrpc": 3668,
        "stsb": 5749
    },
    "cord": 800,
    "webtext": 8013769,
    "c4": 364868892,
    "pile": 825000000,
    "bookcorpus": 74004228,
    
    "qasc": 8134,
    "wikiqa": 20360,
    "quartz": 2696,
    "paws": 49401,
    "story_cloze": 87866,
    "winogrande": 40398,
    "wsc": 554,
    
    # CR tasks
    "snli": 550000,
    "scitail": 27000,
    
    # QA tasks
    "squad_v2": 162000,
    "newsqa": 120000,
    "hotpotqa": 113000,
    "squad_v1": 108000,
    "duorc_p": 100000,
    "duorc_s": 86000,
    "drop": 77000,
    "wikihop": 51000,
    "boolq": 16000,
    "comqa": 11000,
    
    # SL tasks
    "conll2003": 14000,  # NER
    "ptb": 38000,        # POS-PTB
    "conj": 13000,        # Conjunction
}

# Number of classes
CLASS_DICT = {
    # GLUE tasks
    "glue/cola": 2,        # Binary classification (grammatical acceptability)
    "glue/sst2": 2,        # Binary classification (sentiment)
    "glue/mrpc": 2,        # Binary classification (paraphrase)
    "glue/qqp": 2,         # Binary classification (question pairs)
    "glue/mnli": 3,        # 3-way classification (NLI)
    "glue/qnli": 2,        # Binary classification (QA/NLI)
    "glue/rte": 2,         # Binary classification (NLI)
    "glue/wnli": 2,        # Binary classification (NLI)
    "glue/stsb": 1,        # Regression task (similarity scoring)
    
    # Other classification tasks
    "squad": 2,            # Binary classification for answerable/not answerable
    "cord": 30,           # Number of entity types for NER
    
    # Generation tasks (vocab sizes)
    "wikitext-2": 33278,    # Vocabulary size
    "wikitext-103": 267735, # Vocabulary size
    "webtext": 50257,       # GPT-2 vocab size
    "c4": 32000,           # T5 vocab size
    "pile": 50400,         # GPT-3 vocab size
    "bookcorpus": 30000,    # Approximate vocab size
    
    "qasc": 8,          # Multiple choice with 8 options
    "wikiqa": 2,        # Binary classification for answer selection
    "quartz": 4,        # Multiple choice with 4 options
    "paws": 2,          # Binary classification for paraphrase detection
    "story_cloze": 2,   # Binary choice for story ending
    "winogrande": 2,    # Binary choice for coreference resolution
    "wsc": 2,          # Binary choice for coreference resolution
    
    
    # CR tasks
    "snli": 3,        # Entailment, contradiction, neutral
    "scitail": 2,     # Entailment or neutral
    
    # QA tasks (typically span prediction or classification)
    "squad_v2": 2,    # Has answer or no answer
    "newsqa": 2,
    "hotpotqa": 2,
    "squad_v1": 2,
    "duorc": 2,
    "drop": 2,
    "wikihop": 2,
    "boolq": 2,
    "comqa": 2,
    
    # SL tasks (number of labels)
    "conll2003": 9,   # NER labels
    "ptb": 45,        # POS tags
    "conj": 10 ,       # Conjunction types
}


IS_GENERATION_TASK = {
    # NLP tasks
    "wikitext-2": True,
    "wikitext-103": True,
    "squad": False,  # QA task
    "glue": False,   # Classification/Regression
    "cord": False,   # Information extraction
    "webtext": True,
    "c4": True,
    "pile": True,
    "bookcorpus": True,
    
    "qasc": False,
    "wikiqa": False,
    "quartz": False,
    "paws": False,
    "story_cloze": False,
    "winogrande": False,
    "wsc": False,
    
    
    "snli": False,
    "scitail": False,
    "squad_v2": False,
    "newsqa": False,
    "hotpotqa": False,
    "squad_v1": False,
    "duorc": False,
    "drop": False,
    "wikihop": False,
    "boolq": False,
    "comqa": False,
    "conll2003": False,
    "ptb": False,
    "conj": False
}

# Vocabulary sizes for language models
VOCAB_SIZE_DICT = {
    "wikitext-2": 33278,
    "wikitext-103": 267735,
    "webtext": 50257,  # GPT-2 vocab size
    "c4": 32000,
    "pile": 50400,
    "bookcorpus": 30000
}



# For language models, we might want sequence lengths
MAX_SEQ_LENGTH_DICT = {
    "wikitext-2": 512,
    "wikitext-103": 512,
    "squad": 384,
    "glue": 128,
    "cord": 512,
    "webtext": 1024,
    "c4": 512,
    "pile": 2048,
    "bookcorpus": 512,
    
    "qasc": 512,
    "wikiqa": 384,
    "quartz": 512,
    "paws": 128,
    "story_cloze": 256,
    "winogrande": 256,
    "wsc": 256,
    
    
    # CR tasks usually have shorter sequences
    "snli": 128,
    "scitail": 128,
    
    # QA tasks often need longer sequences
    "squad_v2": 384,
    "newsqa": 512,
    "hotpotqa": 512,
    "squad_v1": 384,
    "duorc": 512,
    "drop": 512,
    "wikihop": 512,
    "boolq": 256,
    "comqa": 512,
    
    # SL tasks
    "conll2003": 128,
    "ptb": 128,
    "conj": 128
}
HUGGING_FACE_DICT = {
    # Text generation/language modeling datasets
    "wikitext-2": "wikitext",         # Will use config "wikitext-2-raw-v1"
    "wikitext-103": "wikitext",       # Will use config "wikitext-103-raw-v1"
    "webtext": "openwebtext",
    "c4": "c4",                       # Will use config "en"
    "pile": "EleutherAI/pile",
    "bookcorpus": "bookcorpus",
    
    # Question answering/classification datasets
    "squad": "squad",
    "glue": "glue",                   # Will need task specified #TODO CHECK
    "cord-v2": "naver-clova-ix/cord-v2",
    "cord": "naver-clova-ix/cord-v2",
    
    
    "qasc": "allenai/qasc",
    "wikiqa": "wiki_qa",
    "quartz": "allenai/quartz",
    "paws": "paws",
    "story_cloze": "story_cloze",
    "winogrande": "winogrande",
    "wsc": "super_glue/wsc",
    
    
    # Text Classification/Regression (CR)
    "snli": "stanfordnlp/snli",
    "scitail": "scitail",
    # QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE, WNLI already in GLUE
    
    # Question Answering (QA)
    "squad_v2": "squad_v2",
    "squad_v1": "squad",
    "newsqa": "newsqa",
    "hotpotqa": "hotpot_qa",
    "duorc": "duorc",
    "drop": "drop",
    "wikihop": "hotpot_qa",
    "boolq": "super_glue/boolq",
    "comqa": "comqa",
    
    # Sequence Labeling (SL)
    "conll2003": "conll2003",  # For NER
    "ptb": "ptb_text_only",    # For POS tagging
    "conj": "conll2009",       # For Conjunction
}

# Dataset configurations where needed
HF_CONFIG_DICT = {
        "wikitext-2": "wikitext-2-raw-v1",
        "wikitext-103": "wikitext-103-raw-v1",
        "c4": "en"
    }

DATASET_SPLITS = {
    # GLUE tasks
    "glue": {
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    # Question Answering
    "squad": {
        "train": "train",
        "validation": "validation"  # SQuAD uses validation as test
    },
    # Language modeling
    "wikitext-2": {
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    "wikitext-103": {
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    "webtext": {
        "train": "train"  # Only has train split
    },
    "c4": {
        "train": "train",
        "validation": "validation"
    },
    "pile": {
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    "bookcorpus": {
        "train": "train"  # Only has train split
    },
    "cord": {
        
    },
    
    "qasc": {
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    "wikiqa": {
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    "quartz": {
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    "paws": {
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    "story_cloze": {
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    "winogrande": {
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    "wsc": {
        "train": "train",
        "validation": "validation",
        "test": "test"
    }
}


TASK_MAPPING = {
    ## vision
     "imagenet21": TaskType.CLASSIFICATION,
    "imagenet": TaskType.CLASSIFICATION,
    "tinyimagenet": TaskType.CLASSIFICATION,
    "cifar10": TaskType.CLASSIFICATION,
    "cifar100": TaskType.CLASSIFICATION,
    "mnist": TaskType.CLASSIFICATION,
    "stl10": TaskType.CLASSIFICATION,
    "cinic10": TaskType.CLASSIFICATION,
    "cinic10_wo_cifar10": TaskType.CLASSIFICATION,
    
    # Language Modeling/Generation
    "wikitext-2": TaskType.GENERATION,
    "wikitext-103": TaskType.GENERATION,
    "webtext": TaskType.GENERATION,
    "c4": TaskType.GENERATION,
    "pile": TaskType.GENERATION,
    "bookcorpus": TaskType.GENERATION,
    
    # Classification/Regression
    "cord": TaskType.CLASSIFICATION,
    "cord-v2": TaskType.CLASSIFICATION,
    "scitail": TaskType.NATURAL_LANGUAGE_INFERENCE,
    "snli": TaskType.NATURAL_LANGUAGE_INFERENCE,
    
    # Question Answering
    "squad": TaskType.QUESTION_ANSWERING,
    "squad_v1": TaskType.QUESTION_ANSWERING,
    "squad_v2": TaskType.QUESTION_ANSWERING,
    "newsqa": TaskType.QUESTION_ANSWERING,
    "hotpotqa": TaskType.QUESTION_ANSWERING,
    "duorc": TaskType.QUESTION_ANSWERING,
    "drop": TaskType.QUESTION_ANSWERING,
    "wikihop": TaskType.QUESTION_ANSWERING,
    "boolq": TaskType.QUESTION_ANSWERING,
    "comqa": TaskType.QUESTION_ANSWERING,
    
    # Sequence Labeling
    "conll2003": TaskType.SEQUENCE_LABELING,
    "ptb": TaskType.SEQUENCE_LABELING,
    "conj": TaskType.SEQUENCE_LABELING,
    
    # GLUE and its tasks
    "glue": TaskType.SEQUENCE_PAIR,
    "glue/cola": TaskType.CLASSIFICATION,
    "glue/sst2": TaskType.CLASSIFICATION,
    "glue/mrpc": TaskType.SEQUENCE_PAIR,
    "glue/qqp": TaskType.SEQUENCE_PAIR,
    "glue/mnli": TaskType.NATURAL_LANGUAGE_INFERENCE,
    "glue/qnli": TaskType.NATURAL_LANGUAGE_INFERENCE,
    "glue/rte": TaskType.NATURAL_LANGUAGE_INFERENCE,
    "glue/wnli": TaskType.NATURAL_LANGUAGE_INFERENCE,
    "glue/stsb": TaskType.SEQUENCE_PAIR,  # Regression task
}