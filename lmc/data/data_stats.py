"""Dataset statistics and configurations for both vision and language tasks."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Optional, Union

import numpy as np
from torchvision import datasets as D
from yaml import Loader, load

logger = logging.getLogger(__name__)


def get_instruction_formats():
    p = Path(__file__).parent / "instruction_format.yaml"
    with open(p, "r") as f:
        dct = load(f, Loader=Loader)
    return dct


instruction_formats = get_instruction_formats()


class TaskType(Enum):
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    QUESTION_ANSWERING = "question_answering"
    SEQUENCE_PAIR = "sequence_pair"
    NATURAL_LANGUAGE_INFERENCE = "natural_language_inference"
    SEQUENCE_LABELING = "sequence_labeling"
    REGRESSION = "regression"
    SEGMENTATION = "segmentation"
    INSTRUCTION = "instruction"


# Vision Registry
@dataclass
class VisionConfig:
    samples: int
    classes: int
    channels: int
    resolution: int
    mean: np.ndarray
    std: np.ndarray
    can_cache: bool
    torch_dataset: Optional[Union[str, type]] = None
    hf_path: str = None
    hf_config: Optional[str] = None
    task_type: TaskType = TaskType.CLASSIFICATION


# Language Registry
@dataclass
class LanguageConfig:
    samples: Union[int, Dict[str, int]]
    classes: Optional[int]
    task_type: TaskType
    max_seq_length: int
    hf_path: str
    max_gen_seq_length: int = None
    hf_config: Optional[str] = None
    splits: Dict[str, str] = None
    vocab_size: Optional[int] = None
    metrics: List[str] = field(default_factory=list)  # Add this field
    trust_remote_code: bool = False
    dataset_text_field: str = "text"
    dataset_label_field: str = "label"

    def is_generation(self):
        return self.task_type == TaskType.GENERATION

    def __post_init__(self):
        if self.max_gen_seq_length is None:
            self.max_gen_seq_length = self.max_seq_length
        inst_format = instruction_formats.get(self.hf_path, {})
        if self.hf_config:
            inst_format = inst_format.get(self.hf_config)
        # import code

        # code.interact(local=locals() | globals())
        self.instruction_format = inst_format


@dataclass
class BaseRegistry:
    """Base class for dataset registries."""

    _registry: ClassVar[Dict[str, Union[VisionConfig, LanguageConfig]]] = {}

    def __post_init__(self):
        for key, val in self._registry.items():
            if not hasattr(self, key.replace("-", "_")):
                logger.warn(
                    "Dataset %s not properly set as attribute, doing it now.", key
                )
                setattr(self, key, val)

    @classmethod
    def get(cls, name: str) -> Union[VisionConfig, LanguageConfig]:
        if name not in cls._registry:
            raise ValueError(f"Dataset {name} not found in {cls.__name__}")
        return cls._registry[name]

    @classmethod
    def get_datasets_by_task(
        cls, task_type: TaskType
    ) -> Dict[str, Union[VisionConfig, LanguageConfig]]:
        """Get all datasets of a specific task type."""
        return {
            name: config
            for name, config in cls._registry.items()
            if getattr(config, "task_type", None) == task_type
        }

    @classmethod
    def get_samples(cls, dataset_name: str) -> int:
        """Get number of samples for a dataset."""
        return cls.get(dataset_name).samples

    @classmethod
    def get_mixture_info(cls, dataset_names: List[str]) -> Dict:
        """Get combined information for a mixture of datasets."""
        configs = [cls.get(name) for name in dataset_names]

        # Get common properties
        task_types = set(getattr(config, "task_type", None) for config in configs)
        total_samples = sum(config.samples for config in configs)

        return {
            "total_samples": total_samples,
            "task_types": task_types,
            "individual_configs": {
                name: config for name, config in zip(dataset_names, configs)
            },
        }

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        return list(cls._registry.keys())


class VisionRegistry(BaseRegistry):
    """Registry for vision datasets."""

    _registry: ClassVar[Dict[str, VisionConfig]] = {
        "imagenet21": VisionConfig(
            samples=11801680,
            classes=11230,
            channels=3,
            resolution=64,
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225]),
            can_cache=False,
            torch_dataset=None,
            task_type=TaskType.CLASSIFICATION,
        ),
        "imagenet1k": VisionConfig(
            samples=1281167,
            classes=1000,
            channels=3,
            resolution=64,
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225]),
            can_cache=False,
            torch_dataset="lmc.data.imagenet.ImageNet",
            task_type=TaskType.CLASSIFICATION,
        ),
        "imagenet": VisionConfig(
            samples=1281167,
            classes=1000,
            channels=3,
            resolution=64,
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225]),
            can_cache=False,
            torch_dataset=D.ImageNet,
            task_type=TaskType.CLASSIFICATION,
        ),
        "tinyimagenet": VisionConfig(
            samples=100000,
            classes=200,
            channels=3,
            resolution=64,
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225]),
            can_cache=False,
            torch_dataset="lmc.data.TinyImageNet",
            task_type=TaskType.CLASSIFICATION,
        ),
        "cifar10": VisionConfig(
            samples=50000,
            classes=10,
            channels=3,
            resolution=32,
            mean=np.array([0.49139968, 0.48215827, 0.44653124]),
            std=np.array([0.24703233, 0.24348505, 0.26158768]),
            can_cache=True,
            torch_dataset=D.CIFAR10,
            task_type=TaskType.CLASSIFICATION,
        ),
        "cifar10_random_labels": VisionConfig(
            samples=50000,
            classes=10,
            channels=3,
            resolution=32,
            mean=np.array([0.49139968, 0.48215827, 0.44653124]),
            std=np.array([0.24703233, 0.24348505, 0.26158768]),
            can_cache=True,
            torch_dataset=D.CIFAR10,
            task_type=TaskType.CLASSIFICATION,
        ),
        "cifar10easy": VisionConfig(
            samples=25000,
            classes=10,
            channels=3,
            resolution=32,
            mean=np.array([0.49139968, 0.48215827, 0.44653124]),
            std=np.array([0.24703233, 0.24348505, 0.26158768]),
            can_cache=True,
            torch_dataset="lmc.data.cifar10split.CIFAR10_EL2N_EASY",
            task_type=TaskType.CLASSIFICATION,
        ),
        "cifar10hard": VisionConfig(
            samples=25000,
            classes=10,
            channels=3,
            resolution=32,
            mean=np.array([0.49139968, 0.48215827, 0.44653124]),
            std=np.array([0.24703233, 0.24348505, 0.26158768]),
            can_cache=True,
            torch_dataset="lmc.data.cifar10split.CIFAR10_EL2N_HARD",
            task_type=TaskType.CLASSIFICATION,
        ),
        "cifar100": VisionConfig(
            samples=50000,
            classes=100,
            channels=3,
            resolution=32,
            mean=np.array([0.49139968, 0.48215827, 0.44653124]),
            std=np.array([0.24703233, 0.24348505, 0.26158768]),
            can_cache=False,
            torch_dataset=D.CIFAR100,
            task_type=TaskType.CLASSIFICATION,
        ),
        "eurosat": VisionConfig(
            samples=27000,  # EuroSAT has ~27k images
            classes=10,  # 10 land use classes
            channels=3,  # RGB images
            resolution=64,  # Original is 64x64
            mean=np.array([0.34677, 0.35926, 0.36816]),  # EuroSAT RGB means
            std=np.array([0.13331, 0.12630, 0.13832]),  # EuroSAT RGB stds
            can_cache=True,
            torch_dataset=D.EuroSAT,  # Will define a custom dataset class
            task_type=TaskType.CLASSIFICATION,
        ),
        "mnist": VisionConfig(
            samples=60000,
            classes=10,
            channels=1,
            resolution=28,
            mean=np.array([0.1307]),
            std=np.array([0.3081]),
            can_cache=True,
            torch_dataset=D.MNIST,
            task_type=TaskType.CLASSIFICATION,
        ),
        "stl10": VisionConfig(
            samples=5000,
            classes=10,
            channels=3,
            resolution=64,
            mean=np.array([0.4914, 0.4822, 0.4465]),
            std=np.array([0.2471, 0.2435, 0.2616]),
            can_cache=True,
            torch_dataset=D.STL10,
            task_type=TaskType.CLASSIFICATION,
        ),
        "cinic10": VisionConfig(
            samples=180000,
            classes=10,
            channels=3,
            resolution=32,
            mean=np.array([0.47889522, 0.47227842, 0.43047404]),
            std=np.array([0.24205776, 0.23828046, 0.25874835]),
            can_cache=False,
            torch_dataset="lmc.data.cinic10.CINIC10",
            task_type=TaskType.CLASSIFICATION,
            # hf_path="flwrlabs/cinic10",
        ),
        "cinic10_wo_cifar10": VisionConfig(
            samples=130000,
            classes=10,
            channels=3,
            resolution=32,
            mean=np.array([0.47889522, 0.47227842, 0.43047404]),
            std=np.array([0.24205776, 0.23828046, 0.25874835]),
            can_cache=False,
            torch_dataset="lmc.data.cinic10.CINIC10_WO_CIFAR10",
            task_type=TaskType.CLASSIFICATION,
        ),
        "ade20k": VisionConfig(
            samples=20210,  # ADE20K has ~20k training images
            classes=150,  # 150 semantic categories
            channels=3,  # RGB images
            resolution=384,  # Common input resolution for ADE20K
            mean=np.array([0.485, 0.456, 0.406]),  # ImageNet mean as default
            std=np.array([0.229, 0.224, 0.225]),  # ImageNet std as default
            can_cache=False,
            torch_dataset="lmc.data.ade20k.ADE20KDataset",  # Will define a custom dataset class
            task_type=TaskType.SEGMENTATION,
        ),
    }

    imagenet21: ClassVar[VisionConfig] = _registry["imagenet21"]
    imagenet: ClassVar[VisionConfig] = _registry["imagenet"]
    imagenet1k: ClassVar[VisionConfig] = _registry["imagenet1k"]
    tinyimagenet: ClassVar[VisionConfig] = _registry["tinyimagenet"]
    cifar10: ClassVar[VisionConfig] = _registry["cifar10"]
    cifar10easy: ClassVar[VisionConfig] = _registry["cifar10easy"]
    cifar10hard: ClassVar[VisionConfig] = _registry["cifar10hard"]
    cifar100: ClassVar[VisionConfig] = _registry["cifar100"]
    cifar10_random_labels: ClassVar[VisionConfig] = _registry["cifar10_random_labels"]
    mnist: ClassVar[VisionConfig] = _registry["mnist"]
    stl10: ClassVar[VisionConfig] = _registry["stl10"]
    cinic10: ClassVar[VisionConfig] = _registry["cinic10"]
    eurosat: ClassVar[VisionConfig] = _registry["eurosat"]
    ade20k: ClassVar[VisionConfig] = _registry["ade20k"]
    cinic10_wo_cifar10: ClassVar[VisionConfig] = _registry["cinic10_wo_cifar10"]


@dataclass
class MathDatasetRegistry(BaseRegistry):
    """Registry for mathematical reasoning datasets."""

    _registry: ClassVar[Dict[str, LanguageConfig]] = {
        "gsm8k": LanguageConfig(
            samples=7473,  # Exact train count
            classes=None,  # Generation task
            task_type=TaskType.GENERATION,
            max_seq_length=1024,
            hf_path="openai/gsm8k",
            hf_config="main",
            splits={"train": "train", "validation": "test"},
            metrics=["exact_match"],
            max_gen_seq_length=128,
            dataset_text_field="question",
            dataset_label_field="answer",
        ),
        "math": LanguageConfig(
            samples=1744,  # {"train": 7500, "test": 5000},
            classes=None,
            task_type=TaskType.GENERATION,
            max_seq_length=2048,  # Longer sequences for theorem proofs
            hf_path="EleutherAI/hendrycks_math",
            hf_config="algebra",
            splits={"train": "train", "validation": "test"},
            metrics=["exact_match"],
        ),
        "mathqa": LanguageConfig(
            samples=29838,  # 37200,  # Total dataset size
            classes=5,  # Multiple choice answers
            task_type=TaskType.GENERATION,
            # task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=512,
            hf_path="allenai/math_qa",
            splits={"train": "train", "validation": "validation", "test": "test"},
            metrics=["accuracy", "f1"],
            trust_remote_code=True,
        ),
        "asdiv": LanguageConfig(
            samples=1839,  # {
            #     "train": 1839,
            #     "validation": 230,
            #     "test": 230,
            # },  # Total ~2.3K samples
            classes=None,  # Generation task
            task_type=TaskType.GENERATION,
            max_seq_length=256,  # Problems are relatively short
            hf_path="EleutherAI/asdiv",
            splits={
                "validation": "validation",
            },
            metrics=["exact_match"],  # Using exact match for evaluation
            trust_remote_code=True,
        ),
    }
    gsm8k: ClassVar[LanguageConfig] = _registry["gsm8k"]
    math: ClassVar[LanguageConfig] = _registry["math"]
    mathqa: ClassVar[LanguageConfig] = _registry["mathqa"]
    asdiv: ClassVar[LanguageConfig] = _registry["asdiv"]


class QARegistry(BaseRegistry):
    """Registry for question answering datasets."""

    _registry: ClassVar[Dict[str, LanguageConfig]] = {
        "squad": LanguageConfig(
            samples=87599,
            classes=2,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=384,
            hf_path="squad",
            splits={"train": "train", "validation": "validation"},
        ),
        "squad_v2": LanguageConfig(
            samples=162000,
            classes=2,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=384,
            hf_path="squad_v2",
            splits={"train": "train", "validation": "validation"},
        ),
        "squad_v1": LanguageConfig(
            samples=108000,
            classes=2,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=384,
            hf_path="squad",
            splits={"train": "train", "validation": "validation"},
        ),
        "newsqa": LanguageConfig(
            samples=120000,
            classes=2,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=512,
            hf_path="newsqa",
            splits={"train": "train", "validation": "validation"},
        ),
        "hotpotqa": LanguageConfig(
            samples=113000,
            classes=2,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=512,
            hf_path="hotpot_qa",
            splits={"train": "train", "validation": "validation"},
        ),
        "duorc": LanguageConfig(
            samples=86000,  # Using duorc_s value
            classes=2,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=512,
            hf_path="duorc",
            splits={"train": "train", "validation": "validation"},
        ),
        # "drop": LanguageConfig(
        #     samples=77000,
        #     # classes=2,
        #     classes=None,
        #     task_type=TaskType.QUESTION_ANSWERING,
        #     max_seq_length=512,
        #     hf_path="drop",
        #     splits={"train": "train", "validation": "validation"},
        # ),
        "wikihop": LanguageConfig(
            samples=51000,
            classes=2,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=512,
            hf_path="hotpot_qa",
            splits={"train": "train", "validation": "validation"},
        ),
        "boolq": LanguageConfig(
            samples=16000,
            classes=2,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=256,
            hf_path="super_glue/boolq",
            splits={"train": "train", "validation": "validation"},
        ),
        "comqa": LanguageConfig(
            samples=11000,
            classes=2,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=512,
            hf_path="comqa",
            splits={"train": "train", "validation": "validation"},
        ),
        "qasc": LanguageConfig(
            samples=8134,
            classes=8,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=512,
            hf_path="allenai/qasc",
            splits={"train": "train", "validation": "validation", "test": "test"},
        ),
        "wikiqa": LanguageConfig(
            samples=20360,
            classes=2,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=384,
            hf_path="wiki_qa",
            splits={"train": "train", "validation": "validation", "test": "test"},
        ),
        "quartz": LanguageConfig(
            samples=2696,
            classes=4,
            task_type=TaskType.QUESTION_ANSWERING,
            max_seq_length=512,
            hf_path="allenai/quartz",
            splits={"train": "train", "validation": "validation", "test": "test"},
        ),
        # todo
        "mmlu": LanguageConfig(
            samples=15908,  # {"dev": 15908, "test": 14042},
            classes=4,  # Multiple choice (A, B, C, D)
            task_type=TaskType.GENERATION,
            max_seq_length=512,
            hf_path="cais/mmlu",
            splits={
                "train": "auxiliary_train",  # Note: MMLU uses few-shot evaluation
                "validation": "dev",
                "test": "test",
            },
            metrics=["accuracy"],
        ),
    }

    squad: ClassVar[LanguageConfig] = _registry["squad"]
    squad_v2: ClassVar[LanguageConfig] = _registry["squad_v2"]
    squad_v1: ClassVar[LanguageConfig] = _registry["squad_v1"]
    newsqa: ClassVar[LanguageConfig] = _registry["newsqa"]
    hotpotqa: ClassVar[LanguageConfig] = _registry["hotpotqa"]
    duorc: ClassVar[LanguageConfig] = _registry["duorc"]
    # drop: ClassVar[LanguageConfig] = _registry["drop"]
    wikihop: ClassVar[LanguageConfig] = _registry["wikihop"]
    boolq: ClassVar[LanguageConfig] = _registry["boolq"]
    comqa: ClassVar[LanguageConfig] = _registry["comqa"]
    qasc: ClassVar[LanguageConfig] = _registry["qasc"]
    wikiqa: ClassVar[LanguageConfig] = _registry["wikiqa"]
    quartz: ClassVar[LanguageConfig] = _registry["quartz"]
    mmlu: ClassVar[LanguageConfig] = _registry["mmlu"]


class NLIRegistry(BaseRegistry):
    """Registry for natural language inference datasets."""

    _registry: ClassVar[Dict[str, LanguageConfig]] = {
        "snli": LanguageConfig(
            samples=550000,
            classes=3,
            task_type=TaskType.NATURAL_LANGUAGE_INFERENCE,
            max_seq_length=128,
            hf_path="stanfordnlp/snli",
            splits={"train": "train", "validation": "validation", "test": "test"},
        ),
        "scitail": LanguageConfig(
            samples=27000,
            classes=2,
            task_type=TaskType.NATURAL_LANGUAGE_INFERENCE,
            max_seq_length=128,
            hf_path="scitail",
            splits={"train": "train", "validation": "validation", "test": "test"},
        ),
    }
    snli: ClassVar[LanguageConfig] = _registry["snli"]
    scitail: ClassVar[LanguageConfig] = _registry["scitail"]


class GenerationRegistry(BaseRegistry):
    """Registry for text generation datasets."""

    _registry: ClassVar[Dict[str, LanguageConfig]] = {
        "wikitext-2": LanguageConfig(
            samples=2088628,
            classes=None,
            vocab_size=33278,
            task_type=TaskType.GENERATION,
            max_seq_length=512,
            hf_path="wikitext",
            hf_config="wikitext-2-raw-v1",
            splits={"train": "train", "validation": "validation", "test": "test"},
        ),
        "wikitext-103": LanguageConfig(
            samples=103227021,
            classes=None,
            vocab_size=267735,
            task_type=TaskType.GENERATION,
            max_seq_length=512,
            hf_path="wikitext",
            hf_config="wikitext-103-raw-v1",
            splits={"train": "train", "validation": "validation", "test": "test"},
        ),
        "webtext": LanguageConfig(
            samples=8013769,
            classes=None,
            vocab_size=50257,
            task_type=TaskType.GENERATION,
            max_seq_length=1024,
            hf_path="openwebtext",
            splits={"train": "train"},
        ),
        "c4": LanguageConfig(
            samples=364868892,
            classes=None,
            vocab_size=32000,
            task_type=TaskType.GENERATION,
            max_seq_length=512,
            hf_path="c4",
            hf_config="en",
            splits={"train": "train", "validation": "validation"},
        ),
        "pile": LanguageConfig(
            samples=825000000,
            classes=None,
            vocab_size=50400,
            task_type=TaskType.GENERATION,
            max_seq_length=2048,
            hf_path="EleutherAI/pile",
            splits={"train": "train", "validation": "validation", "test": "test"},
        ),
        "bookcorpus": LanguageConfig(
            samples=74004228,
            classes=None,
            vocab_size=30000,
            task_type=TaskType.GENERATION,
            max_seq_length=512,
            hf_path="bookcorpus",
            splits={"train": "train"},
        ),
    }

    wikitext_2: ClassVar[LanguageConfig] = _registry["wikitext-2"]
    wikitext_103: ClassVar[LanguageConfig] = _registry["wikitext-103"]
    webtext: ClassVar[LanguageConfig] = _registry["webtext"]
    c4: ClassVar[LanguageConfig] = _registry["c4"]
    pile: ClassVar[LanguageConfig] = _registry["pile"]
    bookcorpus: ClassVar[LanguageConfig] = _registry["bookcorpus"]


class InstructionTuningRegistry(BaseRegistry):
    _registry: ClassVar[Dict[str, LanguageConfig]] = {
        "alpaca": LanguageConfig(
            samples=52002,  # Alpaca dataset has 52K instruction-following examples
            classes=None,  # Not a classification task
            vocab_size=None,  # Use default vocab size from the model
            task_type=TaskType.INSTRUCTION,  # Instruction-based task
            max_seq_length=512,  # Standard context length
            hf_path="tatsu-lab/alpaca",  # Hugging Face dataset path
            max_gen_seq_length=128,  # Reasonable generation length for instructions
            splits={"train": "train"},  # Only has a train split
            metrics=["exact_match", "rouge"],  # Common metrics for instruction tuning
            trust_remote_code=False,
        ),
        # "tulu-3-sft-mixture": LanguageConfig(
        #     samples=
        # )
        "tulu-3-sft-personas-instruction-following": LanguageConfig(
            samples=29980,
            classes=None,
            vocab_size=None,
            task_type=TaskType.INSTRUCTION,  # Instruction-based task
            max_seq_length=512,  # Standard context length
            hf_path="allenai/tulu-3-sft-personas-instruction-following",
            splits={"train": "train"},  # Only has a train split
            metrics=["exact_match", "rouge"],  # Common metrics for instruction tuning
            trust_remote_code=True,
        ),
        "drop": LanguageConfig(
            samples=86935,
            hf_path="ucinlp/drop",
            vocab_size=None,
            classes=None,
            task_type=TaskType.INSTRUCTION,  # Instruction-based task
            max_seq_length=512,  # Standard context length
            splits={"train": "train", "test": "validation"},
            metrics=[],  # TODO
        ),
    }
    alpaca: ClassVar[LanguageConfig] = _registry["alpaca"]


class GLUERegistry(BaseRegistry):
    """Registry for GLUE benchmark datasets."""

    _registry: ClassVar[Dict[str, LanguageConfig]] = {
        "cola": LanguageConfig(
            samples=8551,
            classes=2,
            task_type=TaskType.CLASSIFICATION,
            max_seq_length=128,
            hf_path="nyu-mll/glue",
            hf_config="cola",
            splits={"train": "train", "validation": "validation", "test": "test"},
            metrics=[
                "matthews_correlation",
                "accuracy",
            ],  # CoLA uses Matthews Correlation
        ),
        "sst2": LanguageConfig(
            samples=67349,
            classes=2,
            task_type=TaskType.CLASSIFICATION,
            max_seq_length=128,
            hf_path="nyu-mll/glue",
            hf_config="sst2",
            splits={"train": "train", "validation": "validation", "test": "test"},
            metrics=["accuracy"],  # SST-2 uses accuracy
        ),
        "mrpc": LanguageConfig(
            samples=3668,
            classes=2,
            task_type=TaskType.SEQUENCE_PAIR,
            max_seq_length=128,
            hf_path="nyu-mll/glue",
            hf_config="mrpc",
            splits={"train": "train", "validation": "validation", "test": "test"},
            metrics=["accuracy", "f1"],  # MRPC uses both accuracy and F1
        ),
        "qqp": LanguageConfig(
            samples=363849,
            classes=2,
            task_type=TaskType.SEQUENCE_PAIR,
            max_seq_length=128,
            hf_path="nyu-mll/glue",
            hf_config="qqp",
            splits={"train": "train", "validation": "validation", "test": "test"},
            metrics=["accuracy", "f1"],  # QQP uses both accuracy and F1
        ),
        "mnli": LanguageConfig(
            samples=392702,
            classes=3,
            task_type=TaskType.NATURAL_LANGUAGE_INFERENCE,
            max_seq_length=128,
            hf_path="nyu-mll/glue",
            hf_config="mnli",
            splits={
                "train": "train",
                "validation": "validation_matched",
                "test": "test_matched",
            },
            metrics=["accuracy"],
        ),
        "qnli": LanguageConfig(
            samples=104743,
            classes=2,
            task_type=TaskType.NATURAL_LANGUAGE_INFERENCE,
            max_seq_length=128,
            hf_path="nyu-mll/glue",
            hf_config="qnli",
            splits={"train": "train", "validation": "validation", "test": "test"},
            metrics=["accuracy"],
        ),
        "rte": LanguageConfig(
            samples=2490,
            classes=2,
            task_type=TaskType.NATURAL_LANGUAGE_INFERENCE,
            max_seq_length=128,
            hf_path="nyu-mll/glue",
            hf_config="rte",
            splits={"train": "train", "validation": "validation", "test": "test"},
            metrics=["accuracy"],
        ),
        "wnli": LanguageConfig(
            samples=635,
            classes=2,
            task_type=TaskType.NATURAL_LANGUAGE_INFERENCE,
            max_seq_length=128,
            hf_path="nyu-mll/glue",
            hf_config="wnli",
            splits={"train": "train", "validation": "validation", "test": "test"},
            metrics=["accuracy"],
        ),
        "stsb": LanguageConfig(
            samples=5749,
            classes=1,  # Regression task
            task_type=TaskType.REGRESSION,  # Must be regression            task_type=TaskType.SEQUENCE_PAIR,
            max_seq_length=128,
            hf_path="nyu-mll/glue",
            hf_config="stsb",
            splits={
                "train": "train",
                "validation": "validation",
                "test": "test_matched",
            },
            metrics=[
                "pearson_correlation",
                "spearman_correlation",
            ],  # STS-B uses Pearson and Spearman correlations
        ),
    }

    cola: ClassVar[LanguageConfig] = _registry["cola"]
    sst2: ClassVar[LanguageConfig] = _registry["sst2"]
    mrpc: ClassVar[LanguageConfig] = _registry["mrpc"]
    qqp: ClassVar[LanguageConfig] = _registry["qqp"]
    mnli: ClassVar[LanguageConfig] = _registry["mnli"]
    qnli: ClassVar[LanguageConfig] = _registry["qnli"]
    rte: ClassVar[LanguageConfig] = _registry["rte"]
    wnli: ClassVar[LanguageConfig] = _registry["wnli"]
    stsb: ClassVar[LanguageConfig] = _registry["stsb"]


class DatasetRegistry:
    """Main registry that coordinates all sub-registries."""

    vision: ClassVar[VisionRegistry] = VisionRegistry()
    qa: ClassVar[QARegistry] = QARegistry()
    nli: ClassVar[NLIRegistry] = NLIRegistry()
    glue: ClassVar[GLUERegistry] = GLUERegistry()
    math: ClassVar[MathDatasetRegistry] = MathDatasetRegistry()
    generation: ClassVar[GenerationRegistry] = GenerationRegistry()

    @classmethod
    def get_datasets_by_task(
        cls, task_type: TaskType
    ) -> Dict[str, Union[VisionConfig, LanguageConfig]]:
        """Get all datasets of a specific task type across all registries."""
        result = {}
        for registry in [
            cls.vision,
            cls.qa,
            cls.nli,
            cls.generation,
            cls.glue,
            cls.math,
        ]:
            result.update(registry.get_datasets_by_task(task_type))
        return result

    @classmethod
    def get_mixture_info(cls, dataset_names: List[str]) -> Dict:
        """Get combined information for a mixture of datasets across all registries."""
        configs = []
        for name in dataset_names:
            for registry in [
                cls.vision,
                cls.qa,
                cls.nli,
                cls.generation,
                cls.glue,
                cls.math,
            ]:
                try:
                    configs.append(registry.get(name))
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Dataset {name} not found in any registry")

        task_types = set(getattr(config, "task_type", None) for config in configs)
        total_samples = sum(config.samples for config in configs)

        return {
            "total_samples": total_samples,
            "task_types": task_types,
            "individual_configs": {
                name: config for name, config in zip(dataset_names, configs)
            },
        }

    @classmethod
    def get_all_registries(cls) -> List[BaseRegistry]:
        return [cls.vision, cls.qa, cls.generation, cls.nli, cls.glue, cls.math]

    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Union[VisionConfig, LanguageConfig]:
        """Get configuration for a single dataset."""
        # Try each registry
        for registry in cls.get_all_registries():
            if hasattr(registry, dataset_name):
                return registry.get(dataset_name)
        raise ValueError(f"Dataset {dataset_name} not found in any registry")

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        return [
            *cls.vision.get_available_datasets(),
            *cls.nli.get_available_datasets(),
            *cls.qa.get_available_datasets(),
            *cls.glue.get_available_datasets(),
            *cls.generation.get_available_datasets(),
            *cls.math.get_available_datasets(),
        ]

    @classmethod
    def get_all_configs(cls) -> Dict[str, Union[VisionConfig, LanguageConfig]]:
        return (
            cls.vision._registry
            | cls.nli._registry
            | cls.qa._registry
            | cls.glue._registry
            | cls.generation._registry
            | cls.math._registry
        )

    @classmethod
    def get_language_registry(cls) -> Dict[str, LanguageConfig]:
        return (
            cls.nli._registry
            | cls.qa._registry
            | cls.glue._registry
            | cls.generation._registry
            | cls.math._registry
        )


# https://github.com/allenai/open-instruct/blob/main/open_instruct/dataset_transformation.py#L105
CHAT_TEMPLATES = {
    "simple_concat_with_space": (
        "{% for message in messages %}"
        "{{ ' ' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_concat_with_new_line": (
        "{% for message in messages %}"
        "{{ '\n' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_chat": (
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "assistant_message_only": (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "zephyr": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo": "{{ eos_token }}{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
    "custom": "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}",
}

### Backward Compatibility


# Number of samples
SAMPLE_DICT = {
    key: conf.samples for key, conf in DatasetRegistry.get_all_configs().items()
}
# Number of classes
CLASS_DICT = {
    key: conf.classes for key, conf in DatasetRegistry.get_all_configs().items()
}
IS_GENERATION_TASK = {
    key: conf.task_type == TaskType.GENERATION
    for key, conf in DatasetRegistry.get_all_configs().items()
}
TASK_MAPPING = {
    key: conf.task_type for key, conf in DatasetRegistry.get_all_configs().items()
}

### Vision only
# Number of channels
CHANNELS_DICT = {
    key: conf.channels for key, conf in DatasetRegistry.vision._registry.items()
}
# Image resolutions
DEFAULT_RES_DICT = {
    key: conf.resolution for key, conf in DatasetRegistry.vision._registry.items()
}

# Standardization statistics
MEAN_DICT = {key: conf.mean for key, conf in DatasetRegistry.vision._registry.items()}
STD_DICT = {key: conf.std for key, conf in DatasetRegistry.vision._registry.items()}

# Whether dataset can be cached in memory, available in torch
OS_CACHED_DICT = {
    key: conf.can_cache for key, conf in DatasetRegistry.vision._registry.items()
}
TORCH_DICT = {
    key: conf.torch_dataset for key, conf in DatasetRegistry.vision._registry.items()
}

### NLP only
# Vocabulary sizes for language models
VOCAB_SIZE_DICT = {
    key: conf.vocab_size
    for key, conf in DatasetRegistry.get_language_registry().items()
}
MAX_SEQ_LENGTH_DICT = {
    key: conf.max_seq_length
    for key, conf in DatasetRegistry.get_language_registry().items()
}
HUGGING_FACE_DICT = {
    key: conf.hf_path for key, conf in DatasetRegistry.get_language_registry().items()
}
# Dataset configurations where needed
HF_CONFIG_DICT = {
    key: conf.hf_config for key, conf in DatasetRegistry.get_language_registry().items()
}
DATASET_SPLITS = {
    key: conf.splits for key, conf in DatasetRegistry.get_language_registry().items()
}

if __name__ == "__main__":
    config = VisionRegistry.cifar10
    print(config.samples)  # 50000

    # Or through the main registry
    config = DatasetRegistry.vision.cifar10
    print(config.samples)  # 50000

    # Or using get method
    config = VisionRegistry.get("cifar10")
    print(config.samples)  # 50000

    config = DatasetRegistry.get_dataset_info("cifar10")
    print(config.samples)  # 50000

    # Get all QA datasets
    qa_datasets = DatasetRegistry.get_datasets_by_task(TaskType.QUESTION_ANSWERING)

    # Get mixture info
    mixture = DatasetRegistry.get_mixture_info(["cifar10", "squad"])
    print(mixture)
    print(mixture)
    print()

    nlps = DatasetRegistry.get_language_registry()
    print(nlps.keys())
