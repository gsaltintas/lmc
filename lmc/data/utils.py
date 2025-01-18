from enum import Enum


class TaskType(Enum):
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    QUESTION_ANSWERING = "question_answering"
    SEQUENCE_PAIR = "sequence_pair"
    NATURAL_LANGUAGE_INFERENCE = "natural_language_inference"
    SEQUENCE_LABELING = "sequence_labeling"
