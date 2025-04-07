#### WIP
#
import logging
from typing import Any, Dict

from datasets import load_dataset
from transformers import PreTrainedTokenizer

from lmc.config import DataConfig
from lmc.data.data_stats import LanguageConfig

logging.basicConfig(level=logging.INFO)


def format_chat_template(prompt, row):
    row_json = None

    instruction = prompt["instruction"]
    usr_input = prompt["user_input"]
    assistant_output = prompt["assistant_output"]

    # when there are multiple user inputs
    if isinstance(usr_input, dict):
        s = ""
        for k in usr_input:
            if isinstance(usr_input[k], dict):
                s += "\n" + k + " - "
                data_key = usr_input[k]["key_name"]
                data_field = row[data_key]  # e.g. choices
                fields = usr_input[k][
                    "fields"
                ]  # e.g. subfield of choices; labels, description
                concat_symbol = usr_input[k][
                    "concat_symbol"
                ]  # how to concat the fields

                if concat_symbol:
                    s += "\n"
                    for pair in zip(data_field[fields[0]], data_field[fields[1]]):
                        s += pair[0] + " " + concat_symbol + " " + pair[1] + "\n"
                else:
                    s += data_field[fields[0]]
            else:
                s += k + " - " + str(row[usr_input[k]]) + " [SEP] \n"

        usr_query = s.strip().strip("\n")[: -len("[SEP]")].strip()
    else:
        usr_query = row[usr_input]

    row_json = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": usr_query},
        {"role": "assistant", "content": str(row[assistant_output])},
    ]

    return row_json


class MathDatasetLoader:
    """Handles loading and processing of individual math datasets."""

    def __init__(
        self,
        tokenizer,
        config,
        padding: bool = True,
        eval: bool = False,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.padding = padding
        self.eval = eval
        if eval:
            self.padding_side = "left"
        else:
            self.padding_side = "right"

    def process_example(self, example, add_generation_prompt: bool = False):
        """Process a single example"""
        return self._process_data(example, add_generation_prompt)

    def process_batch(self, examples, add_generation_prompt: bool = False):
        """Process a batch of examples"""
        # Initialize result dictionaries with empty lists
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        processed_inputs = []
        # Process each example in the batch
        n = examples
        for i in range(len(examples[next(iter(examples))])):
            # Extract a single example from the batch
            example = {k: examples[k][i] for k in examples}

            # Process the individual example
            processed = self._process_data(example, add_generation_prompt)
            processed_inputs.append(processed)
            continue
        tokenized = self.tokenizer(
            processed_inputs,
            truncation=True,
            padding="max_length",  # Always pad to max_length
            max_length=self.config.max_seq_length,
            return_tensors="pt",
            padding_side=self.padding_side,
        )

        # Add labels for causal language modeling (also padded)
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    def _process_data(self, example, add_generation_prompt: bool = False):
        """Internal method to process a single example"""

        if "gsm8k" in self.config.hf_path:
            instruction_format = self.config.instruction_format
            chat = format_chat_template(instruction_format, example)

            # Get formatted string first
            formatted_text = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_special_tokens=False,
                add_generation_prompt=add_generation_prompt,
            )
            return formatted_text
        # EleutherAI/hendrycks_math
        elif "hendrycks_math" in self.config.hf_path:
            prompt = f"Problem: {example['problem']}"
            target = example["solution"]

            # Tokenize combined input and target
            formatted_text = prompt + self.tokenizer.eos_token + target
            tokenized = self.tokenizer(
                formatted_text,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )

            # Add labels for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        elif "math_qa" in self.config.hf_path:
            prompt = f"Question: {example['Problem']}\nOptions: {example['options']}"
            target = f"{example['Rationale']}\nAnswer: {example['correct']}"

            # Tokenize combined input and target
            formatted_text = prompt + self.tokenizer.eos_token + target
            tokenized = self.tokenizer(
                formatted_text,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )

            # Add labels for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        else:
            raise ValueError(
                f"{self.config.hf_path} not yet supported in math datasets."
            )

    def load_dataset(self, split="train"):
        dataset = load_dataset(
            self.config.hf_path,
            self.config.hf_config,
            split=self.config.splits[split],
            trust_remote_code=self.config.trust_remote_code,
        )

        return dataset.map(
            self.process_example,
            remove_columns=dataset.column_names,
            load_from_cache_file=True,
        )


def get_math_preprocessor(
    tokenizer: PreTrainedTokenizer, data_conf: DataConfig, evaluate: bool
):
    def preprocess_gsm8k(examples: Dict[str, Any]) -> Dict[str, Any]:
        # Create separate messages for input only
        input_messages = [
            [
                {
                    "role": "user",
                    "content": f"Question: {q}\nProvide a step-by-step solution and end with #### followed by the final answer:",
                }
            ]
            for q in examples["question"]
        ]

        # Create full messages for labels (including both question and answer)
        label_messages = [
            [
                {
                    "role": "user",
                    "content": f"Question: {q}\nProvide a step-by-step solution and end with #### followed by the final answer:",
                },
                {"role": "assistant", "content": ans},
            ]
            for q, ans in zip(examples["question"], examples["answer"])
        ]
        # Format inputs (questions only)
        input_texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # Add the assistant prompt for generation
            )
            for messages in input_messages
        ]

        # Format full sequences for labels
        label_texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in label_messages
        ]

        # Tokenize inputs
        model_inputs = tokenizer(
            input_texts,
            add_special_tokens=False,
            padding="max_length",
            max_length=data_conf.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                label_texts,
                add_special_tokens=False,
                padding="max_length",
                max_length=data_conf.max_seq_length,
                truncation=True,
                return_tensors="pt",
            )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_math(examples: Dict[str, Any]) -> Dict[str, Any]:
        # Create input messages (just the problem)
        input_messages = [
            [{"role": "user", "content": f"Problem: {p}"}] for p in examples["problem"]
        ]

        # Create full messages for labels
        label_messages = [
            [
                {"role": "user", "content": f"Problem: {p}"},
                {"role": "assistant", "content": soln},
            ]
            for p, soln in zip(examples["problem"], examples["solution"])
        ]

        # Format inputs (problems only)
        input_texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in input_messages
        ]

        # Format full sequences for labels
        label_texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in label_messages
        ]

        # Tokenize inputs
        model_inputs = tokenizer(
            input_texts,
            add_special_tokens=False,
            padding="max_length",
            max_length=data_conf.max_seq_length,
            truncation=True,
        )

        # Tokenize labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                label_texts,
                add_special_tokens=False,
                padding="max_length",
                max_length=data_conf.max_seq_length,
                truncation=True,
            )

        # Clean up padding in labels
        labels_with_ignore_index = []
        for label_ids in labels["input_ids"]:
            labels_example = [
                label if label != tokenizer.pad_token_id else -100
                for label in label_ids
            ]
            labels_with_ignore_index.append(labels_example)
        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs

    def preprocess_asdiv(examples: Dict[str, Any]) -> Dict[str, Any]:
        # Create input messages
        input_messages = [
            [
                {
                    "role": "user",
                    "content": f"Question: {q}\nSolve this step by step. Show your work and give the final answer after ####.",
                }
            ]
            for q in examples["question"]
        ]

        # Create full messages for labels
        label_messages = [
            [
                {
                    "role": "user",
                    "content": f"Question: {q}\nSolve this step by step. Show your work and give the final answer after ####.",
                },
                {"role": "assistant", "content": f"{soln}\n#### {ans}"},
            ]
            for q, soln, ans in zip(
                examples["question"], examples["solution"], examples["answer"]
            )
        ]

        # Format inputs
        input_texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in input_messages
        ]

        # Format labels
        label_texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in label_messages
        ]

        # Tokenize inputs
        model_inputs = tokenizer(
            input_texts,
            add_special_tokens=False,
            padding="max_length",
            max_length=data_conf.max_seq_length,
            truncation=True,
        )

        # Tokenize labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                label_texts,
                add_special_tokens=False,
                padding="max_length",
                max_length=data_conf.max_seq_length,
                truncation=True,
            )

        # Clean up padding in labels
        labels_with_ignore_index = []
        for label_ids in labels["input_ids"]:
            labels_example = [
                label if label != tokenizer.pad_token_id else -100
                for label in label_ids
            ]
            labels_with_ignore_index.append(labels_example)
        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs

    def preprocess_mathqa(examples: Dict[str, Any]) -> Dict[str, Any]:
        # Create input messages
        input_messages = [
            [{"role": "user", "content": f"Question: {q}\nOptions: {opt}"}]
            for q, opt in zip(examples["Problem"], examples["options"])
        ]

        # Create full messages for labels
        label_messages = [
            [
                {"role": "user", "content": f"Question: {q}\nOptions: {opt}"},
                {"role": "assistant", "content": f"{r}\nAnswer: {a}"},
            ]
            for q, opt, r, a in zip(
                examples["Problem"],
                examples["options"],
                examples["Rationale"],
                examples["correct"],
            )
        ]

        # Format inputs
        input_texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in input_messages
        ]

        # Format labels
        label_texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in label_messages
        ]

        # Tokenize inputs
        model_inputs = tokenizer(
            input_texts,
            add_special_tokens=False,
            padding="max_length",
            max_length=data_conf.max_seq_length,
            truncation=True,
        )

        # Tokenize labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                label_texts,
                add_special_tokens=False,
                padding="max_length",
                max_length=data_conf.max_seq_length,
                truncation=True,
            )

        # Store original IDs for decoding
        model_inputs["reference_ids"] = labels["input_ids"]

        # Create masked labels for training
        labels_with_ignore_index = []
        for idx, (input_text, label_ids) in enumerate(
            zip(input_texts, labels["input_ids"])
        ):
            # Find where assistant response starts
            assistant_start = input_text.find("<|assistant|>")
            if assistant_start != -1:
                # Convert text position to token position
                prefix_tokens = len(
                    tokenizer(input_text[:assistant_start])["input_ids"]
                )
                # Create masked labels
                masked_labels = label_ids.copy()
                # Mask everything before assistant's response
                masked_labels[:prefix_tokens] = [-100] * prefix_tokens
                # Mask padding
                padding_mask = label_ids == tokenizer.pad_token_id
                masked_labels[padding_mask] = -100
                labels_with_ignore_index.append(masked_labels)
            else:
                labels_with_ignore_index.append([-100] * len(label_ids))

        model_inputs["labels"] = labels_with_ignore_index
        return model_inputs

        # Clean up padding in labels
        labels_with_ignore_index = []
        for label_ids in labels["input_ids"]:
            labels_example = [
                label if label != tokenizer.pad_token_id else -100
                for label in label_ids
            ]
            labels_with_ignore_index.append(labels_example)
        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs

    preprocessors = {
        "gsm8k": preprocess_gsm8k,
        "math": preprocess_math,
        "mathqa": preprocess_mathqa,
        "asdiv": preprocess_asdiv,
    }

    return preprocessors[data_conf.dataset]


import json
import re
from typing import Dict, List, Optional, Union

import evaluate

####

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def clean_model_output(out):
    out = out.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    out = out.split("####")[
        -1
    ]  # a few datasets with #### as the final answer designator
    out = out.strip("\n").split("\n")[
        0
    ]  # take the very first answer before the line change
    out = out.replace(
        "<|eot_id|>", ""
    ).lower()  # misc cleaning: remove end of sequence tok, remove space, lower case
    out = out.strip(".").strip()  # remove '.' in case of short answers

    return out


def clean_target_output(tar):
    tar = str(tar)
    tar = tar.split("####")[-1]
    tar = tar.strip(".").strip().lower()

    return tar


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class MathMetricsEvaluator:
    """
    Evaluator for mathematical reasoning datasets like GSM8K and MATH.
    Extracts final numerical answers and computes accuracy metrics.
    """

    def __init__(self, dataset_name: str = "gsm8k"):
        """
        Initialize the evaluator for a specific dataset.

        Args:
            dataset_name: Either "gsm8k" or "math"
        """
        self.dataset_name = dataset_name.lower()
        self.exact_match = evaluate.load("exact_match")

    def extract_answer(self, text: str) -> str:
        """
        Extract numerical answer from model output based on dataset-specific patterns.

        Args:
            text: The model's output text

        Returns:
            Extracted answer as a string
        """
        # Clean number format (remove commas in numbers)
        text = re.sub(r"(\d),(\d)", r"\1\2", text)

        # GSM8K typically has the final answer as the last number, often preceded by
        # phrases like "the answer is", "= ", etc.
        if self.dataset_name == "gsm8k":
            # Look for answer patterns like "the answer is X" or "= X"
            answer_pattern = re.search(
                r"#### (\-?[0-9\.\,]+)",
                text,
                re.IGNORECASE,
                # r"(?:answer|result|=)[^\d-]*?([-+]?\d*\.?\d+)", text, re.IGNORECASE
            )
            if answer_pattern:
                return answer_pattern.group(1)

            # If no explicit answer pattern, extract the last number
            numbers = re.findall(r"[-+]?\d*\.?\d+", text)
            return numbers[-1] if numbers else ""

        # MATH problems often require extracting specific answer formats (fractions, expressions, etc.)
        elif self.dataset_name == "math":
            # Look for answer patterns like "the answer is X" or "= X"
            ## TODO; check
            answer_pattern = re.search(
                r"(?:answer|result|=)[^=]*?([-+]?\d*\.?\d+/?[-+*/\d\^()]*)",
                text,
                re.IGNORECASE,
            )
            if answer_pattern:
                return answer_pattern.group(1).strip()

            # For MATH, try to extract expressions, fractions, etc.
            # Extract fraction patterns
            fraction_pattern = re.search(r"(\d+)/(\d+)", text)
            if fraction_pattern:
                return fraction_pattern.group(0)

            # Extract expressions with operations
            expression_pattern = re.search(
                r"([-+]?\d*\.?\d+\s*[-+*/]\s*[-+]?\d*\.?\d+)", text
            )
            if expression_pattern:
                return expression_pattern.group(0).replace(" ", "")

            # If no specific patterns match, extract the last number
            numbers = re.findall(r"[-+]?\d*\.?\d+", text)
            return numbers[-1] if numbers else ""

        # Default fallback to last number
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        return numbers[-1] if numbers else ""

    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answers to handle different formats for comparison.

        Args:
            answer: Raw answer string

        Returns:
            Normalized answer string
        """
        # Remove spaces
        answer = answer.replace(" ", "")

        # Handle fractions and convert to decimal if needed
        if "/" in answer and all(part.isdigit() for part in answer.split("/")):
            try:
                num, denom = map(int, answer.split("/"))
                # Keep fractions in their simplest form
                import math

                gcd = math.gcd(num, denom)
                return f"{num // gcd}/{denom // gcd}"
            except:
                pass

        return answer

    def compute_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for mathematical reasoning.

        Args:
            predictions: List of model output texts
            references: List of reference answer texts

        Returns:
            Dictionary with evaluation metrics
        """
        # Clean and extract answers from predictions
        cleaned_preds = [
            self.normalize_answer(self.extract_answer(pred)) for pred in predictions
        ]

        # Clean reference answers
        cleaned_refs = [self.normalize_answer(ref) for ref in references]

        # For debugging purposes, you can log the extracted answers
        # for i, (pred, ref) in enumerate(zip(cleaned_preds, cleaned_refs)):
        #     print(f"Example {i}: Prediction: {pred}, Reference: {ref}")

        # Compute exact match score
        metrics = self.exact_match.compute(
            predictions=cleaned_preds,
            references=cleaned_refs,
            ignore_case=True,
            ignore_punctuation=True,
        )

        # For GSM8K, we can also compute numerical equivalence
        # (e.g., "12" matches "12.0" or "12.00")
        if self.dataset_name == "gsm8k":
            try:
                numerical_matches = sum(
                    1
                    for p, r in zip(cleaned_preds, cleaned_refs)
                    if p and r and float(p) == float(r)
                )
                metrics["numerical_match"] = (
                    numerical_matches / len(predictions) if predictions else 0
                )
            except ValueError:
                # If conversion to float fails, skip numerical matching
                pass

        return metrics


# Example usage
if __name__ == "__main__":
    # GSM8K example
    gsm8k_evaluator = MathMetricsEvaluator(dataset_name="gsm8k")
    gsm8k_predictions = [
        "First, I calculate 5 * 10 = 50. Then I add 7 to get 57. The answer is 57.",
        "To solve this, I need to find 20% of 80. 20% = 0.2, so 0.2 * 80 = 16.",
    ]
    gsm8k_references = ["57", "16"]
    gsm8k_results = gsm8k_evaluator.compute_metrics(gsm8k_predictions, gsm8k_references)
    print(f"GSM8K metrics: {gsm8k_results}")

    # MATH example
    math_evaluator = MathMetricsEvaluator(dataset_name="math")
    math_predictions = [
        "The solution is x = 2/3.",
        "After solving the equation, I get y = 4 * π.",
    ]
    math_references = ["2/3", "4π"]
    math_results = math_evaluator.compute_metrics(math_predictions, math_references)
    print(f"MATH metrics: {math_results}")
