import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

import wandb
from lmc.data.data_stats import TaskType
from lmc.experiment.base import ExperimentManager
from lmc.experiment_config import Finetuner, Trainer
from lmc.utils.setup_training import TrainingElements

from .train import TrainingRunner

os.environ["TOKENIZERS_PARALLELISM"] = "false"
@dataclass
class FinetuningRunner(TrainingRunner):
    config: Finetuner = field(init=True, default=Finetuner)
    _name: str = "finetuner"
    
    def description(self):
        return "Finetune a pre-trained model on a new dataset."
    
    def setup(self) -> None:
        # First call parent setup
        super().setup()
        self.tokenizer: PreTrainedTokenizer = self.training_elements[0].tokenizer
        self.max_epochs = self.config.data.max_seq_length
        self.use_language_model = self.config.data.is_language_dataset()
        
        if self.use_language_model:
            self._setup_language_model()
        
    def _setup_language_model(self):
        """Configure language model specific settings"""
        return 
        # Update data collation for language tasks
        for element in self.training_elements:
            if hasattr(element, 'train_loader'):
                element.train_loader.collate_fn = self._collate_language_batch
            if hasattr(element, 'val_loader'):
                element.val_loader.collate_fn = self._collate_language_batch
            if hasattr(element, 'test_loader'):
                element.test_loader.collate_fn = self._collate_language_batch

    def _collate_language_batch(self, batch):
        """Collate batch examples for language tasks"""
        print(batch)
        print(batch[0])
        if self.config.data.dataset == "snli":
            # SNLI specific handling
            premise = [example['premise'] for example in batch]
            hypothesis = [example['hypothesis'] for example in batch]
            labels = [example['label'] for example in batch]
            
            encoded = self.tokenizer(
                premise,
                text_pair=hypothesis,
                padding=True,
                truncation=True,
                max_length=self.config.data.max_seq_length,
                return_tensors="pt"
            )
            
            # Add labels
            encoded['labels'] = torch.tensor(labels)
            return encoded
            
        elif self.config.data.task_type == TaskType.NATURAL_LANGUAGE_INFERENCE:
            # General NLI handling
            text1 = [example.get('premise', example.get('sentence1')) for example in batch]
            text2 = [example.get('hypothesis', example.get('sentence2')) for example in batch]
            labels = [example['label'] for example in batch]
            
            encoded = self.tokenizer(
                text1,
                text_pair=text2,
                padding=True,
                truncation=True,
                max_length=self.config.data.max_seq_length,
                return_tensors="pt"
            )
            encoded['labels'] = torch.tensor(labels)
            return encoded
        # Handle sequence pair tasks (NLI, QA, etc)
        elif self.config.data.task_type in [TaskType.SEQUENCE_PAIR, TaskType.QUESTION_ANSWERING]:
            # Get text pairs based on task
            if self.config.data.task_type == TaskType.QUESTION_ANSWERING:
                text1 = [example['question'] for example in batch]
                text2 = [example['context'] for example in batch]
            else:
                text1 = [example['sentence1'] for example in batch]
                text2 = [example['sentence2'] for example in batch]
                
            return self.tokenizer(
                text1,
                text_pair=text2,
                padding=True,
                truncation=True,
                max_length=self.config.data.max_seq_length,
                return_tensors="pt"
            )
        
        # Handle classification and other single text tasks
        else:
            text = [example['text'] if 'text' in example else example['sentence'] for example in batch]
            return self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.config.data.max_seq_length,
                return_tensors="pt"
            )
    
    def on_epoch_end(self, ep: int, log_dct: Dict):
        # Add fine-tuning specific metrics
        for element in self.training_elements:
            if hasattr(element, 'model'):
                # Track gradients for non-frozen layers
                if self.config.logger.use_wandb:
                    grad_norm_dict = {}
                    
                    # Add language model specific metrics if applicable
                    if self.use_language_model:
                        # Calculate perplexity if loss is present
                        if 'loss' in log_dct:
                            perplexity = np.exp(log_dct['loss'])
                            grad_norm_dict['language/perplexity'] = perplexity
                        
                        # Add sequence length statistics
                        if hasattr(element, 'train_loader') and hasattr(element.train_loader, 'dataset'):
                            lengths = []
                            for batch in element.train_loader:
                                if isinstance(batch, dict) and 'input_ids' in batch:
                                    lengths.extend(batch['input_ids'].ne(self.tokenizer.pad_token_id).sum(1).tolist())
                            
                            if lengths:
                                grad_norm_dict.update({
                                    'language/seq_length/mean': np.mean(lengths),
                                    'language/seq_length/max': np.max(lengths),
                                    'language/seq_length/min': np.min(lengths)
                                })
                    
                    log_dct.update(grad_norm_dict)
        
        # Call parent's on_epoch_end
        super().on_epoch_end(ep, log_dct)