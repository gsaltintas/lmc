import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

import wandb
from lmc.experiment.base import ExperimentManager
from lmc.experiment_config import Finetuner, Trainer
from lmc.utils.setup_training import TrainingElements

from .train import TrainingRunner


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
            
        # Update data collation for language tasks
        for element in self.training_elements:
            if hasattr(element, 'train_loader'):
                element.train_loader.collate_fn = self._collate_language_batch
            if hasattr(element, 'val_loader'):
                element.val_loader.collate_fn = self._collate_language_batch
            if hasattr(element, 'test_loader'):
                element.test_loader.collate_fn = self._collate_language_batch

    def _collate_language_batch(self, examples):
        """Custom collation for language tasks"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized for language tasks")
            
        # Handle different input formats
        if isinstance(examples[0], dict):
            # Assume dict with 'input_ids' and 'labels'
            input_texts = [ex.get('input_ids', '') for ex in examples]
            labels = [ex.get('labels', None) for ex in examples]
        elif isinstance(examples[0], tuple):
            # Assume (input, label) pairs
            input_texts, labels = zip(*examples)
        else:
            input_texts = examples
            labels = None
            
        # Tokenize inputs
        encoded = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_seq_length,
            return_tensors="pt",
            pad_to_multiple_of=self.config.data.pad_to_multiple_of
        )
        
        # Prepare batch
        batch = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
        }
        
        if 'token_type_ids' in encoded:
            batch['token_type_ids'] = encoded['token_type_ids']
            
        if labels is not None:
            if isinstance(labels[0], (int, float)):
                batch['labels'] = torch.tensor(labels)
            else:
                # Handle sequence labels (e.g., for generation tasks)
                label_encoding = self.tokenizer(
                    labels,
                    padding=True,
                    truncation=True,
                    max_length=self.config.data.max_seq_length,
                    return_tensors="pt",
                    pad_to_multiple_of=self.config.data.pad_to_multiple_of
                )
                batch['labels'] = label_encoding['input_ids']
                
        return batch
    
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