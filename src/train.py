#!/usr/bin/env python
"""
Training script for the toxicity classifier.
"""
import os
import argparse
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .model import ToxicityClassifier
from .utils import load_dataset, compute_metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a toxicity classifier")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/train.csv",
        help="Path to training data"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/val.csv",
        help="Path to validation data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    return parser.parse_args()

class ToxicityDataset(Dataset):
    """Dataset for toxicity classification."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    
    def __len__(self):
        return len(self.labels)

def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    
    # Load datasets
    train_texts, train_labels = load_dataset(args.train_file)
    val_texts, val_labels = load_dataset(args.val_file)
    
    # Tokenize datasets
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    
    # Create PyTorch datasets
    train_dataset = ToxicityDataset(train_encodings, train_labels)
    val_dataset = ToxicityDataset(val_encodings, val_labels)
    
    # Load model config and update with custom parameters
    config = DistilBertConfig.from_pretrained(args.model_name)
    config.num_labels = 1
    config.seq_classif_dropout = 0.1
    
    model = ToxicityClassifier.from_pretrained(
        args.model_name,
        config=config
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=3,
        learning_rate=args.learning_rate,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training arguments
    training_args_dict = {k: v for k, v in training_args.to_dict().items() 
                         if not k.endswith('_path') and k != 'log_level'}
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(training_args_dict, f, indent=2)

if __name__ == "__main__":
    main()
