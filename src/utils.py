"""
Utility functions for the toxicity classifier.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_score

def load_dataset(file_path):
    """
    Load dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (texts, labels)
    """
    df = pd.read_csv(file_path)
    # Assuming the CSV has 'text' and 'label' columns
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def compute_metrics(pred):
    """
    Compute metrics for evaluation.
    
    Args:
        pred: Prediction object from Trainer
        
    Returns:
        dict: Dictionary of metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_scores(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def save_training_args(args, output_dir):
    """
    Save training arguments to a JSON file.
    
    Args:
        args: Training arguments
        output_dir (str): Directory to save the arguments
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

def load_training_args(model_dir):
    """
    Load training arguments from a JSON file.
    
    Args:
        model_dir (str): Directory containing the training arguments
        
    Returns:
        dict: Training arguments
    """
    with open(os.path.join(model_dir, 'training_args.json'), 'r') as f:
        return json.load(f)
