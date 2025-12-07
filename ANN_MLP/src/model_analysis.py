"""
Comprehensive model analysis script.

Provides detailed information about the model architecture, parameters,
and performance metrics.
"""

import torch
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import ANN
from configs import config


def analyze_model():
    """Analyze the trained model."""
    print("\n" + "="*60)
    print("Model Analysis")
    print("="*60)
    
    # Load model
    print("\n📥 Loading model...")
    model = ANN(
        input_dim=config.MAX_LEN,
        num_classes=config.OUTPUT_DIM
    )
    
    if os.path.exists(config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location='cpu'))
        print(f"✔ Loaded best model from {config.BEST_MODEL_PATH}")
    elif os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location='cpu'))
        print(f"✔ Loaded model from {config.MODEL_SAVE_PATH}")
    else:
        print("⚠️  No trained model found. Analyzing untrained model.")
    
    # Model architecture
    print("\n" + "─"*60)
    print("Model Architecture:")
    print("─"*60)
    print(model)
    
    # Parameter count
    print("\n" + "─"*60)
    print("Parameter Count:")
    print("─"*60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        print(f"  {name:30s} {num_params:>10,} {'(trainable)' if param.requires_grad else '(frozen)'}")
    
    print(f"\n  {'Total Parameters:':30s} {total_params:>10,}")
    print(f"  {'Trainable Parameters:':30s} {trainable_params:>10,}")
    print(f"  {'Non-trainable Parameters:':30s} {total_params - trainable_params:>10,}")
    
    # Model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    print(f"\n  {'Model Size:':30s} {size_mb:>10.2f} MB")
    
    # Layer details
    print("\n" + "─"*60)
    print("Layer Details:")
    print("─"*60)
    
    for i, layer in enumerate(model.net):
        print(f"\n  Layer {i}: {layer.__class__.__name__}")
        if hasattr(layer, 'in_features'):
            print(f"    Input features: {layer.in_features}")
            print(f"    Output features: {layer.out_features}")
        if hasattr(layer, 'p'):
            print(f"    Dropout probability: {layer.p}")
    
    # Training metrics (if available)
    if os.path.exists(config.METRICS_SAVE_PATH):
        print("\n" + "─"*60)
        print("Training Metrics:")
        print("─"*60)
        
        with open(config.METRICS_SAVE_PATH, 'r') as f:
            metrics = json.load(f)
        
        print(f"\n  Total epochs trained: {len(metrics['train_loss'])}")
        print(f"\n  Final Training Loss: {metrics['train_loss'][-1]:.4f}")
        print(f"  Final Training Accuracy: {metrics['train_acc'][-1]:.4f}")
        print(f"\n  Final Validation Loss: {metrics['val_loss'][-1]:.4f}")
        print(f"  Final Validation Accuracy: {metrics['val_acc'][-1]:.4f}")
        print(f"  Final Validation Precision: {metrics['val_precision'][-1]:.4f}")
        print(f"  Final Validation Recall: {metrics['val_recall'][-1]:.4f}")
        print(f"  Final Validation F1: {metrics['val_f1'][-1]:.4f}")
        
        print(f"\n  Best Validation Loss: {min(metrics['val_loss']):.4f}")
        print(f"  Best Validation Accuracy: {max(metrics['val_acc']):.4f}")
        print(f"  Best Validation F1: {max(metrics['val_f1']):.4f}")
    
    # Dataset statistics (if available)
    if os.path.exists("../outputs/dataset_stats.json"):
        print("\n" + "─"*60)
        print("Dataset Statistics:")
        print("─"*60)
        
        with open("../outputs/dataset_stats.json", 'r') as f:
            stats = json.load(f)
        
        print(f"\n  Total samples: {stats['num_samples']}")
        print(f"  Number of classes: {stats['num_classes']}")
        print(f"  Max sequence length: {stats['max_sequence_length']}")
        print(f"  Tokenizer type: {stats['tokenizer_type']}")
        
        if 'vocab_size' in stats:
            print(f"  Vocabulary size: {stats['vocab_size']}")
        
        if 'emotion_distribution' in stats:
            print(f"\n  Emotion distribution:")
            for emotion, count in stats['emotion_distribution'].items():
                percentage = (count / stats['num_samples']) * 100
                print(f"    {emotion:15s} {count:>6,} ({percentage:>5.2f}%)")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    analyze_model()