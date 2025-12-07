"""
Enhanced training script with:
- Learning rate scheduling
- Early stopping
- Gradient clipping
- Comprehensive metrics (accuracy, precision, recall, F1)
- Model checkpointing
- Progress tracking
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm
from configs import config
import csv
from src.model import ANN
import ast
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def compute_metrics(predictions, labels, num_classes):
    """
    Compute comprehensive metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Compute precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_np, average='weighted', zero_division=0
    )
    
    # Compute per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels_np, preds_np, average=None, zero_division=0
    )
    
    # Accuracy
    accuracy = (preds_np == labels_np).mean()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support.tolist()
    }


def train_model(model, x, y):
    """
    Train the model with enhanced features.
    
    Args:
        model: Neural network model
        x: Input data
        y: Labels
        
    Returns:
        Trained model and training history
    """
    PAD_IDX = config.PAD_IDX
    
    # Split data
    split_idx = int(config.TRAIN_SPLIT * len(x))
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Device: {config.DEVICE}")
    print(f"Train samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Scheduler: {config.SCHEDULER_TYPE if config.USE_SCHEDULER else 'None'}")
    print(f"Early stopping: {config.EARLY_STOPPING}")
    print(f"Gradient clipping: {config.GRADIENT_CLIP}")
    print(f"{'='*60}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = None
    if config.USE_SCHEDULER:
        if config.SCHEDULER_TYPE == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
        elif config.SCHEDULER_TYPE == "step":
            scheduler = StepLR(
                optimizer, 
                step_size=config.SCHEDULER_STEP_SIZE,
                gamma=config.SCHEDULER_FACTOR
            )
        elif config.SCHEDULER_TYPE == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.SCHEDULER_FACTOR,
                patience=config.SCHEDULER_PATIENCE,
                verbose=True
            )
    
    # Early stopping
    early_stopping = None
    if config.EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            verbose=True
        )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(
            range(0, len(x_train), config.BATCH_SIZE),
            desc=f"Epoch {epoch+1}/{config.EPOCHS}"
        )
        
        for i in pbar:
            batch_x = x_train[i:i+config.BATCH_SIZE].clone()
            batch_y = y_train[i:i+config.BATCH_SIZE]
            
            # Replace PAD with zeros (for character encoding compatibility)
            if not config.USE_BPE_TOKENIZER:
                batch_x[batch_x == PAD_IDX] = 0
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if config.GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.GRADIENT_CLIP_VALUE
                )
            
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct / ((i + len(batch_y))):.4f}'
            })
        
        # Calculate training metrics
        train_acc = correct / len(x_train)
        avg_loss = running_loss / num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_x = x_val.clone()
            
            # Replace PAD with zeros (for character encoding compatibility)
            if not config.USE_BPE_TOKENIZER:
                val_x[val_x == PAD_IDX] = 0
            
            val_out = model(val_x)
            val_loss = criterion(val_out, y_val).item()
            val_preds = torch.argmax(val_out, dim=1)
            
            # Compute comprehensive metrics
            val_metrics = compute_metrics(val_preds, y_val, config.OUTPUT_DIM)
        
        # Update history
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1}/{config.EPOCHS} Summary:")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  ✔ Best model saved! (Val Loss: {val_loss:.4f})")
        
        print(f"{'─'*60}\n")
        
        # Learning rate scheduler step
        if scheduler is not None:
            if config.SCHEDULER_TYPE == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        if early_stopping is not None:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    print(f"\n📥 Loading best model from {config.BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH))
    
    return model, history


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Emotion Classification Model Training")
    print("="*60)
    
    # Load data
    print("\n📥 Loading preprocessed data...")
    sentences = []
    labels = []
    
    with open("data/processed_emotions.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in tqdm(reader, desc="Loading data"):
            label = int(row[0])
            sentence = ast.literal_eval(row[1])
            
            labels.append(label)
            sentences.append(sentence)
    
    # Convert to tensors
    x_data = torch.tensor(sentences, dtype=torch.float32)
    y_data = torch.tensor(labels, dtype=torch.long)
    
    print(f"\n✔ Data loaded successfully!")
    print(f"  Input shape: {x_data.shape}")
    print(f"  Labels shape: {y_data.shape}")
    print(f"  Number of classes: {config.OUTPUT_DIM}")
    
    # Initialize model
    print(f"\n🤖 Initializing model...")
    model = ANN(
        input_dim=x_data.size(1),
        num_classes=config.OUTPUT_DIM
    ).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Train model
    trained_model, history = train_model(
        model,
        x_data.to(config.DEVICE),
        y_data.to(config.DEVICE)
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"\n✔ Final model saved to {config.MODEL_SAVE_PATH}")
    
    # Save training history
    with open(config.METRICS_SAVE_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✔ Training metrics saved to {config.METRICS_SAVE_PATH}")
    
    # Print final summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation Loss: {min(history['val_loss']):.4f}")
    print(f"Best Validation Accuracy: {max(history['val_acc']):.4f}")
    print(f"Best Validation F1: {max(history['val_f1']):.4f}")
    print("="*60 + "\n")