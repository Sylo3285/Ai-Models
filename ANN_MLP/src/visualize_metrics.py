"""
Utility script to visualize training metrics.

Creates plots for loss, accuracy, and other metrics over training epochs.
"""

import json
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs import config


def plot_metrics():
    """Plot training metrics from saved JSON file."""
    if not os.path.exists(config.METRICS_SAVE_PATH):
        print(f"❌ Metrics file not found: {config.METRICS_SAVE_PATH}")
        print("Please train the model first.")
        return
    
    # Load metrics
    with open(config.METRICS_SAVE_PATH, 'r') as f:
        metrics = json.load(f)
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, metrics['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[0, 0].plot(epochs, metrics['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, metrics['train_acc'], label='Train Acc', marker='o', markersize=3)
    axes[0, 1].plot(epochs, metrics['val_acc'], label='Val Acc', marker='s', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Validation Metrics (Precision, Recall, F1)
    axes[1, 0].plot(epochs, metrics['val_precision'], label='Precision', marker='o', markersize=3)
    axes[1, 0].plot(epochs, metrics['val_recall'], label='Recall', marker='s', markersize=3)
    axes[1, 0].plot(epochs, metrics['val_f1'], label='F1 Score', marker='^', markersize=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    axes[1, 1].plot(epochs, metrics['learning_rates'], marker='o', markersize=3, color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = 'outputs/training_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✔ Metrics plot saved to {plot_path}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total epochs: {len(epochs)}")
    print(f"\nBest Validation Loss: {min(metrics['val_loss']):.4f} (Epoch {metrics['val_loss'].index(min(metrics['val_loss'])) + 1})")
    print(f"Best Validation Accuracy: {max(metrics['val_acc']):.4f} (Epoch {metrics['val_acc'].index(max(metrics['val_acc'])) + 1})")
    print(f"Best Validation F1: {max(metrics['val_f1']):.4f} (Epoch {metrics['val_f1'].index(max(metrics['val_f1'])) + 1})")
    print(f"\nFinal Training Loss: {metrics['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {metrics['val_loss'][-1]:.4f}")
    print(f"Final Validation Accuracy: {metrics['val_acc'][-1]:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    plot_metrics()
