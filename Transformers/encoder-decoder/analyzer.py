"""
Model Analyzer - Extract and visualize training metrics from checkpoints
Analyzes trained models and displays training history, performance metrics, and comparisons.
"""

import torch
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def find_checkpoints(checkpoint_dir='checkpoints'):
    """Find all checkpoint files in the directory."""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt'):
            filepath = os.path.join(checkpoint_dir, file)
            checkpoints.append(filepath)
    
    return sorted(checkpoints)


def load_checkpoint_info(checkpoint_path):
    """Load checkpoint and extract information."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'path': checkpoint_path,
            'filename': os.path.basename(checkpoint_path),
            'epoch': checkpoint.get('epoch', 'N/A'),
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', []),
            'learning_rates': checkpoint.get('learning_rates', []),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'vocab_size': checkpoint.get('vocab_size', 'N/A'),
            'early_stopping_counter': checkpoint.get('early_stopping_counter', 0),
            'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
        }
        
        # Calculate perplexities
        if info['train_losses']:
            import math
            info['train_ppl'] = [math.exp(loss) for loss in info['train_losses']]
            info['val_ppl'] = [math.exp(loss) for loss in info['val_losses']]
            info['final_train_ppl'] = math.exp(info['train_losses'][-1])
            info['final_val_ppl'] = math.exp(info['val_losses'][-1])
            info['best_val_ppl'] = math.exp(info['best_val_loss'])
        
        return info
    except Exception as e:
        print(f'Error loading {checkpoint_path}: {e}')
        return None


def print_checkpoint_summary(info):
    """Print detailed summary of a checkpoint."""
    print('=' * 100)
    print(f'CHECKPOINT: {info["filename"]}')
    print('=' * 100)
    print(f'Epoch:                  {info["epoch"]}')
    print(f'Vocabulary Size:        {info["vocab_size"]:,}')
    print(f'File Size:              {info["file_size_mb"]:.2f} MB')
    print()
    
    if info['train_losses']:
        print('TRAINING METRICS:')
        print('-' * 100)
        print(f'Final Train Loss:       {info["train_losses"][-1]:.4f}  (PPL: {info["final_train_ppl"]:.2f})')
        print(f'Final Val Loss:         {info["val_losses"][-1]:.4f}  (PPL: {info["final_val_ppl"]:.2f})')
        print(f'Best Val Loss:          {info["best_val_loss"]:.4f}  (PPL: {info["best_val_ppl"]:.2f})')
        print(f'Total Epochs Trained:   {len(info["train_losses"])}')
        
        if info['learning_rates']:
            print(f'Final Learning Rate:    {info["learning_rates"][-1]:.2e}')
            print(f'Initial Learning Rate:  {info["learning_rates"][0]:.2e}')
        
        print(f'Early Stop Counter:     {info["early_stopping_counter"]}')
        
        # Calculate improvement
        if len(info['val_losses']) > 1:
            initial_val_loss = info['val_losses'][0]
            final_val_loss = info['val_losses'][-1]
            improvement = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100
            print(f'Val Loss Improvement:   {improvement:.2f}%')
    
    print()


def compare_checkpoints(checkpoints_info):
    """Compare multiple checkpoints."""
    if len(checkpoints_info) < 2:
        return
    
    print('=' * 100)
    print('CHECKPOINT COMPARISON')
    print('=' * 100)
    print()
    
    # Sort by best validation loss
    sorted_checkpoints = sorted(checkpoints_info, key=lambda x: x['best_val_loss'])
    
    print(f'{"Checkpoint":<30} {"Epoch":<8} {"Val Loss":<12} {"Val PPL":<12} {"Size (MB)":<12}')
    print('-' * 100)
    
    for info in sorted_checkpoints:
        if info['train_losses']:
            print(f'{info["filename"]:<30} {info["epoch"]:<8} '
                  f'{info["best_val_loss"]:<12.4f} {info["best_val_ppl"]:<12.2f} '
                  f'{info["file_size_mb"]:<12.2f}')
    
    print()
    print(f'🏆 Best Model: {sorted_checkpoints[0]["filename"]} '
          f'(Val Loss: {sorted_checkpoints[0]["best_val_loss"]:.4f}, '
          f'PPL: {sorted_checkpoints[0]["best_val_ppl"]:.2f})')
    print()


def plot_training_history(info, output_dir='analysis'):
    """Plot training history."""
    if not info['train_losses']:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = list(range(1, len(info['train_losses']) + 1))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History - {info["filename"]}', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, info['train_losses'], label='Train Loss', marker='o', markersize=3)
    axes[0, 0].plot(epochs, info['val_losses'], label='Val Loss', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Perplexity
    axes[0, 1].plot(epochs, info['train_ppl'], label='Train PPL', marker='o', markersize=3)
    axes[0, 1].plot(epochs, info['val_ppl'], label='Val PPL', marker='s', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_title('Training and Validation Perplexity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Learning Rate
    if info['learning_rates']:
        axes[1, 0].plot(epochs, info['learning_rates'], marker='o', markersize=3, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # Plot 4: Loss Difference (Overfitting indicator)
    loss_diff = [val - train for train, val in zip(info['train_losses'], info['val_losses'])]
    axes[1, 1].plot(epochs, loss_diff, marker='o', markersize=3, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val Loss - Train Loss')
    axes[1, 1].set_title('Overfitting Indicator (lower is better)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{Path(info["filename"]).stem}_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'📊 Training history plot saved to: {output_path}')


def analyze_best_model(checkpoint_dir='checkpoints'):
    """Analyze the best model."""
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    if not os.path.exists(best_model_path):
        print(f'❌ Best model not found at {best_model_path}')
        return None
    
    info = load_checkpoint_info(best_model_path)
    if info:
        print_checkpoint_summary(info)
        plot_training_history(info)
    
    return info


def main():
    print('=' * 100)
    print('                           MODEL ANALYZER')
    print('=' * 100)
    print()
    
    # Find all checkpoints
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print('❌ No checkpoints found in checkpoints/ directory')
        print('   Please train a model first using: python train_csv.py')
        return
    
    print(f'Found {len(checkpoints)} checkpoint(s)')
    print()
    
    # Load all checkpoint info
    checkpoints_info = []
    for checkpoint_path in checkpoints:
        info = load_checkpoint_info(checkpoint_path)
        if info:
            checkpoints_info.append(info)
    
    # Analyze best model
    print('ANALYZING BEST MODEL:')
    print()
    best_info = analyze_best_model()
    
    # Print all checkpoints
    if len(checkpoints_info) > 1:
        print()
        print('ALL CHECKPOINTS:')
        print()
        for info in checkpoints_info:
            if info['filename'] != 'best_model.pt':
                print_checkpoint_summary(info)
    
    # Compare checkpoints
    if len(checkpoints_info) > 1:
        compare_checkpoints(checkpoints_info)
    
    # Summary
    print('=' * 100)
    print('ANALYSIS COMPLETE')
    print('=' * 100)
    print()
    
    if best_info and best_info['train_losses']:
        print('📈 TRAINING SUMMARY:')
        print(f'   Total Epochs: {len(best_info["train_losses"])}')
        print(f'   Best Val PPL: {best_info["best_val_ppl"]:.2f}')
        print(f'   Final Val PPL: {best_info["final_val_ppl"]:.2f}')
        print()
        
        # Quality assessment
        final_ppl = best_info['final_val_ppl']
        if final_ppl < 30:
            quality = '🌟 Excellent'
        elif final_ppl < 60:
            quality = '✅ Good'
        elif final_ppl < 100:
            quality = '⚠️  Acceptable'
        else:
            quality = '❌ Needs Improvement'
        
        print(f'   Model Quality: {quality}')
        print()
        
        # Recommendations
        print('💡 RECOMMENDATIONS:')
        if final_ppl > 60:
            print('   • Consider training longer or using a larger model')
            print('   • Check if early stopping triggered too early')
        elif final_ppl < 20:
            print('   • Excellent performance! Model is ready for deployment')
        else:
            print('   • Good performance! Model should work well for most conversations')
        
        # Check for overfitting
        if best_info['train_losses'] and best_info['val_losses']:
            train_val_gap = best_info['val_losses'][-1] - best_info['train_losses'][-1]
            if train_val_gap > 0.5:
                print('   • ⚠️  Possible overfitting detected (train-val gap is large)')
                print('   • Consider adding more dropout or reducing model size')
        
        print()


if __name__ == '__main__':
    main()
