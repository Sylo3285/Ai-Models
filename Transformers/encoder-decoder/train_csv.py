import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import math
import time
import os
from collections import Counter
from tqdm import tqdm
from encoder_decoder import TransformerModel, generate_square_subsequent_mask


class Vocabulary:
    """Vocabulary class for text tokenization."""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.BOS_TOKEN = '<BOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        
        # Initialize with special tokens
        self.add_word(self.PAD_TOKEN)
        self.add_word(self.BOS_TOKEN)
        self.add_word(self.EOS_TOKEN)
        self.add_word(self.UNK_TOKEN)
    
    def add_word(self, word):
        """Add a word to the vocabulary."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1
    
    def add_sentence(self, sentence):
        """Add all words in a sentence to the vocabulary."""
        for word in sentence.split():
            self.add_word(word.lower())
    
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, sentence, add_special_tokens=True):
        """Convert sentence to token indices."""
        tokens = [self.word2idx.get(word.lower(), self.word2idx[self.UNK_TOKEN]) 
                 for word in sentence.split()]
        
        if add_special_tokens:
            tokens = [self.word2idx[self.BOS_TOKEN]] + tokens + [self.word2idx[self.EOS_TOKEN]]
        
        return tokens
    
    def decode(self, indices, skip_special_tokens=True):
        """Convert token indices back to sentence."""
        words = []
        special_tokens = {self.word2idx[self.PAD_TOKEN], 
                         self.word2idx[self.BOS_TOKEN], 
                         self.word2idx[self.EOS_TOKEN]}
        
        for idx in indices:
            if skip_special_tokens and idx in special_tokens:
                continue
            words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        
        return ' '.join(words)
    
    def save(self, filepath):
        """Save vocabulary to file."""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()},
            'word_count': dict(self.word_count)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f'Vocabulary saved to {filepath}')
    
    @classmethod
    def load(cls, filepath):
        """Load vocabulary from file."""
        vocab = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        vocab.word_count = Counter(vocab_data['word_count'])
        
        print(f'Vocabulary loaded from {filepath}')
        return vocab


def load_csv_data(csv_path, vocab=None, train_split=0.8):
    """
    Load and preprocess CSV data.
    
    Args:
        csv_path: Path to CSV file
        vocab: Existing vocabulary (if None, will build new one)
        train_split: Fraction of data to use for training
        
    Returns:
        train_src, train_tgt, val_src, val_tgt, vocab
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} samples from {csv_path}')
    
    # Build vocabulary if not provided
    if vocab is None:
        print('Building vocabulary...')
        vocab = Vocabulary()
        for _, row in df.iterrows():
            vocab.add_sentence(str(row['input']))
            vocab.add_sentence(str(row['output']))
        print(f'Vocabulary size: {len(vocab)}')
    
    # Tokenize data
    src_data = []
    tgt_data = []
    
    for _, row in df.iterrows():
        src_tokens = vocab.encode(str(row['input']), add_special_tokens=False)
        tgt_tokens = vocab.encode(str(row['output']), add_special_tokens=True)
        
        src_data.append(src_tokens)
        tgt_data.append(tgt_tokens)
    
    # Split into train and validation
    split_idx = int(len(src_data) * train_split)
    
    train_src = src_data[:split_idx]
    train_tgt = tgt_data[:split_idx]
    val_src = src_data[split_idx:]
    val_tgt = tgt_data[split_idx:]
    
    print(f'Training samples: {len(train_src)}')
    print(f'Validation samples: {len(val_src)}')
    
    return train_src, train_tgt, val_src, val_tgt, vocab


class Seq2SeqDataset(Dataset):
    """Dataset for sequence-to-sequence tasks."""
    
    def __init__(self, src_data, tgt_data):
        assert len(src_data) == len(tgt_data)
        self.src_data = src_data
        self.tgt_data = tgt_data
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx], dtype=torch.long), \
               torch.tensor(self.tgt_data[idx], dtype=torch.long)


def collate_fn(batch):
    """Collate function to pad sequences in a batch."""
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=0)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=0)
    
    # Create padding masks
    src_padding_mask = (src_batch == 0).transpose(0, 1)
    tgt_padding_mask = (tgt_batch == 0).transpose(0, 1)
    
    return src_batch, tgt_batch, src_padding_mask, tgt_padding_mask


class Trainer:
    """Trainer class for encoder-decoder transformer model."""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device, checkpoint_dir='checkpoints', scheduler=None, 
                 early_stopping_patience=10, gradient_accumulation_steps=1):
        """
        Args:
            model: TransformerModel instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on (cuda/cpu)
            checkpoint_dir: Directory to save checkpoints
            scheduler: Learning rate scheduler (optional)
            early_stopping_patience: Number of epochs to wait before early stopping
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float('inf')
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.should_stop = False
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        
        # Create progress bar
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                   desc=f'Epoch {epoch:3d} [Train]', 
                   bar_format='{l_bar}{bar:20}{r_bar}')
        
        for batch_idx, (src, tgt, src_padding_mask, tgt_padding_mask) in pbar:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            src_padding_mask = src_padding_mask.to(self.device)
            tgt_padding_mask = tgt_padding_mask.to(self.device)
            
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]
            tgt_padding_mask_input = tgt_padding_mask[:, :-1]
            
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(self.device)
            
            # Forward pass
            output = self.model(src, tgt_input, 
                              tgt_mask=tgt_mask,
                              src_padding_mask=src_padding_mask,
                              tgt_padding_mask=tgt_padding_mask_input,
                              memory_key_padding_mask=src_padding_mask)
            
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            loss = self.criterion(output, tgt_output)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                total_grad_norm += grad_norm.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar with detailed metrics
            avg_loss = total_loss / (batch_idx + 1)
            avg_grad_norm = total_grad_norm / max(1, (batch_idx + 1) // self.gradient_accumulation_steps)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'ppl': f'{math.exp(avg_loss):.1f}',
                'lr': f'{current_lr:.1e}',
                'grad': f'{avg_grad_norm:.2f}',
                'bs': src.size(1) * self.gradient_accumulation_steps
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        # Create progress bar
        pbar = tqdm(self.val_loader, desc='Validating', 
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}',
                   leave=False)
        
        with torch.no_grad():
            for src, tgt, src_padding_mask, tgt_padding_mask in pbar:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                src_padding_mask = src_padding_mask.to(self.device)
                tgt_padding_mask = tgt_padding_mask.to(self.device)
                
                tgt_input = tgt[:-1, :]
                tgt_output = tgt[1:, :]
                tgt_padding_mask_input = tgt_padding_mask[:, :-1]
                
                tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(self.device)
                
                output = self.model(src, tgt_input,
                                  tgt_mask=tgt_mask,
                                  src_padding_mask=src_padding_mask,
                                  tgt_padding_mask=tgt_padding_mask_input,
                                  memory_key_padding_mask=src_padding_mask)
                
                output = output.reshape(-1, output.shape[-1])
                tgt_output = tgt_output.reshape(-1)
                loss = self.criterion(output, tgt_output)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, epoch, vocab, filename='checkpoint.pt'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'vocab_size': len(vocab),
            'early_stopping_counter': self.early_stopping_counter
        }
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f'💾 Checkpoint saved to {filepath}')
    
    def train(self, num_epochs, vocab, save_every=5):
        """Train the model for multiple epochs."""
        print(f'\nStarting training for {num_epochs} epochs...')
        print(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        print(f'Early stopping patience: {self.early_stopping_patience} epochs')
        print(f'Gradient accumulation steps: {self.gradient_accumulation_steps}')
        print('=' * 89)
        
        for epoch in range(1, num_epochs + 1):
            if self.should_stop:
                print(f'\n⚠️  Early stopping triggered after {epoch - 1} epochs')
                break
            
            epoch_start_time = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            # Step scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            elapsed = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f'Epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'train loss {train_loss:5.2f} | train ppl {math.exp(train_loss):8.2f} | '
                  f'val loss {val_loss:5.2f} | val ppl {math.exp(val_loss):8.2f}')
            
            # Save periodic checkpoints
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, vocab, f'checkpoint_epoch_{epoch}.pt')
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, vocab, 'best_model.pt')
                print(f'✓ New best model saved (val loss: {val_loss:.4f})')
            else:
                self.early_stopping_counter += 1
                print(f'⏳ No improvement for {self.early_stopping_counter}/{self.early_stopping_patience} epochs')
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    self.should_stop = True
            
            print('─' * 89)
        
        print('\n✅ Training completed!')
        print(f'Best validation loss: {self.best_val_loss:.4f}')
        print(f'Total epochs trained: {len(self.train_losses)}')


if __name__ == '__main__':
    # Configuration
    CSV_PATH = 'datasets/dailydialogue.csv'
    VOCAB_PATH = 'vocab.json'
    D_MODEL = 256
    NHEAD = 8
    D_HID = 1024
    NLAYERS = 4
    DROPOUT = 0.1
    BATCH_SIZE = 16
    NUM_EPOCHS = 100  # Can train longer with early stopping
    LEARNING_RATE = 0.0001
    
    # Training enhancements
    EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16 * 2 = 32
    LR_SCHEDULER_PATIENCE = 3  # Reduce LR if no improvement for 3 epochs
    LR_SCHEDULER_FACTOR = 0.5  # Multiply LR by 0.5 when reducing
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print()
    
    # Load and preprocess data
    train_src, train_tgt, val_src, val_tgt, vocab = load_csv_data(CSV_PATH, train_split=0.9)
    
    # Save vocabulary
    vocab.save(VOCAB_PATH)
    print()
    
    # Create datasets and dataloaders
    train_dataset = Seq2SeqDataset(train_src, train_tgt)
    val_dataset = Seq2SeqDataset(val_src, val_tgt)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, collate_fn=collate_fn)
    
    # Create model
    print('Creating model...')
    VOCAB_SIZE = len(vocab)
    model = TransformerModel(
        ntoken=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        d_hid=D_HID,
        nlayers=NLAYERS,
        dropout=DROPOUT
    )
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=1e-6
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    )
    
    # Train
    trainer.train(num_epochs=NUM_EPOCHS, vocab=vocab, save_every=10)
