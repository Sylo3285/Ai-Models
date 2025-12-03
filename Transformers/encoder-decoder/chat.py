"""
Simple Interactive Chat Script for neura AI VTuber
Just run this script to chat with your trained model!
"""

import torch
import torch.nn.functional as F
import json
import sys
from encoder_decoder import TransformerModel, generate_square_subsequent_mask
from config import Config


class Vocabulary:
    """Vocabulary class for text tokenization."""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.PAD_TOKEN = '<PAD>'
        self.BOS_TOKEN = '<BOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
    
    @classmethod
    def load(cls, filepath):
        """Load vocabulary from file."""
        vocab = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        return vocab
    
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, sentence, add_special_tokens=False):
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


class Chatbot:
    """Simple chatbot interface."""
    
    def __init__(self, model, vocab, device):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.device = device
        self.bos_token = vocab.word2idx[vocab.BOS_TOKEN]
        self.eos_token = vocab.word2idx[vocab.EOS_TOKEN]
    
    def generate(self, input_text, max_len=100):
        """Generate response for input text."""
        # Encode input
        src_tokens = self.vocab.encode(input_text, add_special_tokens=False)
        # Create tensor in batch_first format: [batch_size, seq_len]
        src = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode source
            memory = self.model.encode(src, None)
            
            # Initialize target with BOS token in batch_first format: [batch_size, seq_len]
            tgt = torch.tensor([[self.bos_token]], dtype=torch.long, device=self.device)
            generated = [self.bos_token]
            
            # Generate tokens one by one
            for _ in range(max_len):
                # tgt_mask size should match tgt sequence length (dimension 1 when batch_first=True)
                tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(self.device)
                output = self.model.decode(tgt, memory, tgt_mask)
                # Get the last token's output: [batch_size, vocab_size]
                output = self.model.output_projection(output[:, -1, :])
                prob = F.softmax(output, dim=-1)
                next_token = prob.argmax(dim=-1).item()
                
                generated.append(next_token)
                
                if next_token == self.eos_token:
                    break
                
                # Concatenate along sequence dimension (dim=1 for batch_first)
                tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long, device=self.device)], dim=1)
            
            # Decode to text
            response = self.vocab.decode(generated, skip_special_tokens=True)
            return response


def main():
    print('=' * 80)
    print('                    neura AI VTuber - Interactive Chat')
    print('=' * 80)
    print()
    
    # Check if files exist
    import os
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f'❌ Error: Checkpoint not found at {Config.BEST_MODEL_PATH}')
        print('   Please train the model first using: python train_csv.py')
        sys.exit(1)
    
    if not os.path.exists(Config.VOCAB_PATH):
        print(f'❌ Error: Vocabulary not found at {Config.VOCAB_PATH}')
        print('   Please train the model first using: python train_csv.py')
        sys.exit(1)
    
    # Device
    print(f'🔧 Loading model on {Config.DEVICE}...')
    
    # Load vocabulary
    vocab = Vocabulary.load(Config.VOCAB_PATH)
    print(f'📚 Vocabulary loaded: {len(vocab)} tokens')
    
    # Create model
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        d_hid=Config.D_HID,
        nlayers=Config.NLAYERS,
        dropout=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f'✅ Model loaded from checkpoint (epoch {checkpoint["epoch"]})')
    print(f'📊 Best validation loss: {checkpoint["best_val_loss"]:.4f}')
    print()
    
    # Create chatbot
    chatbot = Chatbot(model, vocab, Config.DEVICE)
    
    print('=' * 80)
    print('💬 Chat started! Type your message and press Enter.')
    print('   Commands: "quit", "exit", "bye" to end the conversation')
    print('=' * 80)
    print()
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input('You: ').strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print()
                print('neura: Goodbye! Until we meet again in the digital dreamscape. ✨')
                print()
                break
            
            # Generate response
            response = chatbot.generate(user_input)
            print(f'neura: {response}')
            print()
            
        except KeyboardInterrupt:
            print('\n')
            print('neura: Caught in the interrupt! Farewell, wanderer. 👋')
            print()
            break
        except Exception as e:
            print(f'❌ Error: {e}')
            continue


if __name__ == '__main__':
    main()
