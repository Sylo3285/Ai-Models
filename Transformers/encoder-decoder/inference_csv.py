import torch
import torch.nn.functional as F
import json
from encoder_decoder import TransformerModel, generate_square_subsequent_mask


class Vocabulary:
    """Vocabulary class for text tokenization."""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        
        # Special tokens
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
        
        print(f'Vocabulary loaded from {filepath}')
        print(f'Vocabulary size: {len(vocab.word2idx)}')
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


class ChatbotInference:
    """Inference class for chatbot using encoder-decoder transformer."""
    
    def __init__(self, model, vocab, device):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.device = device
        self.bos_token = vocab.word2idx[vocab.BOS_TOKEN]
        self.eos_token = vocab.word2idx[vocab.EOS_TOKEN]
        self.pad_token = vocab.word2idx[vocab.PAD_TOKEN]
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, vocab_path, d_model=256, nhead=8, 
                       d_hid=1024, nlayers=4, dropout=0.1, device=None):
        """Load model and vocabulary from checkpoint."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        vocab = Vocabulary.load(vocab_path)
        
        # Create model
        model = TransformerModel(
            ntoken=len(vocab),
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            dropout=dropout
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f'Model loaded from {checkpoint_path}')
        print(f'Checkpoint epoch: {checkpoint["epoch"]}')
        print(f'Best validation loss: {checkpoint["best_val_loss"]:.4f}')
        
        return cls(model, vocab, device)
    
    def generate_response(self, input_text, max_len=100, method='greedy', beam_width=5):
        """
        Generate response for input text.
        
        Args:
            input_text: Input message
            max_len: Maximum length of generated response
            method: 'greedy' or 'beam'
            beam_width: Beam width for beam search
            
        Returns:
            Generated response text
        """
        # Encode input
        src_tokens = self.vocab.encode(input_text, add_special_tokens=False)
        src = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(1).to(self.device)
        
        # Generate
        if method == 'greedy':
            generated_tokens = self._greedy_decode(src, max_len)
        elif method == 'beam':
            generated_tokens = self._beam_search(src, beam_width, max_len)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Decode to text
        response = self.vocab.decode(generated_tokens, skip_special_tokens=True)
        return response
    
    def _greedy_decode(self, src, max_len):
        """Greedy decoding."""
        self.model.eval()
        
        with torch.no_grad():
            # Encode source
            memory = self.model.encode(src, None)
            
            # Initialize target with BOS token
            tgt = torch.tensor([[self.bos_token]], dtype=torch.long, device=self.device)
            
            generated = [self.bos_token]
            
            for _ in range(max_len):
                tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(self.device)
                output = self.model.decode(tgt, memory, tgt_mask)
                output = self.model.output_projection(output[-1, :, :])
                prob = F.softmax(output, dim=-1)
                next_token = prob.argmax(dim=-1).item()
                
                generated.append(next_token)
                
                if next_token == self.eos_token:
                    break
                
                tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long, device=self.device)], dim=0)
            
            return generated
    
    def _beam_search(self, src, beam_width, max_len):
        """Beam search decoding."""
        self.model.eval()
        
        with torch.no_grad():
            memory = self.model.encode(src, None)
            beams = [([self.bos_token], 0.0)]
            
            for _ in range(max_len):
                candidates = []
                
                for seq, score in beams:
                    if seq[-1] == self.eos_token:
                        candidates.append((seq, score))
                        continue
                    
                    tgt = torch.tensor([seq], dtype=torch.long, device=self.device).transpose(0, 1)
                    tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(self.device)
                    output = self.model.decode(tgt, memory, tgt_mask)
                    output = self.model.output_projection(output[-1, :, :])
                    log_prob = F.log_softmax(output, dim=-1)
                    
                    topk_log_probs, topk_indices = log_prob.topk(beam_width)
                    
                    for k in range(beam_width):
                        token = topk_indices[0, k].item()
                        token_log_prob = topk_log_probs[0, k].item()
                        new_seq = seq + [token]
                        new_score = score + token_log_prob
                        candidates.append((new_seq, new_score))
                
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                
                if all(seq[-1] == self.eos_token for seq, _ in beams):
                    break
            
            best_seq, _ = beams[0]
            return best_seq
    
    def chat(self, method='greedy', beam_width=5):
        """Interactive chat mode."""
        print('=' * 80)
        print('Chatbot ready! Type "quit" or "exit" to end the conversation.')
        print('=' * 80)
        print()
        
        while True:
            try:
                user_input = input('You: ').strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print('neura: Goodbye! Until we meet again in the digital dreamscape.')
                    break
                
                response = self.generate_response(user_input, method=method, beam_width=beam_width)
                print(f'neura: {response}')
                print()
                
            except KeyboardInterrupt:
                print('\n\nneura: Caught in the interrupt! Farewell, wanderer.')
                break
            except Exception as e:
                print(f'Error: {e}')
                continue


if __name__ == '__main__':
    import time
    
    # Configuration
    CHECKPOINT_PATH = 'checkpoints/best_model.pt'
    VOCAB_PATH = 'vocab.json'
    D_MODEL = 256
    NHEAD = 8
    D_HID = 1024
    NLAYERS = 4
    BEAM_WIDTHS = [3, 5, 7]  # Different beam widths to test
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print()
    
    # Load model
    print('Loading model...')
    chatbot = ChatbotInference.from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        vocab_path=VOCAB_PATH,
        d_model=D_MODEL,
        nhead=NHEAD,
        d_hid=D_HID,
        nlayers=NLAYERS,
        device=device
    )
    print()
    
    # Test inputs
    test_inputs = [
        "Hi",
        "Hello",
        "Good morning",
        "What's up",
        "Hey neura",
        "How's it going"
    ]
    
    print('=' * 100)
    print('COMPARING DECODING METHODS WITH TIMING')
    print('=' * 100)
    print()
    
    for input_text in test_inputs:
        print('─' * 100)
        print(f'INPUT: "{input_text}"')
        print('─' * 100)
        
        # Greedy Decoding
        start_time = time.time()
        greedy_response = chatbot.generate_response(input_text, method='greedy')
        greedy_time = time.time() - start_time
        
        print(f'\n[GREEDY DECODING]')
        print(f'Response: {greedy_response}')
        print(f'Time: {greedy_time:.4f} seconds')
        
        # Beam Search with different widths
        for beam_width in BEAM_WIDTHS:
            start_time = time.time()
            beam_response = chatbot.generate_response(input_text, method='beam', beam_width=beam_width)
            beam_time = time.time() - start_time
            
            print(f'\n[BEAM SEARCH - Width {beam_width}]')
            print(f'Response: {beam_response}')
            print(f'Time: {beam_time:.4f} seconds')
            print(f'Speedup vs Greedy: {beam_time/greedy_time:.2f}x slower' if beam_time > greedy_time else f'{greedy_time/beam_time:.2f}x faster')
        
        print()
    
    # Summary statistics
    print('=' * 100)
    print('TIMING SUMMARY')
    print('=' * 100)
    print()
    
    print('Running comprehensive timing test...')
    
    # Warm-up
    for _ in range(3):
        chatbot.generate_response("Hi", method='greedy')
    
    # Greedy timing
    greedy_times = []
    for input_text in test_inputs:
        start = time.time()
        chatbot.generate_response(input_text, method='greedy')
        greedy_times.append(time.time() - start)
    
    # Beam search timing for each width
    beam_times = {width: [] for width in BEAM_WIDTHS}
    for width in BEAM_WIDTHS:
        for input_text in test_inputs:
            start = time.time()
            chatbot.generate_response(input_text, method='beam', beam_width=width)
            beam_times[width].append(time.time() - start)
    
    # Print summary
    avg_greedy = sum(greedy_times) / len(greedy_times)
    print(f'\nGreedy Decoding:')
    print(f'  Average time: {avg_greedy:.4f} seconds')
    print(f'  Min time: {min(greedy_times):.4f} seconds')
    print(f'  Max time: {max(greedy_times):.4f} seconds')
    
    for width in BEAM_WIDTHS:
        avg_beam = sum(beam_times[width]) / len(beam_times[width])
        print(f'\nBeam Search (width={width}):')
        print(f'  Average time: {avg_beam:.4f} seconds')
        print(f'  Min time: {min(beam_times[width]):.4f} seconds')
        print(f'  Max time: {max(beam_times[width]):.4f} seconds')
        print(f'  Slowdown vs Greedy: {avg_beam/avg_greedy:.2f}x')
    
    print()
    print('=' * 100)
    print('INFERENCE COMPARISON COMPLETE')
    print('=' * 100)
