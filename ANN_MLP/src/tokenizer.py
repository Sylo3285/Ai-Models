"""
BPE (Byte Pair Encoding) Tokenizer for text processing.

This tokenizer learns subword units from the training data and can encode/decode
text using the learned vocabulary.

Optimized version with faster merge operations.
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os


class BPETokenizer:
    """Byte Pair Encoding tokenizer implementation with optimizations."""
    
    def __init__(self, vocab_size: int = 5000, pad_token: str = "<PAD>", 
                 unk_token: str = "<UNK>", eos_token: str = "<EOS>",
                 use_multiprocessing: bool = False):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            pad_token: Padding token
            unk_token: Unknown token for out-of-vocabulary words
            eos_token: End of sequence token
            use_multiprocessing: Use multiprocessing for parallel operations (experimental)
        """
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.use_multiprocessing = use_multiprocessing
        
        # Special tokens
        self.special_tokens = [pad_token, unk_token, eos_token]
        
        # Vocabulary mappings
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        
        # Initialize with special tokens
        for i, token in enumerate(self.special_tokens):
            self.token2id[token] = i
            self.id2token[i] = token
    
    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """
        Count frequency of adjacent token pairs (optimized).
        
        Args:
            word_freqs: Dictionary mapping word tuples to their frequencies
            
        Returns:
            Counter of token pair frequencies
        """
        pairs = Counter()
        # Optimized: Use items() directly and avoid repeated lookups
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs
    
    def _merge_pair(self, pair: Tuple[str, str], word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """
        Merge the most frequent pair in the vocabulary (optimized).
        
        Args:
            pair: Tuple of two tokens to merge
            word_freqs: Current word frequencies
            
        Returns:
            Updated word frequencies with merged pair
        """
        new_word_freqs = {}
        replacement = ''.join(pair)
        pair_0, pair_1 = pair  # Unpack once for faster access
        
        for word, freq in word_freqs.items():
            # Skip words that are too short or don't contain the pair
            if len(word) < 2:
                new_word_freqs[word] = freq
                continue
            
            # Optimized merge using list comprehension where possible
            new_word = []
            i = 0
            word_len = len(word)
            
            while i < word_len:
                # Check if we can merge at this position
                if i < word_len - 1 and word[i] == pair_0 and word[i + 1] == pair_1:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def train(self, texts: List[str], verbose: bool = True):
        """
        Train the BPE tokenizer on a corpus of texts (optimized).
        
        Args:
            texts: List of text strings to train on
            verbose: Whether to show progress bar
        """
        # Tokenize into words and count frequencies
        word_freqs = Counter()
        
        if verbose:
            print("Building word frequencies...")
            texts_iter = tqdm(texts, desc="Processing texts")
        else:
            texts_iter = texts
        
        for text in texts_iter:
            # Simple whitespace tokenization
            words = text.lower().split()
            for word in words:
                # Split into characters with end-of-word marker
                word_freqs[tuple(word) + ('</w>',)] += 1
        
        if verbose:
            print(f"Found {len(word_freqs)} unique words")
        
        # Initialize vocabulary with all characters
        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word)
        
        # Add initial vocabulary
        for token in sorted(vocab):
            if token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token
        
        if verbose:
            print(f"Initial vocabulary size: {len(self.token2id)}")
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(self.token2id)
        
        if verbose:
            print(f"Performing up to {num_merges} merges...")
            pbar = tqdm(total=num_merges, desc="BPE merges")
        
        merges_done = 0
        for merge_idx in range(num_merges):
            pairs = self._get_stats(word_freqs)
            
            if not pairs:
                if verbose:
                    print(f"\nNo more pairs to merge. Stopped at {merges_done} merges.")
                break
            
            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)
            merges_done += 1
            
            # Add to vocabulary
            new_token = ''.join(best_pair)
            if new_token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[new_token] = idx
                self.id2token[idx] = new_token
            
            if verbose:
                pbar.update(1)
                # Update progress bar with current stats every 100 merges
                if (merge_idx + 1) % 100 == 0:
                    pbar.set_postfix({
                        'vocab': len(self.token2id),
                        'unique_words': len(word_freqs)
                    })
        
        if verbose:
            pbar.close()
            print(f"✔ Tokenizer trained! Vocabulary size: {len(self.token2id)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using learned BPE merges.
        
        Args:
            word: Word to tokenize
            
        Returns:
            List of subword tokens
        """
        word = tuple(word.lower()) + ('</w>',)
        
        # Apply merges
        for pair in self.merges:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(''.join(pair))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
        
        return list(word)
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        words = text.split()
        token_ids = []
        
        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                token_ids.append(self.token2id.get(token, self.token2id[self.unk_token]))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        tokens = [self.id2token.get(tid, self.unk_token) for tid in token_ids]
        
        # Remove special tokens
        tokens = [t for t in tokens if t not in self.special_tokens]
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        
        return text.strip()
    
    def save(self, filepath: str):
        """
        Save tokenizer to file.
        
        Args:
            filepath: Path to save tokenizer
        """
        data = {
            'vocab_size': self.vocab_size,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'eos_token': self.eos_token,
            'token2id': self.token2id,
            'id2token': {int(k): v for k, v in self.id2token.items()},
            'merges': self.merges
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✔ Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BPETokenizer':
        """
        Load tokenizer from file.
        
        Args:
            filepath: Path to load tokenizer from
            
        Returns:
            Loaded BPETokenizer instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data['vocab_size'],
            pad_token=data['pad_token'],
            unk_token=data['unk_token'],
            eos_token=data['eos_token']
        )
        
        tokenizer.token2id = data['token2id']
        tokenizer.id2token = {int(k): v for k, v in data['id2token'].items()}
        tokenizer.merges = [tuple(pair) for pair in data['merges']]
        
        print(f"✔ Tokenizer loaded from {filepath}")
        return tokenizer
    
    @property
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size."""
        return len(self.token2id)
    
    @property
    def pad_id(self) -> int:
        """Get padding token ID."""
        return self.token2id[self.pad_token]
    
    @property
    def unk_id(self) -> int:
        """Get unknown token ID."""
        return self.token2id[self.unk_token]
    
    @property
    def eos_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.token2id[self.eos_token]


if __name__ == "__main__":
    # Example usage
    texts = [
        "Hello world, this is a test.",
        "BPE tokenizer is awesome!",
        "Machine learning is fun."
    ]
    
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(texts)
    
    # Test encoding/decoding
    test_text = "Hello world"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
