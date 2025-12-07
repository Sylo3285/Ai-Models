"""
Dataset preprocessing with BPE tokenizer support.

This script loads the emotion dataset, trains a BPE tokenizer (or uses character encoding),
and prepares the data for training.
"""

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs import config
from src.tokenizer import BPETokenizer


def pad_sequence(seq, max_len, pad_value=config.PAD_IDX):
    """
    Pad or truncate sequence to max_len.
    
    Args:
        seq: Input sequence
        max_len: Target length
        pad_value: Value to use for padding
        
    Returns:
        Padded/truncated sequence
    """
    if len(seq) < max_len:
        return seq + [pad_value] * (max_len - len(seq))
    else:
        return seq[:max_len]


def process_with_bpe():
    """Process dataset using BPE tokenizer."""
    print("=" * 60)
    print("Processing with BPE Tokenizer")
    print("=" * 60)
    
    # Load dataset
    print("\n📥 Loading dataset...")
    ds = load_dataset("Rizqi/emotion-raw")
    df = pd.DataFrame(ds['train'])
    
    # Save raw dataset copy
    df.to_csv("../data/emotion_raw.csv", index=False)
    print(f"✔ Loaded {len(df)} samples")
    
    # Build emotion mapping
    emotions_mapped = {}
    emotions = []
    texts = []
    
    print("\n🔨 Building emotion mappings...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        emotion = row["Emotion"]
        text = row["Text"]
        
        if emotion not in emotions_mapped:
            emotions_mapped[emotion] = len(emotions_mapped)
        
        emotions.append(emotions_mapped[emotion])
        texts.append(text)
    
    print(f"✔ Found {len(emotions_mapped)} emotion classes")
    
    # Train BPE tokenizer
    print(f"\n🤖 Training BPE tokenizer (vocab_size={config.BPE_VOCAB_SIZE})...")
    tokenizer = BPETokenizer(vocab_size=config.BPE_VOCAB_SIZE)
    tokenizer.train(texts, verbose=True)
    
    # Save tokenizer
    tokenizer.save(config.TOKENIZER_SAVE_PATH)
    
    # Encode all texts
    print("\n📝 Encoding texts...")
    encoded_texts = []
    max_len = 0
    
    for text in tqdm(texts, desc="Encoding"):
        encoded = tokenizer.encode(text)
        encoded_texts.append(encoded)
        max_len = max(max_len, len(encoded))
    
    print(f"✔ Max sequence length: {max_len}")
    
    # Update config if needed
    if max_len > config.MAX_LEN:
        print(f"⚠️  Warning: Max length ({max_len}) exceeds config.MAX_LEN ({config.MAX_LEN})")
        print(f"   Consider updating config.MAX_LEN to {max_len}")
    
    # Pad sequences
    print(f"\n📏 Padding sequences to length {config.MAX_LEN}...")
    padded_texts = [pad_sequence(seq, config.MAX_LEN) for seq in tqdm(encoded_texts, desc="Padding")]
    
    # Save processed data
    print("\n💾 Saving processed data...")
    df_processed = pd.DataFrame({
        "Emotion": emotions,
        "Text": padded_texts
    })
    df_processed.to_csv("../data/processed_emotions.csv", index=False)
    
    # Save mappings
    with open("../configs/mappings.json", "w") as f:
        json.dump(emotions_mapped, f, indent=4)
    
    # Save statistics
    stats = {
        "num_samples": len(df),
        "num_classes": len(emotions_mapped),
        "max_sequence_length": max_len,
        "vocab_size": tokenizer.vocab_size_actual,
        "tokenizer_type": "BPE",
        "emotion_distribution": {k: emotions.count(v) for k, v in emotions_mapped.items()}
    }
    
    with open("../outputs/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print("\n" + "=" * 60)
    print("✔ Processing Complete!")
    print("=" * 60)
    print(f"Total samples: {stats['num_samples']}")
    print(f"Emotion classes: {stats['num_classes']}")
    print(f"Max sequence length: {stats['max_sequence_length']}")
    print(f"Vocabulary size: {stats['vocab_size']}")
    print(f"\nFiles created:")
    print(f"  - processed_emotions.csv")
    print(f"  - mappings.json")
    print(f"  - {config.TOKENIZER_SAVE_PATH}")
    print(f"  - dataset_stats.json")
    print("=" * 60)


def process_with_char_encoding():
    """Process dataset using character-level encoding (original method)."""
    print("=" * 60)
    print("Processing with Character-Level Encoding")
    print("=" * 60)
    
    PAD_IDX = config.PAD_IDX
    
    # Load dataset
    print("\n📥 Loading dataset...")
    ds = load_dataset("Rizqi/emotion-raw")
    df = pd.DataFrame(ds['train'])
    
    # Save raw dataset copy
    df.to_csv("../data/emotion_raw.csv", index=False)
    print(f"✔ Loaded {len(df)} samples")
    
    emotions_mapped = {}
    emotions = []
    sentences = []
    
    # First pass: build list of sentences + emotions
    print("\n🔨 Processing texts...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        emotion = row["Emotion"]
        text = row["Text"]
        
        # Build emotion → id mapping
        if emotion not in emotions_mapped:
            emotions_mapped[emotion] = len(emotions_mapped)
        
        emotions.append(emotions_mapped[emotion])
        
        # Convert characters → ord() list
        sentence = [ord(c) for c in text]
        sentences.append(sentence)
    
    # Auto-padding
    max_len = max(len(s) for s in sentences)
    print(f"\n📏 Max sentence length: {max_len}")
    
    # Apply padding
    print(f"Padding sequences to length {max_len}...")
    padded_sentences = [pad_sequence(s, max_len) for s in tqdm(sentences, desc="Padding")]
    
    # Save processed data
    print("\n💾 Saving processed data...")
    df_processed = pd.DataFrame({
        "Emotion": emotions,
        "Text": padded_sentences
    })
    df_processed.to_csv("../data/processed_emotions.csv", index=False)
    
    with open("../configs/mappings.json", "w") as f:
        json.dump(emotions_mapped, f, indent=4)
    
    # Save statistics
    stats = {
        "num_samples": len(df),
        "num_classes": len(emotions_mapped),
        "max_sequence_length": max_len,
        "tokenizer_type": "character",
        "emotion_distribution": {k: emotions.count(v) for k, v in emotions_mapped.items()}
    }
    
    with open("../outputs/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print("\n" + "=" * 60)
    print("✔ Processing Complete!")
    print("=" * 60)
    print(f"Total samples: {stats['num_samples']}")
    print(f"Emotion classes: {stats['num_classes']}")
    print(f"Max sentence length: {stats['max_sequence_length']}")
    print("=" * 60)


if __name__ == "__main__":
    if config.USE_BPE_TOKENIZER:
        process_with_bpe()
    else:
        process_with_char_encoding()
