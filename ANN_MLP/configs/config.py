import torch
import csv
import os

# Number of unique output classes
# This will be determined from the dataset, but we provide a default
if os.path.exists("data/emotion_raw.csv"):
    with open("data/emotion_raw.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        emotions = set(row["Emotion"] for row in reader)
    OUTPUT_DIM = len(emotions)
elif os.path.exists("configs/mappings.json"):
    import json
    with open("configs/mappings.json", "r") as f:
        mappings = json.load(f)
    OUTPUT_DIM = len(mappings)
else:
    # Default value (will be overwritten after running datasetmaker.py)
    OUTPUT_DIM = 6  # Common emotion datasets have 6-7 classes

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# Training Hyperparameters
# ============================================
BATCH_SIZE = 256  # Reduced from 512 due to larger input size
LEARNING_RATE = 0.0005  # Reduced for more stable training with larger model
EPOCHS = 100  # Increased to allow more training time
WEIGHT_DECAY = 1e-4  # Increased regularization

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"  # Options: "cosine", "step", "plateau"
SCHEDULER_PATIENCE = 3  # For ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5  # For ReduceLROnPlateau and StepLR
SCHEDULER_STEP_SIZE = 10  # For StepLR

# Early stopping
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15  # Increased for larger model
EARLY_STOPPING_MIN_DELTA = 1e-4

# Gradient clipping
GRADIENT_CLIP = True
GRADIENT_CLIP_VALUE = 1.0

# ============================================
# Tokenizer Configuration
# ============================================
USE_BPE_TOKENIZER = True  # Set to False to use character-level encoding
BPE_VOCAB_SIZE = 5000
TOKENIZER_SAVE_PATH = "bpe_tokenizer.json"

# Maximum sequence length (will be determined from data if using BPE)
MAX_LEN = 320  # Increased to 320 to accommodate actual max length of 316
"""
MAX_LEN: Maximum sequence length for tokenized inputs.
- With BPE tokenizer: Set based on your dataset's max sequence length
- With character encoding: needs to match longest character sequence

CHANGING THIS VALUE REQUIRES RE-TRAINING THE MODEL.
Current dataset max: 316 tokens, so using 320 for safety.
"""

# ============================================
# Model Architecture
# ============================================
HIDDEN_DIM = 256  # Increased to 256 for better model capacity
DROPOUT = 0.4  # Increased dropout to prevent overfitting with larger model

# ============================================
# Data Configuration
# ============================================
PAD_IDX = 0
TRAIN_SPLIT = 0.9  # 90% train, 10% validation

# ============================================
# Paths
# ============================================
MODEL_SAVE_PATH = "models/emotion_model.pt"
BEST_MODEL_PATH = "models/emotion_model_best.pt"
METRICS_SAVE_PATH = "outputs/training_metrics.json"
TOKENIZER_SAVE_PATH = "configs/bpe_tokenizer.json"

