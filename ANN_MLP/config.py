import torch
import csv

# Read all emotion labels
with open("emotion_raw.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    emotions = set(row["Emotion"] for row in reader)

# Number of unique output classes
OUTPUT_DIM = len(emotions)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 20

MAX_LEN = 1160 # Based on longest sentence in training data
"""
THIS IS CRUCIAL PART OF THE MODEL. IT IS BASED ON THE EMOTION DATASET USED. 
IF YOU USE A DIFFERENT DATASET, ADJUST THIS VALUE ACCORDINGLY.
FOR EXAMPLE, IF THE LONGEST SENTENCE IN YOUR DATASET IS 500 CHARACTERS,
SET MAX_LEN = 500

CHANGING THIS VALUE REQUIRES RE-TRAINING THE MODEL.
ELSE THE MODEL WILL BREAK
"""
MAX_VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
HIDDEN_DIM = 64

DROPOUT = 0.2
PAD_IDX = 0

MODEL_SAVE_PATH = "emotion_model.pt"

