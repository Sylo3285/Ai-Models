import torch
import os

class Config:
    # Data Paths
    DATASETS_DIR = 'datasets'
    CSV_PATH = os.path.join(DATASETS_DIR, 'dailydialogue.csv')
    VOCAB_PATH = 'vocab.json'
    CHECKPOINT_DIR = 'checkpoints'
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    ANALYSIS_DIR = 'analysis'

    # Model Hyperparameters
    D_MODEL = 128
    NHEAD = 4
    D_HID = 512
    NLAYERS = 2
    DROPOUT = 0.1

    # Training Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    GRADIENT_ACCUMULATION_STEPS = 2
    
    # Scheduler
    LR_SCHEDULER_PATIENCE = 3
    LR_SCHEDULER_FACTOR = 0.5

    # Inference
    BEAM_WIDTHS = [1, 3, 5]
    MAX_RESPONSE_LEN = 100

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def ensure_dirs(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.DATASETS_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.ANALYSIS_DIR, exist_ok=True)
