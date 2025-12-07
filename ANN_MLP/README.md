# ANN/MLP Emotion Classification

A PyTorch-based Artificial Neural Network for emotion classification with BPE tokenization and advanced training features.

## Features

- 🤖 **BPE Tokenizer**: Efficient Byte Pair Encoding tokenizer for text processing
- 📊 **Advanced Training**: Learning rate scheduling, early stopping, gradient clipping
- 📈 **Comprehensive Metrics**: Accuracy, precision, recall, F1-score tracking
- 💾 **Model Checkpointing**: Automatically saves best performing model
- 🎯 **Confidence Scores**: Get prediction confidence with all class probabilities
- 📉 **Visualization**: Plot training metrics and model performance

## Quick Start

### 1. Install Dependencies

```bash
pip install torch datasets pandas tqdm scikit-learn matplotlib
```

### 2. Prepare Dataset

```bash
python datasetmaker.py
```

This will:
- Download the emotion dataset
- Train a BPE tokenizer
- Preprocess and tokenize the data
- Save processed data and statistics

### 3. Train Model

```bash
python train.py
```

Training features:
- Cosine annealing learning rate scheduler
- Early stopping (patience=7)
- Gradient clipping
- Comprehensive metrics tracking
- Automatic best model saving

### 4. Run Inference

```bash
python inference.py
```

Interactive mode allows you to:
- Enter text to classify emotions
- View prediction confidence
- See probability distribution across all emotions

## Configuration

Edit `config.py` to customize:

### Tokenizer Settings
```python
USE_BPE_TOKENIZER = True  # Set to False for character-level encoding
BPE_VOCAB_SIZE = 5000
MAX_LEN = 128
```

### Training Hyperparameters
```python
BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 50
WEIGHT_DECAY = 1e-5

# Scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"  # Options: "cosine", "step", "plateau"

# Early Stopping
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 7

# Gradient Clipping
GRADIENT_CLIP = True
GRADIENT_CLIP_VALUE = 1.0
```

### Model Architecture
```python
HIDDEN_DIM = 128
DROPOUT = 0.3
```

## Project Structure

```
ANN_MLP/
├── config.py              # Configuration settings
├── tokenizer.py           # BPE tokenizer implementation
├── model.py              # ANN model architecture (DO NOT MODIFY)
├── datasetmaker.py       # Dataset preprocessing
├── train.py              # Training script
├── inference.py          # Inference script
├── model_analysis.py     # Model analysis utility
├── visualize_metrics.py  # Metrics visualization
└── README.md            # This file
```

## Usage Examples

### Analyze Model

```bash
python model_analysis.py
```

Shows:
- Model architecture
- Parameter count and size
- Training metrics summary
- Dataset statistics

### Visualize Training Metrics

```bash
python visualize_metrics.py
```

Creates plots for:
- Training/validation loss
- Training/validation accuracy
- Precision, recall, F1-score
- Learning rate schedule

### Programmatic Inference

```python
from inference import EmotionClassifier

# Initialize classifier
classifier = EmotionClassifier()

# Single prediction
emotion, confidence, all_probs = classifier.predict("I am so happy today!")
print(f"Emotion: {emotion}, Confidence: {confidence:.2%}")

# Batch prediction
texts = ["I love this!", "This is terrible", "Just okay"]
results = classifier.predict_batch(texts)
```

## Advanced Features

### Custom Tokenizer

The BPE tokenizer can be used independently:

```python
from tokenizer import BPETokenizer

# Train tokenizer
tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.train(texts)
tokenizer.save("my_tokenizer.json")

# Load and use
tokenizer = BPETokenizer.load("my_tokenizer.json")
encoded = tokenizer.encode("Hello world")
decoded = tokenizer.decode(encoded)
```

### Switch to Character-Level Encoding

In `config.py`:
```python
USE_BPE_TOKENIZER = False
```

Then re-run:
```bash
python datasetmaker.py
python train.py
```

## Output Files

After training, you'll have:

- `emotion_model.pt` - Final model weights
- `emotion_model_best.pt` - Best model weights (lowest validation loss)
- `training_metrics.json` - Complete training history
- `bpe_tokenizer.json` - Trained tokenizer
- `dataset_stats.json` - Dataset statistics
- `mappings.json` - Emotion label mappings
- `training_metrics.png` - Metrics visualization

## Performance Tips

1. **GPU Acceleration**: Automatically uses CUDA if available
2. **Batch Size**: Increase if you have more GPU memory
3. **Learning Rate**: Use scheduler for better convergence
4. **Early Stopping**: Prevents overfitting
5. **BPE Tokenizer**: More efficient than character-level encoding

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` in config.py
- Reduce `MAX_LEN` if using very long sequences

### Poor Performance
- Increase `EPOCHS`
- Adjust `LEARNING_RATE`
- Increase `HIDDEN_DIM` for more model capacity
- Check dataset balance with `model_analysis.py`

### Tokenizer Issues
- Ensure `datasetmaker.py` completed successfully
- Check that `bpe_tokenizer.json` exists
- Verify `USE_BPE_TOKENIZER` setting matches your setup

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: [Rizqi/emotion-raw](https://huggingface.co/datasets/Rizqi/emotion-raw)
- Framework: PyTorch
