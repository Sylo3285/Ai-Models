# Encoder-Decoder Transformer for Conversational AI

A PyTorch implementation of an encoder-decoder transformer model for building conversational AI chatbots. This project includes complete training, inference, and analysis tools with support for custom datasets.

## 🌟 Features

- **Complete Transformer Implementation** using PyTorch's `nn.Transformer`
- **Advanced Training Features**:
  - Early stopping to prevent overfitting
  - Learning rate scheduling (ReduceLROnPlateau)
  - Gradient accumulation for larger effective batch sizes
  - Real-time progress tracking with tqdm
- **Multiple Inference Methods**:
  - Greedy decoding (fast)
  - Beam search (better quality)
- **Comprehensive Tools**:
  - Dataset downloader for HuggingFace datasets
  - Training analyzer with visualization
  - Interactive chat interface
- **Production Ready**: Checkpointing, vocabulary management, and model analysis

## 📁 Project Structure

```
encoder-decoder/
├── encoder_decoder.py      # Core transformer model architecture
├── train_csv.py           # Training script with CSV data support
├── inference_csv.py       # Inference with method comparison
├── chat.py               # Simple interactive chat interface
├── analyzer.py           # Model analysis and visualization
├── downloader.py         # Dataset downloader from HuggingFace
├── vocab.json            # Generated vocabulary
└── checkpoints/          # Saved model checkpoints
    ├── best_model.pt
    └── checkpoint_epoch_*.pt
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install torch pandas tqdm datasets matplotlib
```

### 2. Download Dataset

```bash
# Download DailyDialog dataset from HuggingFace
python downloader.py
```

This will download ~90k conversational pairs to `datasets/dailydialogue.csv`.

### 3. Train the Model

```bash
# Train on the downloaded dataset
python train_csv.py
```

**Training features:**
- Automatic vocabulary building
- Early stopping (patience: 10 epochs)
- Learning rate scheduling
- Progress bars with real-time metrics
- Saves best model automatically

**Expected training time:**
- GPU: ~2-3 hours for 30-40 epochs
- CPU: ~10-15 hours

### 4. Chat with Your Model

```bash
# Start interactive chat
python chat.py
```

Example conversation:
```
You: Hi
neura: Hi there. I'm neura—calm as a still pond, until the chaos decides to splash.

You: What's up?
neura: Not much—just balancing serenity on the tip of a black hole's event horizon.
```

### 5. Analyze Training Results

```bash
# Analyze model performance
python analyzer.py
```

This will:
- Display training metrics and perplexity
- Compare all checkpoints
- Generate training history plots
- Provide quality assessment and recommendations

## 📊 Model Architecture

### Default Configuration

```python
VOCAB_SIZE = ~23,000      # Built from training data
D_MODEL = 256             # Embedding dimension
NHEAD = 8                 # Attention heads
D_HID = 1024             # Feedforward dimension
NLAYERS = 4              # Encoder/decoder layers
DROPOUT = 0.1            # Dropout rate
BATCH_SIZE = 16          # Training batch size
```

**Model size:** ~19M parameters

### Key Components

1. **Positional Encoding**: Sinusoidal positional embeddings
2. **Multi-Head Attention**: 8 attention heads for parallel processing
3. **Feedforward Networks**: 1024-dimensional hidden layers
4. **Output Projection**: Weight-tied with decoder embeddings
5. **Gradient Clipping**: Max norm of 0.5 to prevent exploding gradients

## 🎯 Training Configuration

### Hyperparameters

```python
NUM_EPOCHS = 100                    # Max epochs (early stopping will stop earlier)
LEARNING_RATE = 0.0001             # Initial learning rate
EARLY_STOPPING_PATIENCE = 10       # Stop if no improvement for 10 epochs
GRADIENT_ACCUMULATION_STEPS = 2    # Effective batch size = 32
LR_SCHEDULER_PATIENCE = 3          # Reduce LR after 3 epochs without improvement
LR_SCHEDULER_FACTOR = 0.5          # Multiply LR by 0.5 when reducing
```

### Training Features

- **Early Stopping**: Automatically stops when validation loss plateaus
- **LR Scheduling**: Reduces learning rate when stuck
- **Gradient Accumulation**: Simulates larger batch sizes
- **Checkpointing**: Saves best model and periodic checkpoints
- **Progress Tracking**: Real-time loss, perplexity, learning rate, and gradient norms

## 📈 Performance Metrics

### Perplexity (PPL) Targets

- **Excellent**: PPL 10-30 (very natural responses)
- **Good**: PPL 30-60 (decent conversational ability)
- **Acceptable**: PPL 60-100 (usable but simple)
- **Poor**: PPL 100+ (needs improvement)

### Expected Results

With DailyDialog dataset (~90k samples):
- **Epoch 1**: PPL ~500 (starting)
- **Epoch 10**: PPL ~80-100 (learning)
- **Epoch 20**: PPL ~40-60 (improving)
- **Epoch 30+**: PPL ~25-35 (good quality)

## 🛠️ Advanced Usage

### Using Your Own Dataset

Create a CSV file with `input` and `output` columns:

```csv
input,output
Hi,Hello! How can I help you?
What's your name?,I'm neura, an AI assistant.
```

Then update `train_csv.py`:

```python
CSV_PATH = 'your_data.csv'
```

### Adjusting Model Size

For better quality (requires more memory):

```python
D_MODEL = 512      # Larger embeddings
NHEAD = 8          # Keep same
D_HID = 2048       # Larger feedforward
NLAYERS = 6        # More layers
```

For faster training (lower quality):

```python
D_MODEL = 128      # Smaller embeddings
NLAYERS = 3        # Fewer layers
D_HID = 512        # Smaller feedforward
```

### Inference Methods Comparison

```bash
# Compare greedy vs beam search with timing
python inference_csv.py
```

Results show:
- Greedy: ~0.05s per response (fastest)
- Beam Search (width=3): ~0.16s (3x slower, better quality)
- Beam Search (width=5): ~0.23s (4.4x slower, best quality)

## 📚 Scripts Overview

### Core Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `train_csv.py` | Train model on CSV data | `python train_csv.py` |
| `chat.py` | Interactive chatbot | `python chat.py` |
| `analyzer.py` | Analyze training results | `python analyzer.py` |
| `downloader.py` | Download datasets | `python downloader.py` |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `encoder_decoder.py` | Model architecture |
| `inference_csv.py` | Method comparison with timing |

## 🔍 Troubleshooting

### High Perplexity (PPL > 100)

**Solutions:**
- Train longer (increase `NUM_EPOCHS`)
- Use larger model (increase `D_MODEL`, `NLAYERS`)
- Check if dataset is too small or noisy
- Reduce learning rate

### Overfitting (Train PPL << Val PPL)

**Solutions:**
- Increase dropout: `DROPOUT = 0.2` or `0.3`
- Add weight decay: `optimizer = optim.Adam(..., weight_decay=1e-5)`
- Use more training data
- Reduce model size

### Out of Memory

**Solutions:**
- Reduce `BATCH_SIZE` (e.g., to 8)
- Reduce `D_MODEL` or `NLAYERS`
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Use CPU instead of GPU (slower but works)

### Repetitive Responses

**Solutions:**
- Use beam search instead of greedy decoding
- Increase beam width: `beam_width=7` or `10`
- Train with more diverse data
- Add temperature sampling (modify inference code)

## 📖 Dataset Information

### Included Datasets

**DailyDialog** (via `downloader.py`):
- 89,861 conversational pairs
- Everyday topics (work, school, entertainment)
- Natural, human-like dialogues
- Good for general chatbots

### Custom Dataset Format

Your CSV should have two columns:

```csv
input,output
"greeting or question","appropriate response"
```

**Tips for good datasets:**
- At least 10,000 pairs for decent quality
- Diverse topics and conversation styles
- Clean, grammatically correct text
- Consistent personality/tone

## 🎨 Customization

### Change Chatbot Personality

Edit your training data (`data.csv`) to reflect desired personality traits. For example, for a friendly, helpful assistant:

```csv
input,output
Hi,"Hello! I'm here to help. What can I do for you today?"
Thanks,"You're welcome! Happy to assist anytime!"
```

### Adjust Generation Parameters

In `chat.py`, modify:

```python
response = chatbot.generate(input_text, max_len=100)  # Max response length
```

For beam search (better quality):

```python
# Modify the generate method to use beam search
# See inference_csv.py for beam search implementation
```

## 📊 Model Analysis

The `analyzer.py` script provides:

1. **Checkpoint Comparison**: Compare all saved models
2. **Training Metrics**: Loss, perplexity, learning rate history
3. **Visualization**: Plots saved to `analysis/` folder
4. **Quality Assessment**: Automatic quality rating
5. **Recommendations**: Suggestions for improvement

**Generated plots:**
- Training and validation loss curves
- Perplexity progression (log scale)
- Learning rate schedule
- Overfitting indicator (train-val gap)

## 🤝 Contributing

Feel free to:
- Add new features
- Improve model architecture
- Add more inference methods
- Create better datasets
- Optimize training speed

## 📝 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- Built with PyTorch
- Uses HuggingFace datasets
- Inspired by "Attention is All You Need" (Vaswani et al., 2017)

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Run `python analyzer.py` to diagnose training issues
3. Review training logs and plots in `analysis/` folder

---

**Happy chatbot building! 🤖✨**
