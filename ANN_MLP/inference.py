"""
Enhanced inference script with:
- BPE tokenizer support
- Confidence scores
- Batch inference
- Better error handling
"""

import torch
import json
import os
from configs import config
from src.model import ANN
from src.tokenizer import BPETokenizer


class EmotionClassifier:
    """Emotion classification inference wrapper."""
    
    def __init__(self, model_path=None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to model weights (default: config.BEST_MODEL_PATH)
        """
        if model_path is None:
            # Try to load best model first, fall back to regular model
            if os.path.exists(config.BEST_MODEL_PATH):
                model_path = config.BEST_MODEL_PATH
                print(f"📥 Loading best model from {model_path}")
            else:
                model_path = config.MODEL_SAVE_PATH
                print(f"📥 Loading model from {model_path}")
        
        # Load label mappings
        with open("configs/mappings.json", "r") as f:
            self.label_map = json.load(f)
        
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        
        # Load tokenizer if using BPE
        self.tokenizer = None
        if config.USE_BPE_TOKENIZER:
            if os.path.exists(config.TOKENIZER_SAVE_PATH):
                self.tokenizer = BPETokenizer.load(config.TOKENIZER_SAVE_PATH)
            else:
                raise FileNotFoundError(
                    f"BPE tokenizer not found at {config.TOKENIZER_SAVE_PATH}. "
                    "Please run datasetmaker.py first."
                )
        
        # Load model
        self.model = ANN(
            input_dim=config.MAX_LEN,
            num_classes=config.OUTPUT_DIM
        ).to(config.DEVICE)
        
        self.model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        self.model.eval()
        
        print(f"✔ Model loaded successfully!")
        print(f"  Device: {config.DEVICE}")
        print(f"  Tokenizer: {'BPE' if config.USE_BPE_TOKENIZER else 'Character-level'}")
        print(f"  Classes: {list(self.label_map.keys())}\n")
    
    def _pad_sequence(self, seq, max_len, pad_idx):
        """Pad or truncate sequence to max_len."""
        if len(seq) < max_len:
            seq = seq + [pad_idx] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        return seq
    
    def _preprocess_text(self, text):
        """
        Preprocess text for inference.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed tensor
        """
        if config.USE_BPE_TOKENIZER:
            # Use BPE tokenizer
            seq = self.tokenizer.encode(text)
        else:
            # Use character-level encoding
            seq = [ord(c) for c in text]
        
        # Pad to MAX_LEN
        seq = self._pad_sequence(seq, config.MAX_LEN, config.PAD_IDX)
        
        # Convert to tensor
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
        
        # Replace PAD_IDX with zero (for character encoding compatibility)
        if not config.USE_BPE_TOKENIZER:
            x[x == config.PAD_IDX] = 0
        
        return x
    
    def predict(self, text, return_confidence=True):
        """
        Predict emotion for a single text.
        
        Args:
            text: Input text string
            return_confidence: Whether to return confidence scores
            
        Returns:
            If return_confidence=True: (predicted_emotion, confidence, all_probabilities)
            If return_confidence=False: predicted_emotion
        """
        x = self._preprocess_text(text)
        
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            confidence, pred_id = torch.max(probabilities, dim=1)
            
            pred_id = pred_id.item()
            confidence = confidence.item()
        
        predicted_emotion = self.inv_label_map[pred_id]
        
        if return_confidence:
            # Get all class probabilities
            all_probs = {
                self.inv_label_map[i]: probabilities[0, i].item()
                for i in range(config.OUTPUT_DIM)
            }
            return predicted_emotion, confidence, all_probs
        else:
            return predicted_emotion
    
    def predict_batch(self, texts, return_confidence=True):
        """
        Predict emotions for multiple texts.
        
        Args:
            texts: List of text strings
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of predictions (format depends on return_confidence)
        """
        results = []
        for text in texts:
            result = self.predict(text, return_confidence=return_confidence)
            results.append(result)
        return results


def interactive_mode():
    """Run interactive inference mode."""
    print("\n" + "="*60)
    print("Emotion Classifier - Interactive Mode")
    print("="*60)
    
    classifier = EmotionClassifier()
    
    print("\nCommands:")
    print("  - Type text to classify emotion")
    print("  - Type 'exit' or 'quit' to exit")
    print("  - Type 'help' for more information")
    print("\n" + "="*60 + "\n")
    
    while True:
        text = input("📝 Enter text: ").strip()
        
        if not text:
            continue
        
        if text.lower() in ['exit', 'quit']:
            print("\n👋 Goodbye!\n")
            break
        
        if text.lower() == 'help':
            print("\nThis classifier predicts the emotion expressed in text.")
            print(f"Available emotions: {', '.join(classifier.label_map.keys())}\n")
            continue
        
        # Predict
        emotion, confidence, all_probs = classifier.predict(text)
        
        # Display results
        print(f"\n{'─'*60}")
        print(f"🎯 Prediction: {emotion.upper()}")
        print(f"📊 Confidence: {confidence*100:.2f}%")
        print(f"\n📈 All probabilities:")
        
        # Sort by probability
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for emotion_name, prob in sorted_probs:
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"  {emotion_name:15s} {bar} {prob*100:5.2f}%")
        
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    interactive_mode()
