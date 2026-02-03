import torch
import random
from tqdm import tqdm

from model import load_model
from config import (
    device,
    SEQ_LEN,
    label_to_movement,
    movement_to_label,
)

# =========================
# Generate continuous test price series
# =========================
def generate_price_series(length=12000, start_price=75):
    prices = [start_price]

    for _ in tqdm(range(length - 1)):
        movement = random.choice(["up", "down", "same"])
        last = prices[-1]

        if movement == "up":
            prices.append(last + random.randint(1, 5))
        elif movement == "down":
            prices.append(last - random.randint(1, 5))
        else:
            prices.append(last)

    return prices


# =========================
# Build sliding-window test dataset
# =========================
def build_test_dataset(prices, seq_len):
    data = []
    targets = []

    for i in tqdm(range(len(prices) - seq_len - 1)):
        window = prices[i : i + seq_len]
        next_day = prices[i + seq_len]
        last_day = window[-1]

        if next_day > last_day:
            label = movement_to_label["up"]
        elif next_day < last_day:
            label = movement_to_label["down"]
        else:
            label = movement_to_label["same"]

        data.append([[p] for p in window])  # (SEQ_LEN, 1)
        targets.append(label)

    X = torch.tensor(data, dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.long, device=device)

    # normalize (MUST match training)
    #X /= 100.0

    return X, y


# =========================
# Evaluation
# =========================
def evaluate(model, X, y):
    model.eval()

    with torch.no_grad():
        logits, _ = model(X)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == y).sum().item()

    return correct / len(y), preds


# =========================
# Main
# =========================
if __name__ == "__main__":
    model_path = "rnn_model.pt"
    model = load_model(model_path)

    # Generate test data
    prices = generate_price_series()
    X_test, y_test = build_test_dataset(prices, SEQ_LEN)

    acc, preds = evaluate(model, X_test, y_test)

    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy: {acc * 100:.2f}%")

    # Prediction breakdown
    for label, name in label_to_movement.items():
        count = (preds == label).sum().item()
        print(f"{name}: {count}")
