import torch
import json
import config
from model import ANN


# ---------------------------------------------------
# Load label ↔ id mappings
# ---------------------------------------------------
with open("mappings.json", "r") as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}


# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------
model = ANN(
    input_dim=config.MAX_LEN,     # MUST match training
    num_classes=config.OUTPUT_DIM
).to(config.DEVICE)

model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
model.eval()


# ---------------------------------------------------
# Pad function (same as preprocessing)
# ---------------------------------------------------
def pad_sentence(seq, max_len, pad_idx):
    if len(seq) < max_len:
        seq = seq + [pad_idx] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq


# ---------------------------------------------------
# Predict function
# ---------------------------------------------------
def predict(text):
    # Convert characters → ASCII integers
    seq = [ord(c) for c in text]

    # Pad to MAX_LEN
    seq = pad_sentence(seq, config.MAX_LEN, config.PAD_IDX)

    # Convert to tensor and replace PAD_IDX with zero (same as training)
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    x[x == config.PAD_IDX] = 0

    with torch.no_grad():
        logits = model(x)
        pred_id = torch.argmax(logits, dim=1).item()

    return inv_label_map[pred_id]


# ---------------------------------------------------
# Interactive test
# ---------------------------------------------------
if __name__ == "__main__":
    print("Emotion classifier ready. Type 'exit' to quit.\n")
    while True:
        text = input("Text: ")
        if text.lower() == "exit":
            break
        print("→ Prediction:", predict(text), "\n")
