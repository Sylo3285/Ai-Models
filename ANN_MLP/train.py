import torch
from tqdm import tqdm
import config
import csv
from model import ANN
import ast

best_val_loss = float('inf')
def train_model(model, x, y):

    PAD_IDX = config.PAD_IDX

    # 90/10 split
    split_idx = int(0.9 * len(x))
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()

    for epoch in range(config.EPOCHS):
        running_loss = 0.0
        correct = 0

        for i in tqdm(range(0, len(x_train), config.BATCH_SIZE), desc=f"Epoch {epoch+1}"):

            batch_x = x_train[i:i+config.BATCH_SIZE].clone()
            batch_y = y_train[i:i+config.BATCH_SIZE]

            # Replace PAD with zeros (best we can do in ANN)
            batch_x[batch_x == PAD_IDX] = 0

            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()

        train_acc = correct / len(x_train)
        avg_loss = running_loss / max(1, len(x_train) // config.BATCH_SIZE)

        # Validation
        model.eval()
        with torch.no_grad():

            val_x = x_val.clone()
            val_x[val_x == PAD_IDX] = 0

            val_out = model(val_x)
            val_loss = criterion(val_out, y_val).item()
            val_preds = torch.argmax(val_out, dim=1)
            val_acc = (val_preds == y_val).sum().item() / len(y_val)

        model.train()

        print(f"Epoch {epoch+1}/{config.EPOCHS} "
              f"| Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model


if __name__ == "__main__":
    sentences = []
    labels = []

    with open("processed_emotions.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in tqdm(reader, desc="Loading data"):

            label = int(row[0])

            # Convert the string list → Python list
            sentence = ast.literal_eval(row[1])

            labels.append(label)
            sentences.append(sentence)

    # Convert to tensors
    x_data = torch.tensor(sentences, dtype=torch.float32)
    y_data = torch.tensor(labels, dtype=torch.long)

    print("X shape:", x_data.shape)
    print("Y shape:", y_data.shape)

    model = ANN(
        input_dim=x_data.size(1),
        num_classes=config.OUTPUT_DIM
    ).to(config.DEVICE)

    trained_model = train_model(
        model,
        x_data.to(config.DEVICE),
        y_data.to(config.DEVICE)
    )
    torch.save(trained_model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"✔ Model saved to {config.MODEL_SAVE_PATH}")