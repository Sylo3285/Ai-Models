import torch
from tqdm import tqdm
from model import ANN
import config
import pandas as pd

def train_model(X_train, y_train, X_val, y_val, y_mean=None, y_std=None, patience=20):
    model = ANN().to(config.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    X_train = X_train.to(config.device)
    y_train = y_train.to(config.device)
    X_val = X_val.to(config.device)
    y_val = y_val.to(config.device)
    
    if y_mean is not None:
        y_mean = y_mean.to(config.device)
        y_std = y_std.to(config.device)

    best_val_loss = float("inf")
    best_model_state = None
    wait = 0  # tracks epochs since improvement

    for epoch in tqdm(range(config.num_epochs), desc="Training Progress"):
        model.train()

        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            
            # Un-normalize predictions for readable MAE
            if y_mean is not None and y_std is not None:
                val_pred_orig = val_pred * y_std + y_mean
                y_val_orig = y_val * y_std + y_mean
                val_mae = torch.mean(torch.abs(val_pred_orig - y_val_orig)).item()
            else:
                val_mae = torch.mean(torch.abs(val_pred - y_val)).item()

        print(f"Epoch [{epoch+1}/{config.num_epochs}] "
              f"Train Loss: {loss.item():.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.4f}")

        # -------------------------
        # Early Stopping
        # -------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n⛔ Early Stopping triggered at epoch {epoch+1}")
                break

    # Load best model weights before returning
    model.load_state_dict(best_model_state)
    return model


def split_data(x, y, train_ratio=0.8):
    dataset = torch.utils.data.TensorDataset(x, y)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # unpack back into tensors (since you said no DataLoader)
    x_train = train_dataset[:][0]
    y_train = train_dataset[:][1]

    x_val = val_dataset[:][0]
    y_val = val_dataset[:][1]

    return x_train, y_train, x_val, y_val



df = pd.read_csv('data.csv')

X = torch.tensor(df[['fuel','toll','time','labor']].values, dtype=torch.float32)
y = torch.tensor(df[['cost']].values, dtype=torch.float32)

X_train, y_train, X_val, y_val = split_data(X, y)

# Normalization
X_mean = X_train.mean(dim=0)
X_std = X_train.std(dim=0)
y_mean = y_train.mean(dim=0)
y_std = y_train.std(dim=0)

X_train = (X_train - X_mean) / (X_std + 1e-8)
y_train = (y_train - y_mean) / (y_std + 1e-8)
X_val = (X_val - X_mean) / (X_std + 1e-8)
y_val = (y_val - y_mean) / (y_std + 1e-8)

model = train_model(X_train, y_train, X_val, y_val, y_mean, y_std, patience=20)
torch.save(model.state_dict(), 'model.pt')
