from typing import Tuple

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import MLP, save_model


def make_dataset(n_samples: int = 20000, low: float = -100.0, high: float = 100.0) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(0)
    a = rng.uniform(low, high, size=(n_samples, 1)).astype(np.float32)
    b = rng.uniform(low, high, size=(n_samples, 1)).astype(np.float32)
    x = np.concatenate([a, b], axis=1)
    y = (a + b).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)

def train(epochs: int = 200, batch_size: int = 256, lr: float = 1e-3, save_path: str = "model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = make_dataset()
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MLP()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        running = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        epoch_loss = running / len(ds)
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.6f}")

    save_model(model, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    import config
    train(epochs=config.num_epochs, batch_size=config.batch_size, lr=config.lr, save_path="model.pt")
