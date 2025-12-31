import torch
import torch.nn as nn
import config


class MLP(nn.Module):
    def __init__(self, input_dim: int | None = None, hidden_sizes=None, output_dim: int | None = None):
        super().__init__()
        if input_dim is None:
            input_dim = config.input_size
        if output_dim is None:
            output_dim = config.output_size
        if hidden_sizes is None:
            # config.hidden_size may be a single int; use as single hidden layer
            hidden_sizes = (config.hidden_size,) if isinstance(config.hidden_size, int) else tuple(config.hidden_size)
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str, device: torch.device | None = None) -> nn.Module:
    if device is None:
        device = torch.device(config.device)
    model = MLP()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
