import torch
from model import load_model


def run_demo(model_path: str = "model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device=device)

    examples = [
        (1.2, 3.4),
        (10.0, 20.0),
        (-5.0, 2.0),
        (123.45, 67.89),
        (-100.0, 50.0),
        (1000,1000),
        (-250.5, 125.25),
        (0.0, 0.0),
        (3.1415, 2.7182),
        (-1.0, -1.0),
        (10000,1000)
    ]

    xs = torch.tensor(examples, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(xs).squeeze(-1).cpu().numpy()

    for (a, b), p in zip(examples, preds):
        print(f"{a} + {b} = predicted {p:.4f} (expected {a + b:.4f})")


if __name__ == "__main__":
    run_demo()
