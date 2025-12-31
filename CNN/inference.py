import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN
import config

# Prepare dataset (same transforms as training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

# Load model
model = SimpleCNN().to(config.device)
model.load_state_dict(torch.load("mnist_model.pt", map_location=config.device))
model.eval()

# Run inference on 5 random samples
for i, (img, label) in enumerate(test_loader):
    if i == 5:
        break

    img = img.to(config.device)

    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out, dim=1).item()

    print(f"True: {label.item()} → Predicted: {pred}")
