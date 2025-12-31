import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SimpleCNN
import config

# Dataset + transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

# Init model, loss, optimizer
model = SimpleCNN().to(config.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Training loop
for epoch in range(1, config.epochs + 1):
    model.train()
    loss_sum = 0

    for images, labels in train_loader:
        images, labels = images.to(config.device), labels.to(config.device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    # Eval accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch}/{config.epochs} | Loss: {loss_sum/len(train_loader):.4f} | Accuracy: {acc:.2f}% 😼✨")

# Save model
torch.save(model.state_dict(), "mnist_model.pt")
print("Training complete! Saved as mnist_model.pt ⚡🔥😼")