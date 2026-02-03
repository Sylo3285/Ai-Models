import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import AUTOENCODER
from visualizer import show_reconstruction
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# Dataset
# --------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # [0,1]
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

# --------------------
# Model
# --------------------
model = AUTOENCODER().to(device)
model.train()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 5

# Grab one fixed sample
fixed_img, _ = train_dataset[0]
fixed_img = fixed_img.view(1, -1).to(device)

# --------------------
# Training loop
# --------------------
for epoch in tqdm(range(epochs), desc="Training Epochs",unit = "epoch"):
    total_loss = 0

    for step, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔴 LIVE VISUALIZATION every N steps
        if step % 1 == 0:
            model.eval()
            with torch.no_grad():
                recon = model(fixed_img)
            show_reconstruction(fixed_img[0], recon[0], epoch + 1, step)
            model.train()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), "autoencoder_mnist.pt")