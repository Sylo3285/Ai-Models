import torch
import config
from torchvision import transforms
from PIL import Image
from model import CNN

# Load model
checkpoint = torch.load("model.pt", map_location=config.device)
num_classes = checkpoint['num_classes']

model = CNN(num_classes=num_classes).to(config.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define transforms: resize to 128x128 and convert to grayscale
transform = transforms.Compose([
    transforms.Resize(config.inp_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Load and preprocess image
img_path = "data/images/img_0.png"
img = Image.open(img_path)
img_tensor = transform(img).unsqueeze(0).to(config.device)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()

print(f"Predicted class: {predicted_class}")
