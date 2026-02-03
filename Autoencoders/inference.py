import torch
from model import AUTOENCODER
import PIL.Image as Image

device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "untitled.png"

model = AUTOENCODER().to(device)
model.load_state_dict(torch.load("autoencoder_mnist.pt", map_location=device))
model.eval()

def infer(image):
    image = image.view(1, -1).to(device)
    with torch.no_grad():
        reconstructed = model(image)
    return reconstructed.view(28, 28).cpu()

if __name__ == "__main__":
    img = Image.open(image_path).convert("L").resize((28, 28))
    img = torch.tensor([[pixel / 255.0 for pixel in img.getdata()]], dtype=torch.float32)
    infered_image = infer(img)
    infered_image_pil = Image.fromarray((infered_image.numpy() * 255).astype('uint8'))
    infered_image_pil.show()