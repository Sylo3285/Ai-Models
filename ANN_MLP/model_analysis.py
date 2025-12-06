import model
from config import MAX_LEN, OUTPUT_DIM
model = model.ANN(
    input_dim=MAX_LEN,
    num_classes=OUTPUT_DIM
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters in the model: {total_params}")