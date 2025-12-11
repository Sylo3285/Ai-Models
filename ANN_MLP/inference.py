import torch
from model import ANN
import config
import pandas as pd

def load_model():
    model = ANN().to(config.device)
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    return model

def predict_cost(fuel, toll, time, labor):
    model = load_model()
    inputs = torch.tensor([[fuel, toll, time, labor]], dtype=torch.float32).to(config.device)
    with torch.no_grad():
        prediction = model(inputs)
    return prediction.item()

def main():
    # Example usage
    fuel = 500
    toll = 100
    time = 60
    labor = 1000
    
    predicted_cost = predict_cost(fuel, toll, time, labor)
    print(f"Predicted Cost: {predicted_cost:.2f}")
    cost = (fuel * 1.2) + toll + labor 
    print(f"Actual Cost: {cost:.2f}")

if __name__ == "__main__":
    main()