import torch
from torch.utils.data import DataLoader ,Dataset

from model import SimpleRNN, load_model
from config import device, movement_to_label
from tqdm import tqdm
import random


def train(model,data_loader,criterion,optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "rnn_model.pt")

if __name__ == "__main__":
    model = SimpleRNN().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    #generate a fake dataset to train the model on for 3 days
    data = []
    targets = []

    min_price = 50
    max_price = 100

    for _ in range(1000):
        p1 = random.randint(min_price, max_price)
        movement = random.choice(["up", "down", "same"])

        if movement == "up":
            p2 = p1 + random.randint(1, 5)
        elif movement == "down":
            p2 = p1 - random.randint(1, 5)
        else:
            p2 = p1

        data.append([[p1], [p2]])              # (seq_len=2, input_size=1)
        targets.append(movement_to_label[movement])

        X = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.long)

        #Normalize the input data
        #X /= 100
        dataset = Dataset(X,y)
        dataloader = DataLoader(dataset, batch_size = 500)
        train(model,dataloader,criterion,optimizer)
