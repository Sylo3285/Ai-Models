import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from datasets import load_dataset
import config
from model import CNN
from tqdm import tqdm
import numpy as np

# Load dataset
class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, label_map, transform=None):
        self.ds = hf_dataset
        self.label_map = label_map
        self.transform = transform
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        raw_label = item["value"]
        
        # Map label
        label = self.label_map[raw_label]
        
        # Convert to grayscale and tensor
        if image.mode != 'L':
             image = image.convert('L')
             
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Define wrapper class for training subset (for applying augmentation only to train)
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

if __name__ == "__main__":
    # Load dataset from Hugging Face
    print("Loading dataset from Hugging Face...")
    ds = load_dataset("starkdv123/mnist-numbers-0to10000-128x128", split="train")
    
    # Calculate unique classes and create map
    unique_labels = sorted(set(ds["value"]))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")
    
    # Save label map for inference (optional but good practice)
    # Ideally should save this map with the model checkpoint.

    # Define base transforms (conversion to tensor)
    base_transform = transforms.Compose([
        transforms.Resize(config.inp_size),
        transforms.ToTensor(), # Converts to (C, H, W) typically (1, 128, 128) for L mode and scales 0-1
    ])

    full_dataset = HFImageDataset(ds, label_map=label_map, transform=base_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Split indices
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Define Augmentation transforms (applied after base transform)
    train_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    # Apply transforms ONLY to training set
    train_dataset = TransformedSubset(train_dataset, transform=train_transforms)

    # Note on num_workers: Windows sometimes has issues with multiprocessing in simple scripts. 
    # If it hangs, set num_workers=0. Keeping 4 as per original.
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model = CNN(num_classes).to(config.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(config.device), labels.to(config.device)

            output = model(imgs)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(config.device), labels.to(config.device)
                output = model(imgs)
                loss = criterion(output, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        # Step scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'label_map': label_map
    }, "model.pt")
    print("\nTraining complete!")
