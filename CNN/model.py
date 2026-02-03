import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(
            # 28×28×1 → 28×28×32
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14×14×32

            # 14×14×32 → 14×14×64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7×7×64
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.seq(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


img = [
    [1,2,3,4,5,6,7,8,9]
]