import torch.nn as nn
import config

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # 1 input channel (grayscale)
        # Layer 1: 1 -> 32 channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128 -> 64x64
            nn.Dropout2d(0.25)
        )

        # Layer 2: 32 -> 64 channels
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
            nn.Dropout2d(0.25)
        )

        # Layer 3: 64 -> 128 channels
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Dropout2d(0.3)
        )

        # Layer 4: 128 -> 256 channels
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            nn.Dropout2d(0.3)
        )

        # Calculate feature map size dynamically based on config.inp_size
        # After 4 MaxPool2d layers, size is reduced by 16 (2^4)
        feature_map_size = config.inp_size[0] // 16
        flattened_size = 256 * feature_map_size * feature_map_size
        
        # Fully connected layers with larger hidden layer
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(-1, 1, config.inp_size[0], config.inp_size[1])  # reshape flat input → image
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x
