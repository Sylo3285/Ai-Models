import torch.nn as nn

inp_dim = 784
hid_dim = 256
lat_dim = 64


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(inp_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, lat_dim),
        )
    
    def forward(self, x):
        return self.net(x)
    
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(lat_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, inp_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x)
    
class AUTOENCODER(nn.Module):
    def __init__(self):
        super(AUTOENCODER, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    model = AUTOENCODER()
    print(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params}")