import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Input: (B, 3, 32, 32)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # (16, 16, 16)
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 8, 8)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 4, 4)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.flatten_size = 64 * 4 * 4
        self.encoder_fc = nn.Linear(self.flatten_size, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, self.flatten_size)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 8, 8)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16, 16, 16)
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # (3, 32, 32)
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        return self.encoder_fc(x)

    def decode(self, latent):
        x = self.decoder_fc(latent)
        x = x.view(-1, 64, 4, 4)
        return self.decoder_conv(x)

    def forward(self, x):
        return self.decode(self.encode(x))
