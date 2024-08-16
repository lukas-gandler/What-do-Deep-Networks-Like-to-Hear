"""
Simple implementation of an autoencoder for MNIST digits for testing purposes.
based on the GitHub repository by SuchismitaSahu1993 (https://github.com/SuchismitaSahu1993/Autoencoder-on-MNIST-in-Pytorch/blob/master/Autoencoder.py)
"""

import torch
import torch.nn as nn

class MNIST_Autoencoder(nn.Module):
    def __init__(self):
        super(MNIST_Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16, kernel_size=5),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, kernel_size=5),
            # nn.ReLU(True),
            # nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
