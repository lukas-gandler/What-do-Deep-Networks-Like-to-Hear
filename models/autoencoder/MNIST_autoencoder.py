"""
Simple implementation of an autoencoder for MNIST digits for testing purposes.
based on the post by Marton Trencseni (https://bytepawn.com/building-a-pytorch-autoencoder-for-mnist-digits.html)
"""

import torch
import torch.nn as nn

class MNIST_Autoencoder(nn.Module):
    def __init__(self):
        super(MNIST_Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,4, kernel_size=5),
            nn.ReLU(True),

            nn.Conv2d(4,8,kernel_size=5),
            nn.ReLU(True),

            nn.Flatten(),
            nn.Linear(3200, 10),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 400),
            nn.ReLU(True),
            
            nn.Linear(400, 4000),
            nn.ReLU(True),
            
            nn.Unflatten(1, (10, 20, 20)),
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
