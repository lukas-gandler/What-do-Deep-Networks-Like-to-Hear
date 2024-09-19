import torch
import torch.nn as nn
from archisound import ArchiSound

class AudioAutoencoder(nn.Module):
    def __init__(self, reduce_output: bool=False):
        super(AudioAutoencoder, self).__init__()
        self.reduce_output = reduce_output

        pretrained_autoencoder = ArchiSound.from_pretrained('autoencoder1d-AT-v1')
        self.encoder = pretrained_autoencoder.autoencoder.encoder
        self.decoder = pretrained_autoencoder.autoencoder.decoder

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)

        return decoded.mean(dim=1, keepdim=True) if self.reduce_output else decoded
