import torch
import torch.nn as nn
from archisound import ArchiSound

class AudioAutoencoder(nn.Module):
    def __init__(self, mono_output: bool=False, keep_channel_dim: bool=True):
        """
        Wrapper around the pre-trained Archisound Audio-Waveform Autoencoder for easier use.
        :param mono_output: If True, the AE reconstruction will be mono, otherwise stereo
        :param keep_channel_dim: If True, the channel dim of the reconstruction will be kept, otherwise the output will be of shape [BATCH x LENGTH] (only possible if the output is mono)
        """

        super(AudioAutoencoder, self).__init__()

        self.mono_output = mono_output
        self.keep_channel_dim = keep_channel_dim

        if not self.mono_output and not self.keep_channel_dim:
            raise RuntimeError(f'Channel dimension can only be discarded if output is set to mono.')

        pretrained_autoencoder = ArchiSound.from_pretrained('autoencoder1d-AT-v1')
        self.encoder = pretrained_autoencoder.autoencoder.encoder
        self.decoder = pretrained_autoencoder.autoencoder.decoder

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input_tensor)
        decoded = self.decoder(encoded)

        return decoded.mean(dim=1, keepdim=self.keep_channel_dim) if self.mono_output else decoded
