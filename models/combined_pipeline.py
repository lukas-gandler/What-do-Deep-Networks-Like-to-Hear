import torch
import torch.nn as nn

from typing import Optional

from torch.nn.modules.module import T


def _freeze_weights(network: nn.Module):
    for param in network.parameters():
        param.requires_grad = False


class CombinedPipeline(nn.Module):
    def __init__(self, autoencoder: nn.Module, classifier: nn.Module, finetune_decoder: bool=True, finetune_encoder: bool=True, post_ae_transform: Optional[nn.Module]=None):
        super(CombinedPipeline, self).__init__()

        self.post_ae_transform = post_ae_transform
        if self.post_ae_transform is not None:
            self.post_ae_transform.eval()

        self.autoencoder = autoencoder

        if not finetune_encoder:
            print(f'=> Freezing encoder')
            encoder_network = self.autoencoder.module.encoder if isinstance(self.autoencoder, nn.DataParallel) else self.autoencoder.encoder
            _freeze_weights(encoder_network)

        if not finetune_decoder:
            print(f'=> Freezing decoder')
            decoder_network = self.autoencoder.module.decoder if isinstance(self.autoencoder, nn.DataParallel) else self.autoencoder.decoder
            _freeze_weights(decoder_network)

        self.classifier = classifier
        _freeze_weights(self.classifier)
        self.classifier.eval()  # freeze also layers with state such as BatchNorm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        reconstructed_input = self.autoencoder(input)
        reconstructed_input = self.post_ae_transform(reconstructed_input) if self.post_ae_transform is not None else reconstructed_input

        predicted_labels = self.classifier(reconstructed_input)
        return predicted_labels

    def train(self: T, mode: bool = True) -> T:
        self.autoencoder.train(mode)
        return self

    def eval(self: T) -> T:
        self.autoencoder.eval()
        return self
