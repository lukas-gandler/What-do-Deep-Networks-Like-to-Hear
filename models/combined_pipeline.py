import torch
import torch.nn as nn


def _freeze_weights(network: nn.Module):
    for param in network.parameters():
        param.requires_grad = False


class CombinedPipeline(nn.Module):
    def __init__(self, autoencoder: nn.Module, classifier: nn.Module, finetune_encoder: bool=False):
        super(CombinedPipeline, self).__init__()

        self.autoencoder = autoencoder
        if not finetune_encoder:
            encoder_network = self.autoencoder.module.encoder if isinstance(self.autoencoder, nn.DataParallel) else self.autoencoder.encoder
            _freeze_weights(encoder_network)

        self.classifier = classifier
        _freeze_weights(self.classifier)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        reconstructed_input = self.autoencoder(input)
        predicted_labels = self.classifier(reconstructed_input)
        return predicted_labels
