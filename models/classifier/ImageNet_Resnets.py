import torch

from torchvision.models import resnet50, ResNet50_Weights

def load_pretrained_resnet50(device: torch.device='cpu') -> resnet50:
    return resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
